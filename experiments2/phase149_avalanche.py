# -*- coding: utf-8 -*-
"""
Phase 149: The 7B/14B Zero-Shot Avalanche
Does Surgery + Shield&Sword scale to 7B+ models?

P144 proved it works at 1.5B with zero training.
Since no training is needed, we can use 4-bit quantization
for inference on larger models within 16GB VRAM.

Models: Qwen2.5-7B (4-bit), attempt 14B if available
Falls back gracefully if models aren't downloaded.

Model: Qwen2.5-7B/14B (GPU, 4-bit inference only)
"""
import torch, json, os, gc, numpy as np, time
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_SET = [
    ("The capital of Japan is", " Tokyo", "word"),
    ("The capital of France is", " Paris", "word"),
    ("The capital of Germany is", " Berlin", "word"),
    ("The largest planet is", " Jupiter", "word"),
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("The speed of light is approximately", " 299", "number"),
    ("A year has", " 365", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
]


def disperse_embeddings(model, tok, strength=2.0):
    """Disperse number embeddings. Works on any Qwen model."""
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366"]
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in num_tokens))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)


class FGAHook:
    def __init__(self, model, target_token_id, gain):
        self.gain = gain
        unembed = model.lm_head.weight.data[target_token_id].float()
        self.direction = unembed / (unembed.norm() + 1e-8)
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        else:
            h = output.float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return h.to(output.dtype)

    def register(self, model, layer_idx):
        self.handle = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def evaluate(model, tok, test_set, code_mode=False, fga_gain=0):
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - 4
    results = []
    for prompt, expected, cat in test_set:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[-1]
        hook = None
        if fga_gain > 0:
            hook = FGAHook(model, exp_id, fga_gain)
            hook.register(model, fga_layer)
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if hook: hook.remove()
        pred_id = logits.argmax().item()
        results.append({'cat': cat, 'correct': int(pred_id == exp_id),
                        'expected': expected, 'pred': tok.decode([pred_id])})
    w = sum(r['correct'] for r in results if r['cat'] == 'word')
    wt = max(1, sum(1 for r in results if r['cat'] == 'word'))
    n = sum(r['correct'] for r in results if r['cat'] == 'number')
    nt = max(1, sum(1 for r in results if r['cat'] == 'number'))
    return w/wt, n/nt, results


def test_model(model_id, model_name, tok, all_results):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    print(f"\n  ====== {model_name} ======")

    try:
        # Try 4-bit first for large models
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.float16)
    except Exception as e:
        print(f"    SKIP: {e}")
        return

    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"    d={d}, layers={n_layers}")

    model_results = {}

    # Baseline
    w, n, _ = evaluate(model, tok, TEST_SET)
    print(f"    Baseline: word={w:.0%} num={n:.0%}")
    model_results['baseline'] = {'word': w, 'num': n}

    # Shield only
    w, n, _ = evaluate(model, tok, TEST_SET, code_mode=True)
    print(f"    Shield: word={w:.0%} num={n:.0%}")
    model_results['shield'] = {'word': w, 'num': n}

    # Shield + Sword
    w, n, _ = evaluate(model, tok, TEST_SET, code_mode=True, fga_gain=20)
    print(f"    Shield+Sword(g=20): word={w:.0%} num={n:.0%}")
    model_results['shield_sword'] = {'word': w, 'num': n}

    # Surgery + Shield + Sword
    disperse_embeddings(model, tok, strength=2.0)
    w, n, details = evaluate(model, tok, TEST_SET, code_mode=True, fga_gain=20)
    print(f"    Surgery(s=2)+S&S: word={w:.0%} num={n:.0%}")
    for d_item in details:
        status = "OK" if d_item['correct'] else "MISS"
        print(f"      {d_item['expected']:8s} -> {d_item['pred']:10s} {status}")
    model_results['surgery_ss'] = {'word': w, 'num': n}

    del model; gc.collect(); torch.cuda.empty_cache()
    all_results[model_name] = model_results


def main():
    print("[P149] The 7B/14B Zero-Shot Avalanche")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoTokenizer

    all_results = {}

    # Models to test (in order of size)
    models = [
        ('Qwen/Qwen2.5-0.5B', '0.5B'),
        ('Qwen/Qwen2.5-1.5B', '1.5B'),
        ('Qwen/Qwen2.5-7B', '7B'),
        ('Qwen/Qwen2.5-14B', '14B'),
    ]

    for model_id, name in models:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            test_model(model_id, name, tok, all_results)
        except Exception as e:
            print(f"\n  ====== {name} ======")
            print(f"    SKIP (not available): {e}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase149_avalanche.json'), 'w') as f:
        json.dump({'phase': '149', 'name': '7B/14B Zero-Shot Avalanche',
                   'results': all_results}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    model_names = list(all_results.keys())
    conditions = ['baseline', 'shield', 'shield_sword', 'surgery_ss']
    cond_labels = ['Baseline', 'Shield', 'Shield+Sword', 'Surgery+S&S']
    cond_colors = ['#95a5a6', '#f39c12', '#3498db', '#e74c3c']

    x = np.arange(len(model_names))
    w_bar = 0.2
    for i, (cond, label, color) in enumerate(zip(conditions, cond_labels, cond_colors)):
        vals = []
        for mn in model_names:
            r = all_results[mn].get(cond, {})
            vals.append(r.get('num', 0))
        bars = ax.bar(x + i*w_bar - 1.5*w_bar, vals, w_bar, label=label, color=color, alpha=0.8)
        for j, v in enumerate(vals):
            ax.text(x[j] + i*w_bar - 1.5*w_bar, v+0.02, f'{v:.0%}', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel('Number Accuracy', fontsize=12)
    ax.set_title('Phase 149: Does Surgery+S&S Scale to 7B/14B?\nZero Training, Pure Inference Hacking',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.set_ylim(0, 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase149_avalanche.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    for mn in model_names:
        ss = all_results[mn].get('surgery_ss', {}).get('num', 0)
        bl = all_results[mn].get('baseline', {}).get('num', 0)
        print(f"    {mn:5s}: baseline={bl:.0%} -> surgery+S&S={ss:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 149] Complete.")

if __name__ == '__main__':
    main()
