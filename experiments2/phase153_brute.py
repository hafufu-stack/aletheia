# -*- coding: utf-8 -*-
"""
Phase 153: 14B Brute Force Parameter Sweep
Can we break 14B with extreme surgery/FGA parameters?

P149: 14B(4-bit, s=2, g=20) = 0%.
Maybe the parameters are just too weak for 14B's scale.
14B has d=5120 (vs 896 for 0.5B) -> need proportionally stronger intervention.

Model: Qwen2.5-14B (GPU, 4-bit)
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
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("The speed of light is approximately", " 299", "number"),
    ("A year has", " 365", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
    ("The capital of Japan is", " Tokyo", "word"),
    ("The capital of France is", " Paris", "word"),
]


def disperse_embeddings(model, tok, strength=2.0):
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


class MultiFGAHook:
    """Apply FGA at multiple layers simultaneously."""
    def __init__(self, model, target_token_id, gain, layers):
        unembed = model.lm_head.weight.data[target_token_id].float()
        self.direction = unembed / (unembed.norm() + 1e-8)
        self.gain_per_layer = gain / len(layers)  # distribute gain
        self.handles = []
        for layer_idx in layers:
            def make_hook(g):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].float()
                        if h.dim() == 3: h[:, -1, :] += g * self.direction.to(h.device)
                        elif h.dim() == 2: h[-1, :] += g * self.direction.to(h.device)
                        return (h.to(output[0].dtype),) + output[1:]
                    else:
                        h = output.float()
                        if h.dim() == 3: h[:, -1, :] += g * self.direction.to(h.device)
                        elif h.dim() == 2: h[-1, :] += g * self.direction.to(h.device)
                        return h.to(output.dtype)
                return hook_fn
            handle = model.model.layers[layer_idx].register_forward_hook(
                make_hook(self.gain_per_layer))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()


def evaluate(model, tok, test_set, code_mode=True, fga_gain=20,
             multi_fga_layers=None):
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - 4
    results = []
    for prompt, expected, cat in test_set:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[-1]

        if multi_fga_layers:
            hook = MultiFGAHook(model, exp_id, fga_gain, multi_fga_layers)
        else:
            hook = FGAHook(model, exp_id, fga_gain)
            hook.register(model, fga_layer)

        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        hook.remove()
        pred_id = logits.argmax().item()
        results.append({'cat': cat, 'correct': int(pred_id == exp_id),
                        'expected': expected, 'pred': tok.decode([pred_id])})
    w = sum(r['correct'] for r in results if r['cat'] == 'word')
    wt = max(1, sum(1 for r in results if r['cat'] == 'word'))
    n = sum(r['correct'] for r in results if r['cat'] == 'number')
    nt = max(1, sum(1 for r in results if r['cat'] == 'number'))
    return w/wt, n/nt, results


def main():
    print("[P153] 14B Brute Force Parameter Sweep")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = 'Qwen/Qwen2.5-14B'
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as e:
        print(f"  14B not available: {e}")
        # Fallback to 1.5B for testing the sweep methodology
        model_id = 'Qwen/Qwen2.5-1.5B'
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        print(f"  Falling back to 1.5B for parameter sweep demonstration")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    bnb_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)

    sweep_results = []

    # Parameter sweep: (surgery_strength, fga_gain, multi_layer, label)
    sweep_configs = [
        (2.0, 20, False, 's=2 g=20'),
        (5.0, 20, False, 's=5 g=20'),
        (10.0, 20, False, 's=10 g=20'),
        (2.0, 50, False, 's=2 g=50'),
        (2.0, 100, False, 's=2 g=100'),
        (2.0, 200, False, 's=2 g=200'),
        (5.0, 100, False, 's=5 g=100'),
        (10.0, 100, False, 's=10 g=100'),
        (2.0, 100, True, 's=2 g=100 multi'),
        (5.0, 100, True, 's=5 g=100 multi'),
    ]

    for strength, gain, multi_layer, label in sweep_configs:
        print(f"\n  === {label} ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, quantization_config=bnb_4bit,
            device_map="auto", torch_dtype=torch.float16)

        disperse_embeddings(model, tok, strength=strength)

        n_layers = model.config.num_hidden_layers
        if multi_layer:
            # FGA at last 8 layers
            layers = list(range(n_layers-8, n_layers))
            w, n, details = evaluate(model, tok, TEST_SET, fga_gain=gain,
                                    multi_fga_layers=layers)
        else:
            w, n, details = evaluate(model, tok, TEST_SET, fga_gain=gain)

        print(f"    Word: {w:.0%}, Num: {n:.0%}")
        for d in details:
            if d['cat'] == 'number':
                print(f"      {d['expected']:8s} -> {d['pred']:10s} {'OK' if d['correct'] else 'MISS'}")

        sweep_results.append({
            'label': label, 'strength': strength, 'gain': gain,
            'multi_layer': multi_layer, 'word': w, 'num': n
        })
        del model; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase153_brute.json'), 'w') as f:
        json.dump({'phase': '153', 'name': '14B Brute Force',
                   'model': model_id, 'results': sweep_results}, f, indent=2, default=str)

    # Plot: heatmap of surgery_strength x fga_gain
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    labels = [r['label'] for r in sweep_results]
    num_accs = [r['num'] for r in sweep_results]
    colors = ['#2ecc71' if a > 0.5 else '#f39c12' if a > 0 else '#e74c3c' for a in num_accs]
    bars = ax.bar(range(len(labels)), num_accs, color=colors, alpha=0.8, edgecolor='black', lw=1)
    for i, v in enumerate(num_accs):
        ax.text(i, v + 0.02, f'{v:.0%}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number Accuracy', fontsize=12)
    ax.set_title(f'Phase 153: Parameter Sweep on {model_id.split("/")[-1]}\n'
                f'(Green=success, Red=failure)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase153_brute.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    best = max(sweep_results, key=lambda r: r['num'])
    if best['num'] > 0:
        print(f"  -> BREAKTHROUGH: {best['label']} achieves {best['num']:.0%}!")
    else:
        print(f"  -> {model_id.split('/')[-1]} is immune to all parameter combinations")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 153] Complete.")

if __name__ == '__main__':
    main()
