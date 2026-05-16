# -*- coding: utf-8 -*-
"""
Phase 152: The Quantization Artifact Test
Is 14B's failure caused by 4-bit quantization, not scale?

P149: 1.5B(fp16) = 100%, 14B(4-bit) = 0%.
Critical question: Does 1.5B(4-bit) still work?
If YES -> quantization is fine, 14B failure is scale-related.
If NO -> quantization destroys surgery precision.

Model: Qwen2.5-1.5B (GPU, 4-bit vs fp16)
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


def evaluate(model, tok, test_set, code_mode=True, fga_gain=20):
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - 4
    results = []
    for prompt, expected, cat in test_set:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[-1]
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


def test_config(model_id, tok, dtype_or_bnb, label, strength=2.0, fga_gain=20):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    print(f"\n  === {label} ===")

    if isinstance(dtype_or_bnb, BitsAndBytesConfig):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, quantization_config=dtype_or_bnb,
            device_map="auto", torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=dtype_or_bnb).eval().to(DEVICE)

    # Measure embedding norms before surgery
    embed = model.model.embed_tokens.weight.data
    num_ids = [tok.encode(f" {d}")[-1] for d in range(10)]
    pre_norms = embed[num_ids].float().norm(dim=-1).mean().item()

    disperse_embeddings(model, tok, strength=strength)

    post_norms = embed[num_ids].float().norm(dim=-1).mean().item()
    # L2 between first two num tokens
    v0 = embed[num_ids[0]].float()
    v1 = embed[num_ids[1]].float()
    l2 = (v0 - v1).norm().item()
    cos = F.cosine_similarity(v0.unsqueeze(0), v1.unsqueeze(0)).item()

    print(f"    Pre-norm: {pre_norms:.3f}, Post-norm: {post_norms:.3f}")
    print(f"    L2(0,1): {l2:.3f}, Cos(0,1): {cos:.4f}")

    w, n, details = evaluate(model, tok, TEST_SET, code_mode=True, fga_gain=fga_gain)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    for d in details:
        if d['cat'] == 'number':
            print(f"      {d['expected']:8s} -> {d['pred']:10s} {'OK' if d['correct'] else 'MISS'}")

    result = {
        'word': w, 'num': n, 'pre_norm': pre_norms, 'post_norm': post_norms,
        'l2': l2, 'cos': cos, 'strength': strength, 'fga_gain': fga_gain
    }
    del model; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    print("[P152] The Quantization Artifact Test")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoTokenizer, BitsAndBytesConfig
    model_id = 'Qwen/Qwen2.5-1.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    bnb_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)
    bnb_8bit = BitsAndBytesConfig(load_in_8bit=True)

    configs = {}

    # A: fp16 (reference - should be 100% from P144)
    configs['A_fp16'] = test_config(model_id, tok, torch.float16, '1.5B fp16')

    # B: 8-bit
    configs['B_8bit'] = test_config(model_id, tok, bnb_8bit, '1.5B 8-bit')

    # C: 4-bit (the suspect)
    configs['C_4bit'] = test_config(model_id, tok, bnb_4bit, '1.5B 4-bit')

    # D: 4-bit with stronger surgery (s=5)
    configs['D_4bit_s5'] = test_config(model_id, tok, bnb_4bit, '1.5B 4-bit s=5',
                                        strength=5.0)

    # E: 4-bit with stronger FGA (g=50)
    configs['E_4bit_g50'] = test_config(model_id, tok, bnb_4bit, '1.5B 4-bit g=50',
                                         fga_gain=50)

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase152_quantization.json'), 'w') as f:
        json.dump({'phase': '152', 'name': 'Quantization Artifact Test',
                   'configs': configs}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    names = ['fp16', '8-bit', '4-bit', '4-bit\ns=5', '4-bit\ng=50']
    keys = list(configs.keys())
    num_vals = [configs[k]['num'] for k in keys]
    word_vals = [configs[k]['word'] for k in keys]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x-w/2, word_vals, w, label='Word', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, num_vals, w, label='Number', color='#e74c3c', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w/2, word_vals[i]+0.02, f'{word_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
        ax.text(x[i]+w/2, num_vals[i]+0.02, f'{num_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=11); ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Phase 152: Does Quantization Break Surgery+S&S?\n1.5B model at different precisions',
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase152_quantization.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    fp16_n = configs['A_fp16']['num']
    bit4_n = configs['C_4bit']['num']
    if fp16_n > 0.5 and bit4_n < 0.2:
        print(f"  -> QUANTIZATION IS THE CULPRIT! fp16={fp16_n:.0%}, 4-bit={bit4_n:.0%}")
        print(f"     14B's failure is likely a quantization artifact!")
    elif bit4_n > 0.5:
        print(f"  -> 4-bit works fine ({bit4_n:.0%}). 14B failure is scale-related.")
    else:
        print(f"  -> Both struggle: fp16={fp16_n:.0%}, 4-bit={bit4_n:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 152] Complete.")

if __name__ == '__main__':
    main()
