# -*- coding: utf-8 -*-
"""
Phase 164: The Cosine Critical Point
P162 showed: cos > 0.5 -> 0%, cos < 0.13 -> 100%.
Find the EXACT threshold with fine-grained sweep.

This is a physical constant of Transformer hallucination.

Model: Qwen2.5-0.5B (GPU)
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
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
    ("A year has", " 365", "number"),
    ("The number of days in a week is", " 7", "number"),
]


def dual_surgery(model, tok, strength):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365"]
    ids = list(set(tok.encode(t)[-1] for t in num_tokens))
    embed = model.model.embed_tokens.weight.data
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()
    lm = model.lm_head.weight.data
    vecs_lm = lm[ids].clone().float()
    center_lm = vecs_lm.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs_lm[i] - center_lm
        direction = diff / (diff.norm() + 1e-8)
        lm[idx] += strength * direction * lm[idx].norm()
    # Measure cos
    post_lm = lm[ids].float()
    cos = F.cosine_similarity(post_lm.unsqueeze(0), post_lm.unsqueeze(1), dim=-1)
    mask = ~torch.eye(len(ids), dtype=bool)
    return cos[mask].mean().item()


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


def evaluate(model, tok, fga_gain=20):
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - 4
    correct = 0
    for prompt, expected, cat in TEST_SET:
        text = f"# {prompt}"
        exp_id = tok.encode(expected)[-1]
        hook = FGAHook(model, exp_id, fga_gain)
        hook.register(model, fga_layer)
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        hook.remove()
        if logits.argmax().item() == exp_id: correct += 1
    return correct / len(TEST_SET)


def main():
    print("[P164] The Cosine Critical Point")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Fine-grained sweep: focus on the transition zone
    # P162 showed: s=-1.0 (cos=0.72) = 0%, s=-2.0 (cos=0.06) = 100%
    # Also s=+0.5 (cos=0.13) = 100%, s=0 (cos=0.73) = 0%
    strengths = np.concatenate([
        np.arange(-2.5, -0.5, 0.1),
        np.arange(-0.5, 0.0, 0.05),
        [0.0],
        np.arange(0.05, 0.6, 0.05),
        np.arange(0.6, 2.5, 0.1),
    ])
    strengths = sorted(set(np.round(strengths, 3)))

    results = []
    for s in strengths:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
        if s != 0:
            cos_val = dual_surgery(model, tok, strength=s)
        else:
            lm = model.lm_head.weight.data.float()
            num_ids = [tok.encode(f" {d}")[-1] for d in range(10)]
            c = F.cosine_similarity(lm[num_ids].unsqueeze(0), lm[num_ids].unsqueeze(1), dim=-1)
            cos_val = c[~torch.eye(10, dtype=bool)].mean().item()
        acc = evaluate(model, tok, fga_gain=20)
        results.append({'strength': float(s), 'cos': cos_val, 'acc': acc})
        if acc > 0 or abs(s) < 0.6:
            print(f"    s={s:+6.3f}: cos={cos_val:.4f} acc={acc:.0%}")
        del model; gc.collect(); torch.cuda.empty_cache()

    with open(os.path.join(RESULTS_DIR, 'phase164_threshold.json'), 'w') as f:
        json.dump({'phase': '164', 'name': 'Cosine Critical Point',
                   'results': results}, f, indent=2, default=str)

    # Find exact threshold
    sorted_by_cos = sorted(results, key=lambda r: r['cos'])
    threshold_low = None
    threshold_high = None
    for i in range(len(sorted_by_cos) - 1):
        if sorted_by_cos[i]['acc'] > 0.5 and sorted_by_cos[i+1]['acc'] < 0.5:
            threshold_low = sorted_by_cos[i]['cos']
            threshold_high = sorted_by_cos[i+1]['cos']
            break

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    cos_vals = [r['cos'] for r in sorted_by_cos]
    acc_vals = [r['acc'] for r in sorted_by_cos]
    ax.plot(cos_vals, acc_vals, 'r-o', lw=2, markersize=4, alpha=0.8)
    ax.fill_between(cos_vals, 0, acc_vals, alpha=0.15, color='red')
    if threshold_low is not None and threshold_high is not None:
        tc = (threshold_low + threshold_high) / 2
        ax.axvline(x=tc, color='blue', lw=3, ls='--', alpha=0.8,
                  label=f'Critical Point: cos = {tc:.3f}')
        ax.axvspan(threshold_low, threshold_high, alpha=0.3, color='yellow',
                  label=f'Transition zone: [{threshold_low:.3f}, {threshold_high:.3f}]')
    ax.set_xlabel('LM_head Cosine Similarity (post-surgery)', fontsize=13)
    ax.set_ylabel('Number Accuracy (with S&S)', fontsize=13)
    ax.set_title('Phase 164: The Cosine Critical Point\n'
                'Exact threshold where hallucination -> precision',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='center left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase164_threshold.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    if threshold_low is not None:
        print(f"  -> CRITICAL POINT: cos = {(threshold_low+threshold_high)/2:.3f}")
        print(f"     Transition zone: [{threshold_low:.3f}, {threshold_high:.3f}]")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 164] Complete.")

if __name__ == '__main__':
    main()
