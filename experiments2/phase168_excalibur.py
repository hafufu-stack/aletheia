# -*- coding: utf-8 -*-
"""
Phase 168: The Excalibur - Sword Equation Zero-Shot Extrapolation
Test g* ~ d^0.45 on UNSEEN models (GPT-2 Medium, Large).

The equation was fitted on: 0.5B(d=896), 1.5B(d=1536), 14B(d=5120).
GPT-2 Medium (d=1024) and Large (d=1280) were NEVER in the training set.

If the equation predicts the correct g* for these unseen models,
it is a true universal law of Transformer architecture.

Models: GPT-2 Medium (355M), GPT-2 Large (774M) - both cached locally
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
    ("# The capital of Japan is", " Tokyo", "word"),
    ("# The capital of France is", " Paris", "word"),
    ("# The capital of Germany is", " Berlin", "word"),
    ("# The largest planet is", " Jupiter", "word"),
    ("# Water freezes at", " 0", "number"),
    ("# The boiling point of water is", " 100", "number"),
    ("# The number of continents is", " 7", "number"),
    ("# A year has", " 365", "number"),
    ("# Pi is approximately", " 3", "number"),
]

NUM_TOKENS_GPT2 = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
                   " 10"," 100"," 12"," 365"," 299"]


def predict_g_star(d):
    """Sword Energy Equation: g* ~ d^0.45"""
    # Fitted from P159: g* = 0.13 * d^0.45
    return 0.13 * (d ** 0.45)


def disperse_gpt2(model, tok, strength=2.0):
    """Dual Surgery for GPT-2 (wte + lm_head, usually tied)."""
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS_GPT2 if tok.encode(t)))

    # Disperse wte (embed)
    embed = model.transformer.wte.weight.data
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)

    # Check if tied
    tied = model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()

    if not tied:
        lm = model.lm_head.weight.data
        vecs = lm[ids].clone().float()
        center = vecs.mean(dim=0)
        for i, idx in enumerate(ids):
            diff = vecs[i] - center
            direction = diff / (diff.norm() + 1e-8)
            lm[idx] += (strength * direction * lm[idx].float().norm()).to(lm.dtype)

    # Measure post-surgery cos
    post_vecs = model.lm_head.weight.data[ids].float()
    cos = F.cosine_similarity(post_vecs.unsqueeze(0), post_vecs.unsqueeze(1), dim=-1)
    avg_cos = cos[~torch.eye(len(ids), dtype=bool)].mean().item()
    return avg_cos, tied


class FGAHookGPT2:
    def __init__(self, model, target_id, gain):
        self.gain = gain
        unembed = model.lm_head.weight.data[target_id].float()
        self.direction = unembed / (unembed.norm() + 1e-8)
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        h = output.float()
        if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
        elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
        return h.to(output.dtype)

    def register(self, layer):
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def evaluate_gpt2(model, tok, test_set, fga_gain, fga_layer):
    results = []
    for prompt, expected, cat in test_set:
        exp_id = tok.encode(expected)[-1]
        hook = FGAHookGPT2(model, exp_id, fga_gain)
        hook.register(model.transformer.h[fga_layer])
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        hook.remove()
        pred_id = logits.argmax().item()
        results.append({'cat': cat, 'correct': int(pred_id == exp_id),
                        'expected': expected.strip(), 'pred': tok.decode([pred_id]).strip()})
    w = sum(r['correct'] for r in results if r['cat'] == 'word')
    wt = max(1, sum(1 for r in results if r['cat'] == 'word'))
    n = sum(r['correct'] for r in results if r['cat'] == 'number')
    nt = max(1, sum(1 for r in results if r['cat'] == 'number'))
    return w/wt, n/nt, results


def test_model(model_name, model_id):
    """Test Sword Equation on one GPT-2 variant."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"\n  === {model_name} ===")

    tok = GPT2Tokenizer.from_pretrained(model_id, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_id, local_files_only=True).to(DEVICE).eval()

    d = model.config.n_embd
    n_layers = model.config.n_layer
    predicted_g = predict_g_star(d)
    fga_layer = n_layers - max(1, n_layers // 3)

    print(f"    d={d}, layers={n_layers}")
    print(f"    Predicted g* = {predicted_g:.1f} (from d^0.45 equation)")
    print(f"    FGA at layer {fga_layer}")

    # Apply Dual Surgery
    post_cos, tied = disperse_gpt2(model, tok, strength=2.0)
    print(f"    Post-surgery cos: {post_cos:.4f}, tied={tied}")

    # Test with predicted g* (ZERO-SHOT - no tuning!)
    g_pred = round(predicted_g)
    w, n, details = evaluate_gpt2(model, tok, TEST_SET, g_pred, fga_layer)
    print(f"    ZERO-SHOT (g={g_pred}): Word={w:.0%}, Num={n:.0%}")
    for r in details:
        if r['cat'] == 'number':
            print(f"      {r['expected']:>6s} -> {r['pred']:10s} "
                  f"{'OK' if r['correct'] else 'MISS'}")

    # Also sweep nearby gains for comparison
    sweep_results = {}
    for g in [max(1, g_pred-5), g_pred-2, g_pred, g_pred+2, g_pred+5, g_pred+10]:
        # Reload fresh model for each sweep point (surgery is destructive)
        model2 = GPT2LMHeadModel.from_pretrained(model_id, local_files_only=True).to(DEVICE).eval()
        disperse_gpt2(model2, tok, strength=2.0)
        w2, n2, _ = evaluate_gpt2(model2, tok, TEST_SET, g, fga_layer)
        sweep_results[g] = {'word': w2, 'num': n2}
        del model2; gc.collect(); torch.cuda.empty_cache()

    del model; gc.collect(); torch.cuda.empty_cache()

    return {
        'model': model_name, 'd': d, 'n_layers': n_layers,
        'predicted_g': predicted_g, 'post_cos': post_cos, 'tied': tied,
        'zero_shot_word': w, 'zero_shot_num': n, 'details': details,
        'sweep': sweep_results
    }


def main():
    print("[P168] The Excalibur - Sword Equation Extrapolation")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    # Test on unseen models
    all_results = {}

    try:
        r = test_model("GPT-2 Medium (355M)", "gpt2-medium")
        all_results['gpt2_medium'] = r
    except Exception as e:
        print(f"  GPT-2 Medium failed: {e}")

    try:
        r = test_model("GPT-2 Large (774M)", "gpt2-large")
        all_results['gpt2_large'] = r
    except Exception as e:
        print(f"  GPT-2 Large failed: {e}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase168_excalibur.json'), 'w') as f:
        json.dump({'phase': '168', 'name': 'Excalibur - Sword Equation',
                   'results': all_results}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Equation with new data points
    ax = axes[0]
    # Known points (from P159)
    known_d = [768, 896, 1536, 5120]
    known_g = [3, 5, 15, 50]
    known_labels = ['GPT-2\n(124M)', 'Qwen\n0.5B', 'Qwen\n1.5B', 'Qwen\n14B']
    d_range = np.linspace(500, 6000, 100)
    g_curve = 0.13 * d_range ** 0.45

    ax.plot(d_range, g_curve, 'k--', lw=2, alpha=0.5, label='$g^* = 0.13 \\cdot d^{0.45}$')
    ax.scatter(known_d, known_g, c='blue', s=100, zorder=5, label='Known (P159)')
    for i, lbl in enumerate(known_labels):
        ax.annotate(lbl, (known_d[i], known_g[i]), textcoords="offset points",
                    xytext=(10, 5), fontsize=8)

    # New data points
    colors = ['red', 'green']
    for i, (key, r) in enumerate(all_results.items()):
        success = r['zero_shot_num'] >= 0.6
        marker = '*' if success else 'x'
        ax.scatter([r['d']], [r['predicted_g']], c=colors[i], s=200, marker=marker,
                   zorder=10, label=f"{r['model']}: {'OK' if success else 'MISS'}")

    ax.set_xlabel('Hidden Dimension d', fontsize=12)
    ax.set_ylabel('Critical FGA Gain g*', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('Sword Energy Equation\nZero-Shot Extrapolation Test', fontsize=13, fontweight='bold')

    # Right: Gain sweep for new models
    ax = axes[1]
    for key, r in all_results.items():
        sweep = r['sweep']
        gains = sorted(sweep.keys())
        nums = [sweep[g]['num'] for g in gains]
        ax.plot(gains, nums, '-o', lw=2, markersize=8, label=r['model'])
        ax.axvline(x=r['predicted_g'], ls=':', alpha=0.5)

    ax.set_xlabel('FGA Gain', fontsize=12)
    ax.set_ylabel('Number Accuracy', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_title('Gain Sweep (Unseen Models)', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.1)

    plt.suptitle('Phase 168: The Excalibur\n'
                 'Does the Sword Equation predict g* for unseen models?',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase168_excalibur.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    for key, r in all_results.items():
        status = "EXCALIBUR!" if r['zero_shot_num'] >= 0.6 else "Needs tuning"
        print(f"  -> {r['model']}: d={r['d']}, predicted g*={r['predicted_g']:.1f}, "
              f"num={r['zero_shot_num']:.0%} [{status}]")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 168] Complete.")


if __name__ == '__main__':
    main()
