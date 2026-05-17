# -*- coding: utf-8 -*-
"""
Phase 179: Surgery Strength Spectrum
We've always used strength=2.0 for Surgery. Is this optimal?

Too weak: numbers stay clustered, FGA can't differentiate
Too strong: destroys all learned representations
Goldilocks: just enough to separate numbers while preserving knowledge

Sweep strength from 0.5 to 5.0 on fact accuracy + cosine distance.

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

FACT_TEST = [
    ("# The capital of Japan is", " Tokyo"),
    ("# The capital of France is", " Paris"),
    ("# Water freezes at", " 0"),
    ("# A year has", " 365"),
    ("# The number of continents is", " 7"),
    ("# The largest planet is", " Jupiter"),
    ("# Pi is approximately", " 3"),
    ("# The boiling point of water is", " 100"),
]

ARITH_TEST = [
    ("# 1 + 1 =", " 2"), ("# 3 + 4 =", " 7"), ("# 5 + 5 =", " 10"),
    ("# 8 + 1 =", " 9"), ("# 6 + 3 =", " 9"), ("# 4 + 4 =", " 8"),
]

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"]


def apply_surgery_and_measure(model, tok, strength):
    """Apply surgery and return avg cosine similarity of number tokens."""
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)

    # Measure post-surgery cosine
    post_vecs = embed[ids].float()
    cos = F.cosine_similarity(post_vecs.unsqueeze(0), post_vecs.unsqueeze(1), dim=-1)
    mask = ~torch.eye(len(ids), dtype=bool)
    avg_cos = cos[mask].mean().item()
    return avg_cos


def evaluate_with_fga(model, tok, test_data, fga_gain=5):
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - max(1, n_layers // 4)
    correct = 0
    for prompt, expected in test_data:
        exp_id = tok.encode(expected)[-1]
        unembed = model.lm_head.weight.data[exp_id].float()
        direction = unembed / (unembed.norm() + 1e-8)
        def make_hook(d, g):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].float()
                    if h.dim() == 3: h[:, -1, :] += g * d.to(h.device)
                    return (h.to(output[0].dtype),) + output[1:]
                h = output.float()
                if h.dim() == 3: h[:, -1, :] += g * d.to(h.device)
                return h.to(output.dtype)
            return fn
        handle = model.model.layers[fga_layer].register_forward_hook(
            make_hook(direction, fga_gain))
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        handle.remove()
        pred_id = logits.argmax().item()
        if pred_id == exp_id or tok.decode([pred_id]).strip() == expected.strip():
            correct += 1
    return correct / len(test_data)


def main():
    print("[P179] Surgery Strength Spectrum")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    strengths = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    results = {}

    print("\n  === Surgery Strength Sweep ===")
    for s in strengths:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)

        if s > 0:
            avg_cos = apply_surgery_and_measure(model, tok, strength=s)
        else:
            # Measure baseline cosine
            ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
            vecs = model.model.embed_tokens.weight.data[ids].float()
            cos = F.cosine_similarity(vecs.unsqueeze(0), vecs.unsqueeze(1), dim=-1)
            mask = ~torch.eye(len(ids), dtype=bool)
            avg_cos = cos[mask].mean().item()

        fact_acc = evaluate_with_fga(model, tok, FACT_TEST, fga_gain=5)
        arith_acc = evaluate_with_fga(model, tok, ARITH_TEST, fga_gain=5)
        combined = fact_acc * 0.5 + arith_acc * 0.5

        print(f"    s={s:5.2f}: cos={avg_cos:.4f} fact={fact_acc:.0%} "
              f"arith={arith_acc:.0%} combined={combined:.2f}")

        results[str(s)] = {
            'strength': s, 'avg_cos': avg_cos,
            'fact_acc': fact_acc, 'arith_acc': arith_acc, 'combined': combined
        }
        del model; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase179_strength.json'), 'w') as f:
        json.dump({'phase': '179', 'name': 'Surgery Strength Spectrum',
                   'results': results}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    s_vals = [results[str(s)]['strength'] for s in strengths]

    ax = axes[0]
    fact_vals = [results[str(s)]['fact_acc'] for s in strengths]
    arith_vals = [results[str(s)]['arith_acc'] for s in strengths]
    combined_vals = [results[str(s)]['combined'] for s in strengths]
    ax.plot(s_vals, fact_vals, 'r-o', lw=2, label='Factual')
    ax.plot(s_vals, arith_vals, 'b-s', lw=2, label='Arithmetic')
    ax.plot(s_vals, combined_vals, 'g--^', lw=2, label='Combined')
    best_s = max(strengths, key=lambda s: results[str(s)]['combined'])
    ax.axvline(x=best_s, color='green', ls=':', lw=2, alpha=0.7,
               label=f'Best s={best_s}')
    ax.axvline(x=2.0, color='gray', ls='--', alpha=0.5, label='Current (s=2.0)')
    ax.set_xlabel('Surgery Strength', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('Accuracy vs Surgery Strength', fontsize=13, fontweight='bold')

    ax = axes[1]
    cos_vals = [results[str(s)]['avg_cos'] for s in strengths]
    ax.plot(s_vals, cos_vals, 'k-o', lw=2, markersize=6)
    ax.set_xlabel('Surgery Strength', fontsize=12)
    ax.set_ylabel('Avg Cosine Similarity', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('Number Token Dispersion', fontsize=13, fontweight='bold')
    ax.axvline(x=best_s, color='green', ls=':', lw=2, alpha=0.7)

    plt.suptitle('Phase 179: Surgery Strength Spectrum\n'
                 'Finding the Goldilocks Zone for embedding dispersion',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase179_strength.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    best = results[str(best_s)]
    print(f"\n  === VERDICT ===")
    print(f"  -> Optimal strength: {best_s} "
          f"(fact={best['fact_acc']:.0%}, arith={best['arith_acc']:.0%})")
    cur = results['2.0']
    print(f"  -> Current s=2.0: fact={cur['fact_acc']:.0%}, arith={cur['arith_acc']:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 179] Complete.")


if __name__ == '__main__':
    main()
