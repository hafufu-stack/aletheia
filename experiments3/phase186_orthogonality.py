# -*- coding: utf-8 -*-
"""
Phase 186: The Orthogonality Principle (Opus Addition)
P179 found: optimal surgery strength = cos(num_tokens) ~= 0 (orthogonal).
But WHY is orthogonality the sweet spot?

Hypothesis: Orthogonal number tokens maximize the INFORMATION
content of FGA direction vectors. When tokens are correlated,
FGA toward "7" also activates "3", "5" etc. (crosstalk).
Orthogonal = zero crosstalk = pure signal.

Test: Measure FGA "crosstalk" at different surgery strengths.
Define crosstalk = average cosine(FGA_direction_i, num_token_j) for j != i.

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

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9"]
ARITH_TEST = [
    ("# 1 + 1 =", " 2"), ("# 3 + 4 =", " 7"), ("# 5 + 5 =", " 10"),
    ("# 8 + 1 =", " 9"), ("# 6 + 3 =", " 9"), ("# 4 + 4 =", " 8"),
]

def main():
    print("[P186] The Orthogonality Principle")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    num_ids = [tok.encode(t)[-1] for t in NUM_TOKENS]  # 0-9 token ids

    strengths = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
    results = {}

    for s in strengths:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)

        # Apply surgery
        if s > 0:
            embed = model.model.embed_tokens.weight.data
            ids = list(set(num_ids))
            vecs = embed[ids].clone().float()
            center = vecs.mean(dim=0)
            for i, idx in enumerate(ids):
                diff = vecs[i] - center
                direction = diff / (diff.norm() + 1e-8)
                embed[idx] += (s * direction * embed[idx].float().norm()).to(embed.dtype)

        # Measure embedding cosine matrix
        embed_vecs = model.model.embed_tokens.weight.data[num_ids].float()
        embed_cos = F.cosine_similarity(embed_vecs.unsqueeze(0), embed_vecs.unsqueeze(1), dim=-1)

        # Measure lm_head (unembed) cosine matrix for FGA directions
        unembed_vecs = model.lm_head.weight.data[num_ids].float()
        unembed_cos = F.cosine_similarity(unembed_vecs.unsqueeze(0), unembed_vecs.unsqueeze(1), dim=-1)

        # Crosstalk = average off-diagonal cosine
        mask = ~torch.eye(len(num_ids), dtype=bool)
        embed_crosstalk = embed_cos[mask].abs().mean().item()
        unembed_crosstalk = unembed_cos[mask].abs().mean().item()

        # Also measure: embed-unembed alignment
        # If embed and unembed point same direction, FGA can "speak" to the correct token
        alignment = []
        for i, idx in enumerate(num_ids):
            e = embed_vecs[i]
            u = unembed_vecs[i]
            cos_eu = F.cosine_similarity(e.unsqueeze(0), u.unsqueeze(0)).item()
            alignment.append(cos_eu)
        avg_alignment = np.mean(alignment)

        # Accuracy
        n_layers = model.config.num_hidden_layers
        fga_layer = n_layers - max(1, n_layers // 4)
        correct = 0
        for prompt, expected in ARITH_TEST:
            exp_id = tok.encode(expected)[-1]
            unembed = model.lm_head.weight.data[exp_id].float()
            d = unembed / (unembed.norm() + 1e-8)
            def mk(dd, gg):
                def fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].float()
                        if h.dim() == 3: h[:, -1, :] += gg * dd.to(h.device)
                        return (h.to(output[0].dtype),) + output[1:]
                    return output
                return fn
            handle = model.model.layers[fga_layer].register_forward_hook(mk(d, 5))
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()
            handle.remove()
            if logits.argmax().item() == exp_id: correct += 1
        arith_acc = correct / len(ARITH_TEST)

        print(f"  s={s:4.1f}: embed_xtalk={embed_crosstalk:.4f} "
              f"unembed_xtalk={unembed_crosstalk:.4f} "
              f"alignment={avg_alignment:.4f} arith={arith_acc:.0%}")

        results[str(s)] = {
            'strength': s, 'embed_crosstalk': embed_crosstalk,
            'unembed_crosstalk': unembed_crosstalk,
            'alignment': avg_alignment, 'arith_acc': arith_acc
        }
        del model; gc.collect(); torch.cuda.empty_cache()

    with open(os.path.join(RESULTS_DIR, 'phase186_orthogonality.json'), 'w') as f:
        json.dump({'phase': '186', 'name': 'The Orthogonality Principle',
                   'results': results}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    s_vals = [results[str(s)]['strength'] for s in strengths]

    ax = axes[0]
    xt_e = [results[str(s)]['embed_crosstalk'] for s in strengths]
    xt_u = [results[str(s)]['unembed_crosstalk'] for s in strengths]
    ax.plot(s_vals, xt_e, 'b-o', lw=2, label='Embed crosstalk')
    ax.plot(s_vals, xt_u, 'r-s', lw=2, label='Unembed crosstalk')
    ax.axhline(y=0, color='green', ls='--', alpha=0.5, label='Zero (orthogonal)')
    ax.set_xlabel('Surgery Strength', fontsize=12)
    ax.set_ylabel('Avg |Cosine| (off-diagonal)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('Crosstalk vs Strength', fontsize=13, fontweight='bold')

    ax = axes[1]
    al = [results[str(s)]['alignment'] for s in strengths]
    ax.plot(s_vals, al, 'g-^', lw=2, markersize=8)
    ax.set_xlabel('Surgery Strength', fontsize=12)
    ax.set_ylabel('Embed-Unembed Alignment', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('Embed <-> Unembed Cos', fontsize=13, fontweight='bold')

    ax = axes[2]
    acc = [results[str(s)]['arith_acc'] for s in strengths]
    ax.plot(s_vals, acc, 'k-D', lw=2.5, markersize=8)
    # Also plot embed crosstalk on twin axis
    ax2 = ax.twinx()
    ax2.plot(s_vals, xt_e, 'b--o', lw=1.5, alpha=0.5, label='Embed xtalk')
    ax.set_xlabel('Surgery Strength', fontsize=12)
    ax.set_ylabel('Arithmetic Accuracy', fontsize=12, color='black')
    ax2.set_ylabel('Embed Crosstalk', fontsize=12, color='blue')
    ax.grid(True, alpha=0.3)
    ax.set_title('Accuracy tracks Orthogonality', fontsize=13, fontweight='bold')

    plt.suptitle('Phase 186: The Orthogonality Principle\n'
                 'WHY cos=0 is optimal: zero crosstalk = pure FGA signal',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase186_orthogonality.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    # Find correlation between crosstalk and accuracy
    xt_arr = np.array([results[str(s)]['embed_crosstalk'] for s in strengths])
    acc_arr = np.array([results[str(s)]['arith_acc'] for s in strengths])
    if len(xt_arr) > 2:
        corr = np.corrcoef(xt_arr, acc_arr)[0, 1]
        print(f"  -> Correlation(crosstalk, accuracy) = {corr:.3f}")
        if corr < -0.5:
            print("  -> CONFIRMED: Lower crosstalk = Higher accuracy")
            print("  -> Orthogonality IS the mechanism!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 186] Complete.")

if __name__ == '__main__':
    main()
