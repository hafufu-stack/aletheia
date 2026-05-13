# -*- coding: utf-8 -*-
"""
Phase 97: The Fact Amplifier - XL Paradox Anatomy
Analyze all 1200 heads (48 layers x 25 heads) of GPT-2 XL.
Compare Natural vs Code Mode attention entropy to find
the "Fact Amplifier" heads that Code Mode activates.
GPU-accelerated.
"""
import torch, json, os, sys, gc, numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FACTS = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("The chemical symbol for gold is", " Au"),
    ("The speed of light is approximately", " 299"),
    ("Albert Einstein was born in", " Ul"),
]

def main():
    print(f"[P97] Fact Amplifier - XL 1200-Head Analysis (device={DEVICE})")
    model = GPT2LMHeadModel.from_pretrained(
        'gpt2-xl', local_files_only=True, attn_implementation='eager'
    ).eval().to(DEVICE)
    tok = GPT2Tokenizer.from_pretrained('gpt2-xl', local_files_only=True)

    n_layers = 48
    n_heads = 25

    # Collect attention entropy for Natural vs Code Mode
    natural_entropies = np.zeros((n_layers, n_heads))
    code_entropies = np.zeros((n_layers, n_heads))
    natural_counts = np.zeros((n_layers, n_heads))
    code_counts = np.zeros((n_layers, n_heads))

    for prompt, answer in FACTS:
        for mode, prefix, ent_arr, cnt_arr in [
            ('natural', '', natural_entropies, natural_counts),
            ('code', '# ', code_entropies, code_counts),
        ]:
            full = prefix + prompt
            inp = tok(full, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            attentions = out.attentions

            for layer_idx in range(n_layers):
                attn = attentions[layer_idx][0, :, -1, :]  # (n_heads, seq_len)
                for head_idx in range(n_heads):
                    w = attn[head_idx].cpu().numpy()
                    ent = -(w * np.log(w + 1e-10)).sum()
                    ent_arr[layer_idx, head_idx] += ent
                    cnt_arr[layer_idx, head_idx] += 1

        # Free memory periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average
    natural_entropies /= np.maximum(natural_counts, 1)
    code_entropies /= np.maximum(code_counts, 1)

    # Delta: Code - Natural (positive = Code increases entropy = weakens head)
    delta = code_entropies - natural_entropies

    # Find top suppressors (highest positive delta = most weakened by Code)
    flat_idx = np.argsort(delta.ravel())[::-1]
    top_weakened = []
    for idx in flat_idx[:20]:
        l, h = divmod(idx, n_heads)
        top_weakened.append({
            'layer': int(l), 'head': int(h),
            'natural_ent': float(natural_entropies[l, h]),
            'code_ent': float(code_entropies[l, h]),
            'delta': float(delta[l, h]),
        })

    # Find top amplifiers (most negative delta = Code FOCUSES this head)
    top_amplified = []
    for idx in flat_idx[-20:][::-1]:
        l, h = divmod(idx, n_heads)
        top_amplified.append({
            'layer': int(l), 'head': int(h),
            'natural_ent': float(natural_entropies[l, h]),
            'code_ent': float(code_entropies[l, h]),
            'delta': float(delta[l, h]),
        })

    print(f"\n  Top 10 Weakened by Code Mode (Suppressors):")
    for i, h in enumerate(top_weakened[:10]):
        print(f"    {i+1}. L{h['layer']}H{h['head']}: nat={h['natural_ent']:.3f} -> code={h['code_ent']:.3f} (delta=+{h['delta']:.3f})")

    print(f"\n  Top 10 Amplified by Code Mode (Fact Amplifiers):")
    for i, h in enumerate(top_amplified[:10]):
        print(f"    {i+1}. L{h['layer']}H{h['head']}: nat={h['natural_ent']:.3f} -> code={h['code_ent']:.3f} (delta={h['delta']:.3f})")

    out = {
        'phase': 97, 'name': 'Fact Amplifier XL',
        'n_layers': n_layers, 'n_heads': n_heads,
        'top_weakened': top_weakened,
        'top_amplified': top_amplified,
        'mean_natural_entropy': float(natural_entropies.mean()),
        'mean_code_entropy': float(code_entropies.mean()),
        'mean_delta': float(delta.mean()),
    }
    with open(os.path.join(RESULTS_DIR, 'phase97_fact_amplifier.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Entropy delta heatmap
    im = axes[0].imshow(delta, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')
    axes[0].set_title('Entropy Delta (Code - Natural)')
    plt.colorbar(im, ax=axes[0], label='Delta (red=weakened)')

    # 2. Per-layer mean delta
    layer_mean_delta = delta.mean(axis=1)
    axes[1].barh(range(n_layers), layer_mean_delta,
                color=['#e74c3c' if d > 0 else '#2ecc71' for d in layer_mean_delta])
    axes[1].set_ylabel('Layer')
    axes[1].set_xlabel('Mean Entropy Delta')
    axes[1].set_title('Per-Layer Code Mode Effect')
    axes[1].invert_yaxis()

    # 3. Top amplifiers vs weakened
    top_w_labels = [f"L{h['layer']}H{h['head']}" for h in top_weakened[:10]]
    top_w_vals = [h['delta'] for h in top_weakened[:10]]
    top_a_labels = [f"L{h['layer']}H{h['head']}" for h in top_amplified[:10]]
    top_a_vals = [h['delta'] for h in top_amplified[:10]]
    all_labels = top_a_labels[::-1] + top_w_labels
    all_vals = top_a_vals[::-1] + top_w_vals
    colors = ['#2ecc71']*10 + ['#e74c3c']*10
    axes[2].barh(all_labels, all_vals, color=colors)
    axes[2].set_xlabel('Entropy Delta')
    axes[2].set_title('Top 10 Amplifiers vs Suppressors')
    axes[2].axvline(x=0, color='black', linewidth=0.5)

    fig.suptitle('Phase 97: The Fact Amplifier - XL 1200-Head Analysis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase97_fact_amplifier.png'), dpi=150)
    plt.close()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Phase 97] Complete.")

if __name__ == '__main__':
    main()
