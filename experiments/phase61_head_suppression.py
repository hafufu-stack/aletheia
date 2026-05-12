# -*- coding: utf-8 -*-
"""
Phase 61: Attention Head Suppression Identification (Original idea)
Which specific attention heads in L11-L12 are responsible for suppressing facts?
Ablate each head individually and measure fact rank recovery.
Identify the "Grammar Police" heads.
"""
import os, json, sys
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import phase_complete

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_model():
    print("[P61] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    """Get rank of fact token in logits (1-indexed)."""
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 61: Attention Head Suppression Identification")
    print("  Which heads in L11-L12 kill facts?")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The Earth orbits the", [4252], "Sun"),
        ("The boiling point of water is", [1802], "100"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("Oxygen has the atomic number", [807], "8"),
    ]

    n_heads = model.config.n_head  # 12 for GPT-2

    # === 1. Baseline ranks ===
    print("\n[P61a] Baseline fact ranks (no ablation)...")
    baseline_ranks = {}
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        with torch.no_grad():
            out = model(inp)
        rank = get_fact_rank(out.logits[0, -1, :], fact_ids[0])
        baseline_ranks[expected] = rank
        print(f"  {expected:>12s}: rank={rank}")

    # === 2. Individual head ablation in L10, L11 ===
    # Only ablate the final layers (L10 and L11 in GPT-2 0-indexed,
    # which are the 11th and 12th transformer blocks)
    print("\n[P61b] Head ablation in L10 and L11...")
    ablation_results = {}

    for layer_idx in [10, 11]:
        print(f"\n  --- Layer {layer_idx} ---")
        for head_idx in range(n_heads):
            head_ranks = {}

            # Hook to zero out specific head
            def make_ablation_hook(li, hi):
                def hook_fn(module, args, output):
                    # output[0] is hidden state after attention
                    # We need to modify the attention output
                    # GPT-2 attention: output shape = (batch, seq, hidden)
                    # Each head contributes hidden_dim/n_heads dimensions
                    hs = output[0]
                    head_dim = hs.shape[-1] // n_heads
                    start = hi * head_dim
                    end = start + head_dim
                    # Zero out this head's contribution
                    modified = hs.clone()
                    modified[:, :, start:end] = 0.0
                    return (modified,) + output[1:]
                return hook_fn

            for prompt, fact_ids, expected in tests:
                inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
                handle = model.transformer.h[layer_idx].register_forward_hook(
                    make_ablation_hook(layer_idx, head_idx))
                with torch.no_grad():
                    out = model(inp)
                handle.remove()

                rank = get_fact_rank(out.logits[0, -1, :], fact_ids[0])
                head_ranks[expected] = rank

            # Compute improvement
            improvements = []
            for exp in head_ranks:
                base = baseline_ranks[exp]
                ablated = head_ranks[exp]
                improvements.append(base - ablated)  # positive = rank improved

            mean_improvement = float(np.mean(improvements))
            median_improvement = float(np.median(improvements))
            ablation_results[(layer_idx, head_idx)] = {
                'ranks': head_ranks,
                'mean_improvement': mean_improvement,
                'median_improvement': median_improvement,
            }

            tag = 'SUPPRESSOR' if mean_improvement > 10 else 'neutral' if mean_improvement > -10 else 'HELPER'
            print(f"  L{layer_idx}H{head_idx:>2d}: mean_impr={mean_improvement:>8.1f} [{tag}]")

    # === 3. Find top suppressors ===
    print("\n[P61c] Top fact-suppressing heads (removing them IMPROVES rank):")
    sorted_heads = sorted(ablation_results.items(),
                          key=lambda x: x[1]['mean_improvement'], reverse=True)
    for (li, hi), data in sorted_heads[:5]:
        print(f"  L{li}H{hi}: mean_improvement={data['mean_improvement']:.1f}")

    print("\n[P61c] Top fact-helping heads (removing them WORSENS rank):")
    for (li, hi), data in sorted_heads[-5:]:
        print(f"  L{li}H{hi}: mean_improvement={data['mean_improvement']:.1f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Heatmap of improvements
    for plot_idx, layer in enumerate([10, 11]):
        improvements = [ablation_results[(layer, h)]['mean_improvement']
                       for h in range(n_heads)]
        colors = ['green' if v > 10 else 'red' if v < -10 else 'gray'
                  for v in improvements]
        axes[plot_idx].bar(range(n_heads), improvements, color=colors, alpha=0.7,
                          edgecolor='black', linewidth=0.5)
        axes[plot_idx].axhline(y=0, color='black', linewidth=0.5)
        axes[plot_idx].set_xlabel('Head Index')
        axes[plot_idx].set_ylabel('Mean Rank Improvement')
        axes[plot_idx].set_title(f'L{layer} Head Ablation')
        axes[plot_idx].set_xticks(range(n_heads))

    # 3. Top 5 suppressors with per-prompt detail
    top5 = sorted_heads[:5]
    top5_labels = [f"L{li}H{hi}" for (li, hi), _ in top5]
    top5_vals = [d['mean_improvement'] for _, d in top5]
    axes[2].barh(range(len(top5)), top5_vals, color='green', alpha=0.7,
                edgecolor='black')
    axes[2].set_yticks(range(len(top5)))
    axes[2].set_yticklabels(top5_labels)
    axes[2].set_xlabel('Mean Rank Improvement')
    axes[2].set_title('Top 5 Fact Suppressors\n(removing them helps)')

    plt.suptitle('Phase 61: Attention Head Suppression ID', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase61_head_suppression.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 61, 'name': 'Attention Head Suppression Identification',
        'baseline_ranks': baseline_ranks,
        'ablation_results': {f'L{k[0]}H{k[1]}': v for k, v in ablation_results.items()},
        'top_suppressors': [(f'L{li}H{hi}', d['mean_improvement'])
                           for (li, hi), d in sorted_heads[:5]],
        'top_helpers': [(f'L{li}H{hi}', d['mean_improvement'])
                       for (li, hi), d in sorted_heads[-5:]],
    }
    with open(os.path.join(RESULTS_DIR, 'phase61_head_suppression.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 61 RESULTS")
    print("=" * 70)
    print("  Top suppressors (removing helps facts):")
    for (li, hi), d in sorted_heads[:3]:
        print(f"    L{li}H{hi}: +{d['mean_improvement']:.1f} rank improvement")
    print("  Top helpers (removing hurts facts):")
    for (li, hi), d in sorted_heads[-3:]:
        print(f"    L{li}H{hi}: {d['mean_improvement']:.1f} rank change")
    print("=" * 70)
    phase_complete(61)

if __name__ == '__main__':
    main()
