# -*- coding: utf-8 -*-
"""
Phase 14: Attention Head Surgery
- Ablate individual attention heads and measure factual accuracy change
- Identify which heads store facts vs grammar
- Create a "fact head" vs "skill head" map
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_model():
    print("[P14] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


class HeadAblationHook:
    """Zero out a specific attention head's output."""
    def __init__(self, head_idx, n_heads=12):
        self.head_idx = head_idx
        self.n_heads = n_heads

    def __call__(self, module, input, output):
        # output[0] shape: (batch, seq, hidden)
        hidden = output[0].clone()
        head_dim = hidden.shape[-1] // self.n_heads
        start = self.head_idx * head_dim
        end = start + head_dim
        hidden[:, :, start:end] = 0  # Zero out this head
        return (hidden,) + output[1:]


def get_fact_logit_rank(model, tok, prompt, fact_ids):
    """Get the rank of fact token in output distribution."""
    inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[:, -1, :].squeeze(0)
    sorted_ids = logits.argsort(descending=True)
    for rank, tid in enumerate(sorted_ids.tolist()):
        if tid in fact_ids:
            return rank + 1, float(logits[fact_ids[0]])
    return len(sorted_ids), 0.0


def main():
    print("=" * 70)
    print("  Phase 14: Attention Head Surgery")
    print("  Which heads store facts vs grammar?")
    print("=" * 70)

    model, tok = load_model()
    n_layers = 12
    n_heads = 12  # GPT-2 small has 12 heads per layer

    qa_pairs = [
        ("The capital of Japan is", [11790]),
        ("The capital of France is", [6342]),
        ("Water freezes at", [657]),
        ("The largest planet is", [22721]),
        ("DNA stands for", [390]),
        ("The chemical symbol for gold is", [7591]),
        ("Einstein developed the theory of", [823]),
        ("Shakespeare wrote", [13483]),
    ]

    # === Baseline ===
    print("\n[P14a] Baseline ranks...")
    baseline_ranks = {}
    for prompt, fact_ids in qa_pairs:
        rank, logit = get_fact_logit_rank(model, tok, prompt, fact_ids)
        baseline_ranks[prompt[:30]] = {'rank': rank, 'logit': logit}
        print(f"  rank={rank:>4d}: {prompt[:40]}")

    mean_baseline_rank = np.mean([v['rank'] for v in baseline_ranks.values()])

    # === Ablate each head and measure impact ===
    print(f"\n[P14b] Ablating {n_layers}x{n_heads}={n_layers*n_heads} heads...")
    impact_matrix = np.zeros((n_layers, n_heads))  # rank change

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            rank_changes = []
            for prompt, fact_ids in qa_pairs:
                # Register ablation hook
                hook = HeadAblationHook(head_idx, n_heads)
                handle = model.transformer.h[layer_idx].register_forward_hook(hook)

                rank_abl, _ = get_fact_logit_rank(model, tok, prompt, fact_ids)
                handle.remove()

                base_rank = baseline_ranks[prompt[:30]]['rank']
                rank_changes.append(rank_abl - base_rank)

            impact_matrix[layer_idx, head_idx] = np.mean(rank_changes)

        # Print summary for this layer
        worst_head = np.argmax(impact_matrix[layer_idx])
        best_head = np.argmin(impact_matrix[layer_idx])
        print(f"  Layer {layer_idx:>2d}: worst head={worst_head} "
              f"(+{impact_matrix[layer_idx, worst_head]:.0f} rank), "
              f"best head={best_head} "
              f"({impact_matrix[layer_idx, best_head]:.0f} rank)")

    # === Identify fact heads vs skill heads ===
    print("\n[P14c] Identifying fact heads (ablation hurts factuality)...")
    # Positive impact = ablating hurts factuality = fact head
    # Negative impact = ablating helps factuality = skill/noise head
    fact_heads = []
    skill_heads = []

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            impact = impact_matrix[layer_idx, head_idx]
            if impact > 5:  # Ablating causes >5 rank degradation
                fact_heads.append((layer_idx, head_idx, impact))
            elif impact < -5:
                skill_heads.append((layer_idx, head_idx, impact))

    print(f"  Fact heads (ablation hurts): {len(fact_heads)}")
    for l, h, imp in sorted(fact_heads, key=lambda x: -x[2])[:10]:
        print(f"    L{l}H{h}: +{imp:.0f} rank")
    print(f"  Skill heads (ablation helps): {len(skill_heads)}")
    for l, h, imp in sorted(skill_heads, key=lambda x: x[2])[:10]:
        print(f"    L{l}H{h}: {imp:.0f} rank")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Impact heatmap
    im = axes[0].imshow(impact_matrix, aspect='auto', cmap='RdBu_r',
                         vmin=-np.percentile(np.abs(impact_matrix), 95),
                         vmax=np.percentile(np.abs(impact_matrix), 95))
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')
    axes[0].set_title('Head Ablation Impact (rank change)')
    plt.colorbar(im, ax=axes[0])

    # Plot 2: Layer-averaged impact
    layer_impact = impact_matrix.mean(axis=1)
    axes[1].barh(range(n_layers), layer_impact, color='steelblue', alpha=0.7)
    axes[1].set_ylabel('Layer')
    axes[1].set_xlabel('Mean Rank Change')
    axes[1].set_title('Layer-averaged Factuality Impact')
    axes[1].set_yticks(range(n_layers))

    # Plot 3: Distribution of impacts
    all_impacts = impact_matrix.flatten()
    axes[2].hist(all_impacts, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[2].axvline(x=0, color='k', linestyle='--')
    axes[2].set_xlabel('Rank Change on Ablation')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Impact Distribution ({len(fact_heads)} fact, {len(skill_heads)} skill)')

    plt.suptitle('Phase 14: Attention Head Surgery', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase14_head_surgery.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 14, 'name': 'Attention Head Surgery',
        'n_fact_heads': len(fact_heads),
        'n_skill_heads': len(skill_heads),
        'fact_heads': [(l, h, float(i)) for l, h, i in fact_heads],
        'skill_heads': [(l, h, float(i)) for l, h, i in skill_heads],
        'impact_matrix': impact_matrix.tolist(),
        'mean_baseline_rank': float(mean_baseline_rank),
    }
    with open(os.path.join(RESULTS_DIR, 'phase14_head_surgery.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  PHASE 14 RESULTS: Head Surgery")
    print("=" * 70)
    print(f"  Fact heads: {len(fact_heads)}")
    print(f"  Skill heads: {len(skill_heads)}")
    print(f"  Most critical: L{fact_heads[0][0]}H{fact_heads[0][1]} "
          f"(+{fact_heads[0][2]:.0f})") if fact_heads else None
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
