# -*- coding: utf-8 -*-
"""
Phase 30: Topological Truth Dimension
Inspired by Blue Brain Project's 11D clique structures in biological brains.
Measure the topological dimension of "truth circuits" vs "hallucination circuits"
in GPT-2's attention graph using simplicial complex analysis.

Hypothesis: Truth-encoding circuits form higher-dimensional cliques than
hallucination circuits, mirroring the brain's 11D information structures.
"""
import os, json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import phase_complete

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_model():
    print("[P30] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                             attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_attention_graph(model, tok, prompt):
    """Extract attention pattern across all heads for a prompt."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**inp, output_attentions=True, return_dict=True)
    # attentions: tuple of (batch, n_heads, seq, seq) per layer
    attentions = out.attentions
    if attentions is None:
        model.config.output_attentions = True
        with torch.no_grad():
            out = model(**inp, return_dict=True)
        attentions = out.attentions
    # Extract last-token attention for each head
    head_activations = []
    for layer_idx, attn in enumerate(attentions):
        for head_idx in range(attn.shape[1]):
            # Attention from last token to all previous tokens
            attn_vec = attn[0, head_idx, -1, :].cpu().numpy()
            head_activations.append({
                'layer': layer_idx, 'head': head_idx,
                'entropy': float(-np.sum(attn_vec * np.log(attn_vec + 1e-12))),
                'max_attn': float(np.max(attn_vec)),
                'attn_vec': attn_vec,
            })
    return head_activations

def build_co_activation_graph(activations_list, threshold=0.5):
    """Build graph where heads are nodes, edges = co-activation above threshold.
    Two heads are connected if their attention patterns correlate highly."""
    n_heads = len(activations_list[0])
    # Average correlation across all prompts
    corr_matrix = np.zeros((n_heads, n_heads))
    for activations in activations_list:
        vecs = np.array([a['attn_vec'][:min(len(a['attn_vec']), 20)] for a in activations])
        # Pad to same length
        max_len = max(len(v) for v in vecs)
        padded = np.zeros((len(vecs), max_len))
        for i, v in enumerate(vecs):
            padded[i, :len(v)] = v
        corr = np.corrcoef(padded)
        corr = np.nan_to_num(corr, nan=0)
        corr_matrix += corr
    corr_matrix /= len(activations_list)

    # Build adjacency from correlation
    adj = (np.abs(corr_matrix) > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj, corr_matrix

def find_cliques(adj, max_dim=8):
    """Find maximal cliques up to max_dim using greedy search."""
    n = adj.shape[0]
    clique_counts = {}
    # Count cliques of each size
    for dim in range(2, max_dim + 1):
        count = 0
        # Sample random subsets to estimate
        n_samples = min(5000, int(np.prod(range(min(n, 15) - dim + 1, min(n, 15) + 1)) /
                        np.prod(range(1, dim + 1))) if dim <= min(n, 15) else 0)
        if n_samples == 0:
            break
        for _ in range(n_samples):
            nodes = np.random.choice(n, dim, replace=False)
            # Check if all pairs connected
            is_clique = True
            for i, j in combinations(nodes, 2):
                if adj[i, j] == 0:
                    is_clique = False
                    break
            if is_clique:
                count += 1
        # Estimate total
        total_possible = 1
        for k in range(dim):
            total_possible *= (n - k) / (dim - k)
        estimated = count / n_samples * total_possible if n_samples > 0 else 0
        clique_counts[dim] = int(estimated)
        if estimated == 0 and dim > 3:
            break
    return clique_counts

def main():
    print("=" * 70)
    print("  Phase 30: Topological Truth Dimension")
    print("  Do truth circuits form higher-dimensional cliques?")
    print("=" * 70)

    model, tok = load_model()

    # Factual prompts (model knows these)
    fact_prompts = [
        "The capital of Japan is",
        "The capital of France is",
        "Water freezes at",
        "The largest planet is",
        "The speed of light is approximately",
        "DNA stands for",
        "Albert Einstein developed the theory of",
        "The first president of the United States was",
    ]

    # Hallucination-inducing prompts (model will guess)
    hallu_prompts = [
        "The 37th element of the periodic table is",
        "The population of the city Xanthe on Mars is",
        "The inventor of the quantum flux capacitor was",
        "The capital of the underwater nation Atlantis is",
        "The winner of the 2089 Nobel Prize in Physics was",
        "The chemical formula for unobtanium is",
        "The exact number of stars in the universe is",
        "The 15th digit of pi times e is",
    ]

    # === P30a: Extract attention graphs ===
    print("\n[P30a] Extracting attention graphs...")
    fact_activations = []
    hallu_activations = []
    for p in fact_prompts:
        fact_activations.append(get_attention_graph(model, tok, p))
    for p in hallu_prompts:
        hallu_activations.append(get_attention_graph(model, tok, p))

    # === P30b: Build co-activation graphs ===
    print("\n[P30b] Building co-activation graphs...")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    fact_dims = {}
    hallu_dims = {}

    for thresh in thresholds:
        fact_adj, fact_corr = build_co_activation_graph(fact_activations, thresh)
        hallu_adj, hallu_corr = build_co_activation_graph(hallu_activations, thresh)

        fact_edges = np.sum(fact_adj) // 2
        hallu_edges = np.sum(hallu_adj) // 2

        fact_cliques = find_cliques(fact_adj)
        hallu_cliques = find_cliques(hallu_adj)

        fact_max_dim = max(fact_cliques.keys()) if fact_cliques else 1
        hallu_max_dim = max(hallu_cliques.keys()) if hallu_cliques else 1

        fact_dims[thresh] = {'edges': int(fact_edges), 'max_dim': fact_max_dim,
                            'cliques': fact_cliques}
        hallu_dims[thresh] = {'edges': int(hallu_edges), 'max_dim': hallu_max_dim,
                             'cliques': hallu_cliques}

        print(f"  thresh={thresh:.1f}: fact_edges={fact_edges}, fact_maxdim={fact_max_dim} | "
              f"hallu_edges={hallu_edges}, hallu_maxdim={hallu_max_dim}")

    # === P30c: Entropy comparison ===
    print("\n[P30c] Attention entropy: fact vs hallucination...")
    fact_entropies = [np.mean([a['entropy'] for a in acts]) for acts in fact_activations]
    hallu_entropies = [np.mean([a['entropy'] for a in acts]) for acts in hallu_activations]
    print(f"  Fact mean entropy:  {np.mean(fact_entropies):.3f}")
    print(f"  Hallu mean entropy: {np.mean(hallu_entropies):.3f}")

    # === P30d: Per-layer analysis ===
    print("\n[P30d] Per-layer attention concentration...")
    for layer in range(12):
        fact_max = np.mean([acts[layer*12:(layer+1)*12][0]['max_attn']
                          for acts in fact_activations if len(acts) > layer*12])
        hallu_max = np.mean([acts[layer*12:(layer+1)*12][0]['max_attn']
                           for acts in hallu_activations if len(acts) > layer*12])
        delta = fact_max - hallu_max
        bar = '+' * int(abs(delta) * 100) if delta > 0 else '-' * int(abs(delta) * 100)
        print(f"  L{layer:>2d}: fact={fact_max:.3f} hallu={hallu_max:.3f} delta={delta:+.3f} {bar[:20]}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Clique dimension comparison
    best_thresh = 0.5
    fc = fact_dims[best_thresh]['cliques']
    hc = hallu_dims[best_thresh]['cliques']
    all_dims = sorted(set(list(fc.keys()) + list(hc.keys())))
    fc_vals = [fc.get(d, 0) for d in all_dims]
    hc_vals = [hc.get(d, 0) for d in all_dims]
    x = range(len(all_dims))
    axes[0].bar([i-0.2 for i in x], fc_vals, 0.4, label='Fact circuits', color='green', alpha=0.7)
    axes[0].bar([i+0.2 for i in x], hc_vals, 0.4, label='Hallu circuits', color='red', alpha=0.7)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([str(d) for d in all_dims])
    axes[0].set_xlabel('Clique Dimension')
    axes[0].set_ylabel('Estimated Count')
    axes[0].set_title(f'Clique Dimensions (thresh={best_thresh})')
    axes[0].legend()

    # Plot 2: Entropy distribution
    axes[1].hist(fact_entropies, bins=8, alpha=0.6, color='green', label='Fact')
    axes[1].hist(hallu_entropies, bins=8, alpha=0.6, color='red', label='Hallu')
    axes[1].set_xlabel('Mean Attention Entropy')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Attention Entropy Distribution')
    axes[1].legend()

    # Plot 3: Max dim vs threshold
    ts = sorted(fact_dims.keys())
    fd = [fact_dims[t]['max_dim'] for t in ts]
    hd = [hallu_dims[t]['max_dim'] for t in ts]
    axes[2].plot(ts, fd, 'g.-', linewidth=2, label='Fact', markersize=10)
    axes[2].plot(ts, hd, 'r.-', linewidth=2, label='Hallu', markersize=10)
    axes[2].set_xlabel('Correlation Threshold')
    axes[2].set_ylabel('Max Clique Dimension')
    axes[2].set_title('Topological Dimension')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 30: Topological Truth Dimension\n(Inspired by Blue Brain 11D Cliques)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase30_topology.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 30, 'name': 'Topological Truth Dimension',
        'inspiration': 'Blue Brain Project 11D clique structures',
        'fact_entropy': float(np.mean(fact_entropies)),
        'hallu_entropy': float(np.mean(hallu_entropies)),
        'fact_topology': {str(k): v for k, v in fact_dims.items()},
        'hallu_topology': {str(k): v for k, v in hallu_dims.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase30_topology.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 30 RESULTS: Topological Truth Dimension")
    print("=" * 70)
    print(f"  Fact entropy:  {np.mean(fact_entropies):.3f}")
    print(f"  Hallu entropy: {np.mean(hallu_entropies):.3f}")
    best = fact_dims[0.5]
    print(f"  Fact max clique dim: {best['max_dim']}")
    print(f"  Hallu max clique dim: {hallu_dims[0.5]['max_dim']}")
    print("=" * 70)

    phase_complete(30)
    return results

if __name__ == '__main__':
    main()
