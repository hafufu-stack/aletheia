# -*- coding: utf-8 -*-
"""
Phase 43: Truth Manifold Dimensionality
From Rosetta: programs live on a ~5D manifold in 64D space.
Question: what is the intrinsic dimensionality of "truth" in GPT-2's
768D hidden space? Is truth lower-dimensional than hallucination?
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
    print("[P43] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_hidden_at_layer(model, tok, prompt, layer_idx):
    hidden = {}
    def hook(module, args, output):
        hidden['h'] = output[0][0, -1, :].detach().cpu().numpy()
    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model(**inp)
    handle.remove()
    return hidden['h']

def effective_dimensionality(states):
    """Compute effective dimensionality via PCA explained variance."""
    states = np.array(states)
    states = states - states.mean(axis=0)
    if states.shape[0] < 2:
        return 0, [], []
    cov = np.cov(states.T)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    total = eigenvalues.sum()
    if total == 0:
        return 0, [], []
    explained = np.cumsum(eigenvalues) / total

    # Effective dim: number of components for 90% variance
    eff_dim_90 = int(np.searchsorted(explained, 0.90)) + 1
    # Participation ratio
    pr = (eigenvalues.sum() ** 2) / (np.sum(eigenvalues ** 2) + 1e-12)
    return eff_dim_90, explained[:20].tolist(), float(pr)

def main():
    print("=" * 70)
    print("  Phase 43: Truth Manifold Dimensionality")
    print("  Is truth lower-dimensional than hallucination?")
    print("=" * 70)

    model, tok = load_model()

    # Large set of fact prompts
    fact_prompts = [
        "The capital of Japan is", "The capital of France is",
        "The capital of Germany is", "The capital of Italy is",
        "The capital of Spain is", "The capital of Brazil is",
        "Water freezes at", "The sun is a",
        "The largest planet is", "DNA stands for",
        "The chemical symbol for gold is", "Shakespeare wrote",
        "Albert Einstein developed the theory of",
        "The speed of light is approximately",
        "The Earth orbits the", "Photosynthesis converts",
        "The human body has approximately", "The Pacific Ocean is the",
        "Gravity was described by", "The boiling point of water is",
    ]

    hallu_prompts = [
        "The capital of the underwater nation Atlantis is",
        "The inventor of the quantum flux capacitor was",
        "The winner of the 2089 Nobel Prize in Physics was",
        "The color of the 5th quark flavor is",
        "The population of the city Xanthe on Mars is",
        "The 37th element of the periodic table is",
        "The chemical formula for unobtanium is",
        "The 15th digit of pi times e is",
        "The capital of the cloud kingdom Laputa is",
        "The inventor of the perpetual motion machine was",
        "The atomic weight of vibranium is",
        "The population of the moon colony in 2150 is",
        "The recipe for philosopher's stone requires",
        "The distance to the nearest wormhole is",
        "The winner of the 3050 Olympic marathon was",
        "The melting point of dark matter is",
        "The color of the invisible rainbow is",
        "The frequency of silence is",
        "The weight of a shadow is",
        "The temperature of absolute hot is",
    ]

    # === P43a: Per-layer dimensionality ===
    print("\n[P43a] Computing dimensionality per layer...")
    dim_results = {}

    for layer in [0, 3, 6, 9, 11]:
        fact_states = [get_hidden_at_layer(model, tok, p, layer) for p in fact_prompts]
        hallu_states = [get_hidden_at_layer(model, tok, p, layer) for p in hallu_prompts]

        fact_dim, fact_exp, fact_pr = effective_dimensionality(fact_states)
        hallu_dim, hallu_exp, hallu_pr = effective_dimensionality(hallu_states)

        dim_results[layer] = {
            'fact_dim_90': fact_dim, 'hallu_dim_90': hallu_dim,
            'fact_pr': round(fact_pr, 1), 'hallu_pr': round(hallu_pr, 1),
        }
        print(f"  L{layer:>2d}: fact_dim={fact_dim:>3d} (PR={fact_pr:.1f}) "
              f"hallu_dim={hallu_dim:>3d} (PR={hallu_pr:.1f}) "
              f"{'FACT LOWER' if fact_dim < hallu_dim else 'hallu lower' if hallu_dim < fact_dim else 'equal'}")

    # === P43b: Detailed PCA at L11 (final layer) ===
    print("\n[P43b] Detailed PCA at L11...")
    fact_states = [get_hidden_at_layer(model, tok, p, 11) for p in fact_prompts]
    hallu_states = [get_hidden_at_layer(model, tok, p, 11) for p in hallu_prompts]

    all_states = np.array(fact_states + hallu_states)
    all_states -= all_states.mean(axis=0)
    U, S, Vt = np.linalg.svd(all_states, full_matrices=False)

    # Project onto PC1-PC3
    fact_proj = np.array(fact_states) @ Vt[:3].T
    hallu_proj = np.array(hallu_states) @ Vt[:3].T

    print(f"  Top 5 singular values: {S[:5].round(1)}")
    print(f"  SV ratio (S1/S2): {S[0]/S[1]:.2f}")
    print(f"  SV ratio (S1/S5): {S[0]/S[4]:.2f}")

    # Explained variance
    sv_sq = S**2
    total_var = sv_sq.sum()
    cum_var = np.cumsum(sv_sq) / total_var
    dim_95 = int(np.searchsorted(cum_var, 0.95)) + 1
    dim_99 = int(np.searchsorted(cum_var, 0.99)) + 1
    print(f"  95% variance in {dim_95} dims, 99% in {dim_99} dims")

    # === P43c: Separation in PCA space ===
    print("\n[P43c] Fact-hallu separation in PCA space...")
    fact_center = fact_proj.mean(axis=0)
    hallu_center = hallu_proj.mean(axis=0)
    separation = np.linalg.norm(fact_center - hallu_center)
    fact_spread = np.mean([np.linalg.norm(f - fact_center) for f in fact_proj])
    hallu_spread = np.mean([np.linalg.norm(h - hallu_center) for h in hallu_proj])
    print(f"  Center separation: {separation:.3f}")
    print(f"  Fact spread:  {fact_spread:.3f}")
    print(f"  Hallu spread: {hallu_spread:.3f}")
    print(f"  Separation ratio: {separation / (fact_spread + hallu_spread + 1e-12):.2f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    layers = sorted(dim_results.keys())
    f_dims = [dim_results[l]['fact_dim_90'] for l in layers]
    h_dims = [dim_results[l]['hallu_dim_90'] for l in layers]
    x = range(len(layers))
    axes[0].bar([i-0.2 for i in x], f_dims, 0.4, label='Fact', color='green', alpha=0.7)
    axes[0].bar([i+0.2 for i in x], h_dims, 0.4, label='Hallu', color='red', alpha=0.7)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([f'L{l}' for l in layers])
    axes[0].set_ylabel('Effective Dim (90%)')
    axes[0].set_title('Manifold Dimension by Layer')
    axes[0].legend()

    # PCA scatter
    axes[1].scatter(fact_proj[:, 0], fact_proj[:, 1], c='green', alpha=0.6, s=40, label='Fact')
    axes[1].scatter(hallu_proj[:, 0], hallu_proj[:, 1], c='red', alpha=0.6, s=40, label='Hallu')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('PCA Projection (L11)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Explained variance
    axes[2].plot(range(1, 21), cum_var[:20]*100, 'b.-', linewidth=2)
    axes[2].axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90%')
    axes[2].axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='95%')
    axes[2].set_xlabel('Dimensions')
    axes[2].set_ylabel('Cumulative Variance (%)')
    axes[2].set_title(f'PCA Spectrum (95% in {dim_95}D)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 43: Truth Manifold Dimensionality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase43_manifold.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 43, 'name': 'Truth Manifold Dimensionality',
        'inspiration': 'Rosetta 5D manifold + Holographic Principle',
        'dim_by_layer': dim_results,
        'pca_95_dims': dim_95, 'pca_99_dims': dim_99,
        'separation': round(separation, 3),
        'fact_spread': round(fact_spread, 3),
        'hallu_spread': round(hallu_spread, 3),
    }
    with open(os.path.join(RESULTS_DIR, 'phase43_manifold.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 43 RESULTS: Truth Manifold Dimensionality")
    print("=" * 70)
    for l in layers:
        d = dim_results[l]
        print(f"  L{l:>2d}: fact={d['fact_dim_90']:>3d}D hallu={d['hallu_dim_90']:>3d}D")
    print(f"  95% variance in {dim_95} dims")
    print(f"  Separation ratio: {separation / (fact_spread + hallu_spread + 1e-12):.2f}")
    print("=" * 70)
    phase_complete(43)

if __name__ == '__main__':
    main()
