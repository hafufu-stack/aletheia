# -*- coding: utf-8 -*-
"""
Phase 75: The Dark Matter Hypothesis
Why does math bypass GSF while facts don't?
Analyze the geometric relationship between token embeddings
and L11H7's weight matrices. Are math tokens orthogonal to
the grammar subspace?
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
    print("[P75] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def main():
    print("=" * 70)
    print("  Phase 75: The Dark Matter Hypothesis")
    print("  Are math tokens orthogonal to grammar heads?")
    print("=" * 70)

    model, tok = load_model()
    n_heads = model.config.n_head  # 12
    head_dim = model.config.n_embd // n_heads  # 64

    # Get L11's attention weight matrices
    # GPT-2: c_attn projects to Q,K,V concatenated
    l11_attn = model.transformer.h[11].attn
    # c_attn.weight shape: (768, 2304) = (hidden, 3*hidden)
    w_attn = l11_attn.c_attn.weight.data  # (768, 2304)
    # Split into Q, K, V
    w_q, w_k, w_v = w_attn.split(model.config.n_embd, dim=1)
    # Extract H7's slice (head_dim=64, head 7 = indices 448:512)
    h7_start = 7 * head_dim
    h7_end = h7_start + head_dim
    w_q_h7 = w_q[:, h7_start:h7_end]  # (768, 64)
    w_k_h7 = w_k[:, h7_start:h7_end]
    w_v_h7 = w_v[:, h7_start:h7_end]

    # Token embeddings
    embeddings = model.transformer.wte.weight.data  # (vocab, 768)

    # Define token categories
    fact_tokens = {
        'Tokyo': 11790, 'Paris': 6342, 'Jupiter': 22721, 'Sun': 4252,
        'Au': 7591, 'Hamlet': 13483, 'relativity': 44449,
    }
    math_tokens = {
        '0': 15, '1': 16, '2': 17, '3': 18, '4': 19, '5': 20,
        '6': 21, '7': 22, '8': 23, '9': 24, '+': 10, '=': 28,
    }
    grammar_tokens_map = {}
    for w in [' the', ' a', ' is', ' was', ' to', ' of', ' in', ' that']:
        ids = tok.encode(w)
        if ids:
            grammar_tokens_map[w.strip()] = ids[0]

    categories = {
        'fact': fact_tokens,
        'math': math_tokens,
        'grammar': grammar_tokens_map,
    }

    # Compute projection magnitude onto H7's Q,K,V subspaces
    results = {}
    for cat_name, token_map in categories.items():
        projections = {'Q': [], 'K': [], 'V': []}
        for tok_name, tok_id in token_map.items():
            emb = embeddings[tok_id]  # (768,)
            # Project onto H7's subspaces
            proj_q = torch.norm(emb @ w_q_h7).item()
            proj_k = torch.norm(emb @ w_k_h7).item()
            proj_v = torch.norm(emb @ w_v_h7).item()
            projections['Q'].append(proj_q)
            projections['K'].append(proj_k)
            projections['V'].append(proj_v)

        results[cat_name] = {
            'Q_mean': float(np.mean(projections['Q'])),
            'K_mean': float(np.mean(projections['K'])),
            'V_mean': float(np.mean(projections['V'])),
            'Q_std': float(np.std(projections['Q'])),
            'K_std': float(np.std(projections['K'])),
            'V_std': float(np.std(projections['V'])),
            'total_mean': float(np.mean(projections['Q']) + np.mean(projections['K']) + np.mean(projections['V'])),
        }

    print("\n  L11H7 Projection Magnitudes:")
    print(f"  {'Category':>10s} | {'Q_mean':>8s} | {'K_mean':>8s} | {'V_mean':>8s} | {'Total':>8s}")
    print("  " + "-" * 55)
    for cat, data in results.items():
        print(f"  {cat:>10s} | {data['Q_mean']:>8.2f} | {data['K_mean']:>8.2f} | "
              f"{data['V_mean']:>8.2f} | {data['total_mean']:>8.2f}")

    # Also compute for L10H5 (top helper) for comparison
    w_q_h5_l10 = model.transformer.h[10].attn.c_attn.weight.data[:, :model.config.n_embd][:, 5*head_dim:(5+1)*head_dim]
    helper_results = {}
    for cat_name, token_map in categories.items():
        projs = []
        for tok_name, tok_id in token_map.items():
            emb = embeddings[tok_id]
            projs.append(torch.norm(emb @ w_q_h5_l10).item())
        helper_results[cat_name] = float(np.mean(projs))

    print("\n  L10H5 (Helper) Q Projection:")
    for cat, val in helper_results.items():
        print(f"  {cat:>10s}: {val:.2f}")

    # Cosine similarity between category centroids and H7 principal direction
    # H7's principal direction = top singular vector of W_V_H7
    U, S, Vt = torch.svd(w_v_h7)
    h7_principal = U[:, 0]  # (768,) - input-space direction that H7 amplifies most

    centroid_cosines = {}
    for cat_name, token_map in categories.items():
        embs = torch.stack([embeddings[tid] for tid in token_map.values()])
        centroid = embs.mean(dim=0)
        centroid = centroid / (torch.norm(centroid) + 1e-12)
        cos = torch.dot(centroid, h7_principal).item()
        centroid_cosines[cat_name] = cos
        print(f"\n  cos(centroid_{cat_name}, H7_principal) = {cos:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Projection magnitudes by category
    cats = list(results.keys())
    q_vals = [results[c]['Q_mean'] for c in cats]
    k_vals = [results[c]['K_mean'] for c in cats]
    v_vals = [results[c]['V_mean'] for c in cats]
    x = range(len(cats))
    w = 0.25
    axes[0].bar([i-w for i in x], q_vals, w, label='Q', color='blue', alpha=0.7)
    axes[0].bar(list(x), k_vals, w, label='K', color='green', alpha=0.7)
    axes[0].bar([i+w for i in x], v_vals, w, label='V', color='red', alpha=0.7)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(cats)
    axes[0].set_ylabel('Mean Projection Magnitude')
    axes[0].set_title('L11H7 Projection by Token Category')
    axes[0].legend()

    # 2. Cosine similarity
    cos_vals = [centroid_cosines[c] for c in cats]
    colors = ['red' if abs(v) > 0.05 else 'green' for v in cos_vals]
    axes[1].bar(range(len(cats)), cos_vals, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(cats)))
    axes[1].set_xticklabels(cats)
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title("Alignment with H7's\nPrincipal Direction")
    axes[1].axhline(y=0, color='black', linewidth=0.5)

    # 3. Total projection: fact vs math
    totals = [results[c]['total_mean'] for c in cats]
    colors3 = ['red', 'blue', 'gray']
    axes[2].bar(range(len(cats)), totals, color=colors3, alpha=0.7, edgecolor='black')
    axes[2].set_xticks(range(len(cats)))
    axes[2].set_xticklabels(cats)
    axes[2].set_ylabel('Total QKV Projection')
    axes[2].set_title('Total Interaction with L11H7')

    plt.suptitle('Phase 75: The Dark Matter Hypothesis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase75_dark_matter.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Is math more orthogonal?
    math_total = results['math']['total_mean']
    fact_total = results['fact']['total_mean']
    dark_matter_confirmed = math_total < fact_total

    output = {
        'phase': 75, 'name': 'The Dark Matter Hypothesis',
        'projection_results': results,
        'centroid_cosines': centroid_cosines,
        'helper_projections': helper_results,
        'dark_matter_confirmed': dark_matter_confirmed,
        'math_total_projection': math_total,
        'fact_total_projection': fact_total,
    }
    with open(os.path.join(RESULTS_DIR, 'phase75_dark_matter.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  Dark Matter confirmed: {dark_matter_confirmed}")
    print(f"  Math total projection: {math_total:.2f}")
    print(f"  Fact total projection: {fact_total:.2f}")
    if dark_matter_confirmed:
        print("  -> Math tokens interact LESS with L11H7 (Grammar Police)")
        print("  -> Math is 'dark matter' passing through grammar undetected!")
    else:
        print("  -> Math tokens interact equally or MORE with L11H7")
        print("  -> Immunity mechanism is not geometric orthogonality")
    print("=" * 70)
    phase_complete(75)

if __name__ == '__main__':
    main()
