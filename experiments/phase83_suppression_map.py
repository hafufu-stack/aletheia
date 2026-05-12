# -*- coding: utf-8 -*-
"""
Phase 83: The Suppression Gradient Map
Create a complete heatmap: for EVERY layer (0-11) x EVERY head (0-11),
measure the suppression score when ablated. P61 only tested L10-L11.
This reveals the full spatial map of fact suppression across the
entire Transformer.
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
    print("[P83] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 83: The Suppression Gradient Map")
    print("  Full 12x12 heatmap of head ablation effects")
    print("=" * 70)

    model, tok = load_model()
    n_layers = 12
    n_heads = 12
    hidden_dim = model.config.n_embd
    head_dim = hidden_dim // n_heads

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("The largest planet is", [22721], "Jupiter"),
        ("The Earth orbits the", [4252], "Sun"),
        ("Water freezes at", [657], "0"),
        ("Shakespeare wrote", [13483], "Hamlet"),
    ]

    # Get baseline ranks
    print("[P83] Computing baseline ranks...")
    baseline_ranks = {}
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        with torch.no_grad():
            out = model(inp)
        baseline_ranks[expected] = get_fact_rank(out.logits[0, -1, :], fact_ids[0])

    # Full ablation grid: layer x head
    print("[P83] Running full 12x12 ablation grid...")
    suppression_grid = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            total_improvement = 0
            for prompt, fact_ids, expected in tests:
                inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

                def make_hook(hi, hd):
                    def fn(module, args, output):
                        hs = output[0].clone()
                        start = hi * hd
                        end = start + hd
                        hs[:, :, start:end] = 0.0
                        return (hs,) + output[1:]
                    return fn

                handle = model.transformer.h[layer_idx].register_forward_hook(
                    make_hook(head_idx, head_dim))
                with torch.no_grad():
                    out = model(inp)
                handle.remove()

                ablated_rank = get_fact_rank(out.logits[0, -1, :], fact_ids[0])
                improvement = baseline_ranks[expected] - ablated_rank
                total_improvement += improvement

            avg_improvement = total_improvement / len(tests)
            suppression_grid[layer_idx, head_idx] = avg_improvement

        # Progress indicator
        best_h = np.argmax(suppression_grid[layer_idx])
        worst_h = np.argmin(suppression_grid[layer_idx])
        print(f"  L{layer_idx:>2d}: best_ablation=H{best_h}({suppression_grid[layer_idx, best_h]:>+.1f}) "
              f"worst=H{worst_h}({suppression_grid[layer_idx, worst_h]:>+.1f})")

    # Find global top suppressors and helpers
    flat = [(suppression_grid[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
    flat.sort(reverse=True)
    top_suppressors = flat[:10]  # Most improvement when ablated = suppressors
    top_helpers = flat[-10:]     # Most damage when ablated = helpers

    print("\n  TOP 10 SUPPRESSORS (ablation improves facts):")
    for score, l, h in top_suppressors:
        print(f"    L{l}H{h}: {score:>+8.1f}")
    print("\n  TOP 10 HELPERS (ablation hurts facts):")
    for score, l, h in top_helpers:
        print(f"    L{l}H{h}: {score:>+8.1f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Full heatmap
    im = axes[0].imshow(suppression_grid, cmap='RdYlGn', aspect='equal',
                        vmin=-max(abs(suppression_grid.min()), abs(suppression_grid.max())),
                        vmax=max(abs(suppression_grid.min()), abs(suppression_grid.max())))
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')
    axes[0].set_title('Suppression Map\n(green=suppressor, red=helper)')
    axes[0].set_xticks(range(n_heads))
    axes[0].set_yticks(range(n_layers))
    plt.colorbar(im, ax=axes[0], shrink=0.8, label='Rank improvement when ablated')

    # 2. Per-layer total suppression
    layer_totals = suppression_grid.sum(axis=1)
    colors = ['green' if v > 0 else 'red' for v in layer_totals]
    axes[1].barh(range(n_layers), layer_totals, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_yticks(range(n_layers))
    axes[1].set_yticklabels([f'L{i}' for i in range(n_layers)])
    axes[1].set_xlabel('Total Suppression Score')
    axes[1].set_title('Net Suppression by Layer')
    axes[1].axvline(x=0, color='black', linewidth=0.5)
    axes[1].invert_yaxis()

    plt.suptitle('Phase 83: The Suppression Gradient Map', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase83_suppression_map.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Where is suppression concentrated?
    layer_totals_list = layer_totals.tolist()
    suppression_layer = int(np.argmax(layer_totals))

    output = {
        'phase': 83, 'name': 'The Suppression Gradient Map',
        'suppression_grid': suppression_grid.tolist(),
        'top_suppressors': [(float(s), int(l), int(h)) for s, l, h in top_suppressors],
        'top_helpers': [(float(s), int(l), int(h)) for s, l, h in top_helpers],
        'layer_totals': layer_totals_list,
        'peak_suppression_layer': suppression_layer,
    }
    with open(os.path.join(RESULTS_DIR, 'phase83_suppression_map.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  Peak suppression layer: L{suppression_layer}")
    print(f"  L{suppression_layer} total: {layer_totals[suppression_layer]:+.1f}")
    print(f"  #1 Suppressor: L{top_suppressors[0][1]}H{top_suppressors[0][2]} "
          f"({top_suppressors[0][0]:+.1f})")
    print(f"  #1 Helper:     L{top_helpers[0][1]}H{top_helpers[0][2]} "
          f"({top_helpers[0][0]:+.1f})")
    print("=" * 70)
    phase_complete(83)

if __name__ == '__main__':
    main()
