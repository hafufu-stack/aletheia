# -*- coding: utf-8 -*-
"""
Phase 67: Residual Fact Bypass
Instead of ablating suppressor heads, ADD a residual connection
from L10 directly to the output, bypassing L11-L12.
Like a skip connection for facts. Test different mixing ratios.
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
    print("[P67] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 67: Residual Fact Bypass")
    print("  Skip connection: L10 -> output (bypassing grammar layers)")
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

    # Sweep mixing ratios: final_logits = (1-alpha)*L12 + alpha*L10
    print("\n[P67] Sweeping residual mixing ratio...")
    sweep_results = {}
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        correct = 0
        ranks = []
        for prompt, fact_ids, expected in tests:
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

            h10 = {}
            def hook(m, a, o):
                h10['h'] = o[0][0, -1, :].detach()
            handle = model.transformer.h[10].register_forward_hook(hook)
            with torch.no_grad():
                out = model(inp)
            handle.remove()

            final_logits = out.logits[0, -1, :]
            normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
            l10_logits = model.lm_head(normed).squeeze(0)

            # Residual bypass: mix L10 and L12
            mixed = (1 - alpha) * final_logits + alpha * l10_logits

            rank = get_fact_rank(mixed, fact_ids[0])
            ranks.append(rank)
            if torch.argmax(mixed).item() in fact_ids:
                correct += 1

        acc = correct / len(tests)
        med_rank = float(np.median(ranks))
        sweep_results[alpha] = {'accuracy': acc, 'median_rank': med_rank,
                                'mean_rank': float(np.mean(ranks))}
        print(f"  alpha={alpha:.1f}: acc={acc:.0%} median_rank={med_rank:.0f}")

    best_alpha = max(sweep_results, key=lambda a: (sweep_results[a]['accuracy'],
                                                    -sweep_results[a]['median_rank']))
    best = sweep_results[best_alpha]

    # Sweep bypass layers (which layer to bypass FROM)
    print("\n[P67b] Which layer to bypass from?")
    layer_results = {}
    best_mix = best_alpha
    for bypass_layer in range(4, 12):
        correct = 0
        ranks = []
        for prompt, fact_ids, expected in tests:
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

            h_store = {}
            def make_hook(li):
                def fn(m, a, o):
                    h_store[li] = o[0][0, -1, :].detach()
                return fn
            handle = model.transformer.h[bypass_layer].register_forward_hook(
                make_hook(bypass_layer))
            with torch.no_grad():
                out = model(inp)
            handle.remove()

            final_logits = out.logits[0, -1, :]
            normed = model.transformer.ln_f(h_store[bypass_layer].unsqueeze(0))
            layer_logits = model.lm_head(normed).squeeze(0)

            mixed = (1 - best_mix) * final_logits + best_mix * layer_logits
            rank = get_fact_rank(mixed, fact_ids[0])
            ranks.append(rank)
            if torch.argmax(mixed).item() in fact_ids:
                correct += 1

        acc = correct / len(tests)
        layer_results[bypass_layer] = {'accuracy': acc, 'median_rank': float(np.median(ranks))}
        print(f"  L{bypass_layer} bypass (alpha={best_mix:.1f}): acc={acc:.0%} "
              f"median_rank={np.median(ranks):.0f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Alpha sweep
    alphas = sorted(sweep_results.keys())
    accs = [sweep_results[a]['accuracy']*100 for a in alphas]
    meds = [sweep_results[a]['median_rank'] for a in alphas]
    axes[0].plot(alphas, accs, 'g.-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Alpha (L10 weight)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title(f'Residual Bypass: Accuracy\n(best alpha={best_alpha:.1f})')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=best_alpha, color='red', linestyle='--', alpha=0.5)

    # 2. Median rank sweep
    axes[1].plot(alphas, meds, 'b.-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Alpha (L10 weight)')
    axes[1].set_ylabel('Median Fact Rank')
    axes[1].set_title('Residual Bypass: Median Rank')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    # 3. Layer comparison
    bypass_layers = sorted(layer_results.keys())
    layer_accs = [layer_results[l]['accuracy']*100 for l in bypass_layers]
    axes[2].bar(range(len(bypass_layers)), layer_accs, color='green', alpha=0.7,
               edgecolor='black')
    axes[2].set_xticks(range(len(bypass_layers)))
    axes[2].set_xticklabels([f'L{l}' for l in bypass_layers])
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title(f'Bypass Layer Comparison\n(alpha={best_mix:.1f})')

    plt.suptitle('Phase 67: Residual Fact Bypass', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase67_residual_bypass.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 67, 'name': 'Residual Fact Bypass',
        'best_alpha': best_alpha,
        'best_accuracy': best['accuracy'],
        'best_median_rank': best['median_rank'],
        'alpha_sweep': {str(k): v for k, v in sweep_results.items()},
        'layer_sweep': {str(k): v for k, v in layer_results.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase67_residual_bypass.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  Best alpha: {best_alpha:.1f} -> acc={best['accuracy']:.0%}")
    best_layer = max(layer_results, key=lambda l: layer_results[l]['accuracy'])
    print(f"  Best bypass layer: L{best_layer} -> acc={layer_results[best_layer]['accuracy']:.0%}")
    print("=" * 70)
    phase_complete(67)

if __name__ == '__main__':
    main()
