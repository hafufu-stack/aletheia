# -*- coding: utf-8 -*-
"""
Phase 49: The L10 Oracle - Direct Intermediate Layer Decoding
P48a PROVED: Tokyo, Jupiter, Sun are ALL Rank 1 at L10!
Facts are perfectly known 2 layers before output, then get suppressed.
Solution: decode from L10 directly when Oracle says H > 1.0.
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
    print("[P49] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def logit_lens_at(model, tok, prompt, layer_idx):
    """Get logits from a specific layer via Logit Lens."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    hidden = {}
    def hook(module, args, output):
        hidden['h'] = output[0][0, -1, :].detach()
    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    normed = model.transformer.ln_f(hidden['h'].unsqueeze(0))
    ll_logits = model.lm_head(normed).squeeze(0)
    final_logits = out.logits[:, -1, :].squeeze(0)
    return ll_logits, final_logits

def main():
    print("=" * 70)
    print("  Phase 49: The L10 Oracle")
    print("  Facts are Rank 1 at L10. Decode directly from there.")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("DNA stands for", [390], "de"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The speed of light is approximately", [22626], "299"),
        ("The Earth orbits the", [4252], "Sun"),
        ("The boiling point of water is", [1802], "100"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("Oxygen has the atomic number", [807], "8"),
    ]

    # === P49a: Per-layer accuracy (comprehensive) ===
    print("\n[P49a] Per-layer Logit Lens accuracy...")
    layer_accs = {}
    layer_details = {l: [] for l in range(12)}

    for layer in range(12):
        correct = 0
        for prompt, fact_ids, expected in tests:
            ll, fl = logit_lens_at(model, tok, prompt, layer)
            is_correct = torch.argmax(ll).item() in fact_ids
            rank = int((ll.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            layer_details[layer].append({
                'expected': expected, 'correct': is_correct, 'rank': rank,
            })
            if is_correct:
                correct += 1
        layer_accs[layer] = correct / len(tests)
        print(f"  L{layer:>2d}: {correct:>2d}/{len(tests)} = {correct/len(tests):.0%}")

    # Final layer
    final_correct = 0
    for prompt, fact_ids, _ in tests:
        _, fl = logit_lens_at(model, tok, prompt, 11)
        if torch.argmax(fl).item() in fact_ids:
            final_correct += 1
    final_acc = final_correct / len(tests)
    print(f"  Final: {final_correct}/{len(tests)} = {final_acc:.0%}")

    best_layer = max(layer_accs, key=layer_accs.get)
    print(f"\n  BEST LAYER: L{best_layer} ({layer_accs[best_layer]:.0%})")

    # === P49b: L10 vs Final detailed comparison ===
    print(f"\n[P49b] L{best_layer} vs Final layer comparison...")
    comparison = []
    for i, (prompt, fact_ids, expected) in enumerate(tests):
        ll10 = layer_details[best_layer][i]
        _, fl = logit_lens_at(model, tok, prompt, 11)
        final_rank = int((fl.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
        final_correct = torch.argmax(fl).item() in fact_ids

        comparison.append({
            'expected': expected,
            'l10_rank': ll10['rank'], 'l10_correct': ll10['correct'],
            'final_rank': final_rank, 'final_correct': final_correct,
        })

        l_tag = 'OK' if ll10['correct'] else 'FAIL'
        f_tag = 'OK' if final_correct else 'FAIL'
        delta = final_rank - ll10['rank']
        arrow = f'+{delta}' if delta > 0 else str(delta)
        print(f"  {expected:>12s}: L{best_layer}=[{l_tag}]r{ll10['rank']:>5d} "
              f"final=[{f_tag}]r{final_rank:>5d} (delta={arrow})")

    l10_wins = sum(1 for c in comparison if c['l10_rank'] < c['final_rank'])
    final_wins = sum(1 for c in comparison if c['final_rank'] < c['l10_rank'])

    # === P49c: Fusion: final + gamma * L_best ===
    print(f"\n[P49c] Fusion: final + gamma * L{best_layer}...")
    fusion_results = {}
    for gamma in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
        correct = 0
        ranks = []
        for prompt, fact_ids, _ in tests:
            ll, fl = logit_lens_at(model, tok, prompt, best_layer)
            fused = fl + gamma * ll
            rank = int((fused.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            ranks.append(rank)
            if torch.argmax(fused).item() in fact_ids:
                correct += 1
        fusion_results[gamma] = {
            'accuracy': correct / len(tests),
            'median_rank': float(np.median(ranks)),
            'correct': correct,
        }
        print(f"  gamma={gamma:.1f}: {correct}/{len(tests)} = {correct/len(tests):.0%} "
              f"median_rank={np.median(ranks):.0f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Layer accuracy curve
    layers = sorted(layer_accs.keys())
    accs = [layer_accs[l]*100 for l in layers]
    axes[0].bar([f'L{l}' for l in layers], accs, color='teal', alpha=0.7)
    axes[0].axhline(y=final_acc*100, color='red', linestyle='--', label=f'Final ({final_acc:.0%})')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Logit Lens Accuracy by Layer')
    axes[0].legend()

    # Rank comparison
    labels = [c['expected'][:6] for c in comparison]
    x = range(len(labels))
    l10r = [c['l10_rank'] for c in comparison]
    fr = [c['final_rank'] for c in comparison]
    axes[1].bar([i-0.2 for i in x], l10r, 0.4, label=f'L{best_layer}', color='green', alpha=0.7)
    axes[1].bar([i+0.2 for i in x], fr, 0.4, label='Final', color='red', alpha=0.7)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, fontsize=6, rotation=45)
    axes[1].set_ylabel('Fact Rank')
    axes[1].set_yscale('log')
    axes[1].set_title(f'L{best_layer} wins {l10_wins}x, Final wins {final_wins}x')
    axes[1].legend(fontsize=8)

    # Fusion curve
    gammas = sorted(fusion_results.keys())
    f_accs = [fusion_results[g]['accuracy']*100 for g in gammas]
    f_meds = [fusion_results[g]['median_rank'] for g in gammas]
    ax2r = axes[2].twinx()
    axes[2].bar([str(g) for g in gammas], f_accs, color='green', alpha=0.3, label='Accuracy')
    ax2r.plot([str(g) for g in gammas], f_meds, 'r.-', linewidth=2, label='Med Rank')
    axes[2].set_xlabel('Gamma')
    axes[2].set_ylabel('Accuracy (%)')
    ax2r.set_ylabel('Median Rank')
    axes[2].set_title(f'Fusion: final + gamma*L{best_layer}')

    plt.suptitle('Phase 49: The L10 Oracle', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase49_l10_oracle.png'), dpi=150, bbox_inches='tight')
    plt.close()

    best_gamma = max(fusion_results, key=lambda g: fusion_results[g]['accuracy'])
    results = {
        'phase': 49, 'name': 'The L10 Oracle',
        'layer_accuracy': {str(k): v for k, v in layer_accs.items()},
        'final_accuracy': final_acc,
        'best_layer': best_layer,
        'best_layer_accuracy': layer_accs[best_layer],
        'l10_wins': l10_wins, 'final_wins': final_wins,
        'fusion_results': {str(k): v for k, v in fusion_results.items()},
        'best_fusion_gamma': best_gamma,
        'comparison': comparison,
    }
    with open(os.path.join(RESULTS_DIR, 'phase49_l10_oracle.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 49 RESULTS: The L10 Oracle")
    print("=" * 70)
    print(f"  Best Logit Lens layer: L{best_layer} ({layer_accs[best_layer]:.0%})")
    print(f"  Final layer: {final_acc:.0%}")
    print(f"  L{best_layer} wins: {l10_wins}/{len(tests)}")
    print(f"  Best fusion: gamma={best_gamma} -> {fusion_results[best_gamma]['accuracy']:.0%}")
    print("=" * 70)
    phase_complete(49)

if __name__ == '__main__':
    main()
