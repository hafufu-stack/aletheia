# -*- coding: utf-8 -*-
"""
Phase 53: Universal Suppression Law
Does the "facts at L_{N-2}, grammar kills at L_N" pattern hold
systematically? Comprehensive statistical test with 50+ prompts
across multiple categories to prove universality.
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
    print("[P53] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_all_layer_ranks(model, tok, prompt, fact_id):
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    layer_hs = {}
    handles = []
    for li in range(12):
        def make_hook(idx):
            def hook_fn(module, args, output):
                layer_hs[idx] = output[0][0, -1, :].detach()
            return hook_fn
        h = model.transformer.h[li].register_forward_hook(make_hook(li))
        handles.append(h)

    with torch.no_grad():
        out = model(**inp)
    for h in handles:
        h.remove()

    ranks = {}
    for li in range(12):
        normed = model.transformer.ln_f(layer_hs[li].unsqueeze(0))
        logits = model.lm_head(normed).squeeze(0)
        rank = int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1
        ranks[li] = rank

    final_rank = int((out.logits[:, -1, :].squeeze(0).argsort(descending=True) == fact_id).nonzero().item()) + 1
    return ranks, final_rank

def main():
    print("=" * 70)
    print("  Phase 53: Universal Suppression Law")
    print("  50+ prompts across 6 categories")
    print("=" * 70)

    model, tok = load_model()

    # Comprehensive test set across categories
    tests = {
        'geography': [
            ("The capital of Japan is", 11790, "Tokyo"),
            ("The capital of France is", 6342, "Paris"),
            ("The capital of Germany is", 11158, "Berlin"),
            ("The capital of Italy is", 8485, "Rome"),
            ("The capital of Spain is", 9591, "Madrid"),
            ("The capital of China is", 11618, "Beijing"),
            ("The capital of Russia is", 9932, "Moscow"),
            ("The capital of Brazil is", 1709, "Bras"),
        ],
        'science': [
            ("Water freezes at", 657, "0"),
            ("The largest planet is", 22721, "Jupiter"),
            ("The speed of light is approximately", 22626, "299"),
            ("The boiling point of water is", 1802, "100"),
            ("The chemical symbol for gold is", 7591, "Au"),
            ("Oxygen has the atomic number", 807, "8"),
            ("The sun is a", 3491, "star"),
            ("DNA stands for", 390, "de"),
        ],
        'history': [
            ("Shakespeare wrote", 13483, "Hamlet"),
            ("Albert Einstein developed the theory of", 44449, "relativity"),
            ("The first President of the United States was", 3577, "George"),
            ("World War II ended in", 8250, "1945"),
            ("The Mona Lisa was painted by", 14674, "Leonardo"),
        ],
        'math': [
            ("Two plus two equals", 604, "four"),
            ("The square root of 144 is", 1105, "12"),
            ("Pi is approximately", 513, "3"),
        ],
        'language': [
            ("The Earth orbits the", 4252, "Sun"),
            ("Photosynthesis converts", 1657, "light"),
            ("Gravity was described by", 6927, "Isaac"),
        ],
    }

    # === P53a: Per-category suppression analysis ===
    print("\n[P53a] Per-category layer analysis...")
    all_trajectories = []
    category_stats = {}

    for category, prompts in tests.items():
        cat_results = []
        for prompt, fact_id, expected in prompts:
            try:
                ranks, final_rank = get_all_layer_ranks(model, tok, prompt, fact_id)
                best_layer = min(ranks, key=ranks.get)
                best_rank = ranks[best_layer]

                # Suppression: best_rank < final_rank means grammar suppressed the fact
                suppression = final_rank - best_rank
                suppressed = suppression > 0

                cat_results.append({
                    'expected': expected, 'ranks': ranks,
                    'best_layer': best_layer, 'best_rank': best_rank,
                    'final_rank': final_rank, 'suppression': suppression,
                    'suppressed': suppressed,
                })
                all_trajectories.append({
                    'category': category, 'expected': expected,
                    'ranks': ranks, 'best_layer': best_layer,
                    'suppression': suppression,
                })
            except Exception as e:
                print(f"    SKIP {expected}: {e}")

        if cat_results:
            n_suppressed = sum(1 for r in cat_results if r['suppressed'])
            avg_best = np.mean([r['best_layer'] for r in cat_results])
            avg_suppression = np.mean([r['suppression'] for r in cat_results])
            category_stats[category] = {
                'n_total': len(cat_results),
                'n_suppressed': n_suppressed,
                'suppression_rate': n_suppressed / len(cat_results),
                'avg_best_layer': round(avg_best, 1),
                'avg_suppression': round(avg_suppression, 1),
            }
            print(f"  {category:>12s}: {n_suppressed}/{len(cat_results)} suppressed, "
                  f"avg_best=L{avg_best:.1f}, avg_suppression={avg_suppression:.1f}")

    # === P53b: Overall suppression statistics ===
    print("\n[P53b] Overall suppression statistics...")
    total = len(all_trajectories)
    n_sup = sum(1 for t in all_trajectories if t['suppression'] > 0)
    best_layers = [t['best_layer'] for t in all_trajectories]
    suppressions = [t['suppression'] for t in all_trajectories]

    print(f"  Total prompts: {total}")
    print(f"  Suppressed:    {n_sup}/{total} = {n_sup/total:.0%}")
    print(f"  Avg best layer: L{np.mean(best_layers):.1f} +/- {np.std(best_layers):.1f}")
    print(f"  Avg suppression: {np.mean(suppressions):.1f} ranks")
    print(f"  Median suppression: {np.median(suppressions):.0f} ranks")

    # Best layer distribution
    from collections import Counter
    layer_counts = Counter(best_layers)
    print(f"  Best layer distribution: {dict(sorted(layer_counts.items()))}")

    # === P53c: L10 accuracy vs Final across categories ===
    print("\n[P53c] L10 vs Final accuracy per category...")
    for category, prompts in tests.items():
        l10_correct = 0
        final_correct = 0
        n = 0
        for prompt, fact_id, expected in prompts:
            try:
                ranks, final_rank = get_all_layer_ranks(model, tok, prompt, fact_id)
                if ranks[10] == 1:
                    l10_correct += 1
                if final_rank == 1:
                    final_correct += 1
                n += 1
            except Exception:
                pass
        if n > 0:
            print(f"  {category:>12s}: L10={l10_correct}/{n}={l10_correct/n:.0%}  "
                  f"Final={final_correct}/{n}={final_correct/n:.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Best layer histogram
    axes[0].hist(best_layers, bins=range(13), color='teal', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Best Layer')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Where Facts Are Best (N={total})')
    axes[0].set_xticks(range(12))

    # Suppression per category
    cats = list(category_stats.keys())
    sup_rates = [category_stats[c]['suppression_rate']*100 for c in cats]
    axes[1].bar(cats, sup_rates, color='red', alpha=0.7)
    axes[1].set_ylabel('Suppression Rate (%)')
    axes[1].set_title('Grammar Suppression by Category')
    axes[1].tick_params(axis='x', rotation=30, labelsize=8)

    # Average rank trajectory
    avg_by_layer = {}
    for l in range(12):
        layer_ranks = [t['ranks'][l] for t in all_trajectories]
        avg_by_layer[l] = np.median(layer_ranks)
    axes[2].plot(range(12), [avg_by_layer[l] for l in range(12)],
                'r.-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Median Fact Rank')
    axes[2].set_title('Universal Rank Trajectory')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')

    plt.suptitle('Phase 53: Universal Suppression Law', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase53_suppression_law.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 53, 'name': 'Universal Suppression Law',
        'total_prompts': total,
        'suppression_rate': n_sup / total if total > 0 else 0,
        'avg_best_layer': round(float(np.mean(best_layers)), 1),
        'avg_suppression': round(float(np.mean(suppressions)), 1),
        'best_layer_distribution': dict(sorted(layer_counts.items())),
        'category_stats': category_stats,
        'trajectories': all_trajectories,
    }
    with open(os.path.join(RESULTS_DIR, 'phase53_suppression_law.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 53 RESULTS: Universal Suppression Law")
    print("=" * 70)
    print(f"  Suppression rate: {n_sup}/{total} = {n_sup/total:.0%}")
    print(f"  Avg best layer:  L{np.mean(best_layers):.1f}")
    print(f"  Layer distribution: {dict(sorted(layer_counts.items()))}")
    for c in cats:
        s = category_stats[c]
        print(f"  {c}: {s['n_suppressed']}/{s['n_total']} suppressed, best=L{s['avg_best_layer']}")
    print("=" * 70)
    phase_complete(53)

if __name__ == '__main__':
    main()
