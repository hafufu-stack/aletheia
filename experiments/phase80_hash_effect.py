# -*- coding: utf-8 -*-
"""
Phase 80: The Hash Effect - Is '#' the key or the structure?
P76: "# Japan capital:" = 100% accuracy.
Test variations: //, --, *, /*, >>, $, %, @
Also test: no symbol at all ("Japan capital:").
Is '#' special, or does ANY non-natural prefix work?
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
    print("[P80] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 80: The Hash Effect")
    print("  Is '#' special, or does ANY prefix work?")
    print("=" * 70)

    model, tok = load_model()

    facts = [
        ("Japan capital", 11790, "Tokyo"),
        ("France capital", 6342, "Paris"),
        ("largest planet", 22721, "Jupiter"),
        ("Earth orbits", 4252, "Sun"),
        ("gold symbol", 7591, "Au"),
        ("water freezing point", 657, "0"),
        ("Shakespeare play", 13483, "Hamlet"),
        ("oxygen atomic number", 807, "8"),
    ]

    # Test different prefixes
    prefixes = {
        'none': '',
        'hash': '# ',
        'double_hash': '## ',
        'double_slash': '// ',
        'dash_dash': '-- ',
        'star': '* ',
        'greater': '>> ',
        'dollar': '$ ',
        'percent': '% ',
        'at': '@ ',
        'semicolon': '; ',
        'exclaim': '! ',
        'pipe': '| ',
        'tilde': '~ ',
        'natural': 'The ',
    }

    prefix_results = {}
    for prefix_name, prefix in prefixes.items():
        correct = 0
        top5 = 0
        ranks = []

        for desc, fact_id, expected in facts:
            if prefix_name == 'natural':
                prompt = f"The {desc} is"
            else:
                prompt = f"{prefix}{desc}:"

            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
            with torch.no_grad():
                out = model(inp)
            logits = out.logits[0, -1, :]
            rank = get_fact_rank(logits, fact_id)
            ranks.append(rank)
            if torch.argmax(logits).item() == fact_id:
                correct += 1
            if rank <= 5:
                top5 += 1

        prefix_results[prefix_name] = {
            'accuracy': correct / len(facts),
            'top5_rate': top5 / len(facts),
            'median_rank': float(np.median(ranks)),
            'mean_rank': float(np.mean(ranks)),
        }

    # Sort by accuracy then median rank
    sorted_prefixes = sorted(prefix_results.keys(),
                            key=lambda p: (-prefix_results[p]['accuracy'],
                                          prefix_results[p]['median_rank']))

    print("\n  Results (sorted by accuracy):")
    print(f"  {'Prefix':>15s} | {'Acc':>6s} | {'Top5':>6s} | {'Med Rank':>9s}")
    print("  " + "-" * 45)
    for p in sorted_prefixes:
        d = prefix_results[p]
        marker = ' <--' if p == 'hash' else ''
        print(f"  {p:>15s} | {d['accuracy']:>5.0%} | {d['top5_rate']:>5.0%} | "
              f"{d['median_rank']:>9.0f}{marker}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Accuracy by prefix
    names = sorted_prefixes
    accs = [prefix_results[p]['accuracy']*100 for p in names]
    colors = ['gold' if p == 'hash' else 'red' if p == 'natural' else 'steelblue' for p in names]
    axes[0].barh(range(len(names)), accs, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_title('Fact Accuracy by Prefix')
    axes[0].invert_yaxis()

    # 2. Median rank by prefix
    meds = [prefix_results[p]['median_rank'] for p in names]
    axes[1].barh(range(len(names)), meds, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlabel('Median Fact Rank (lower=better)')
    axes[1].set_title('Median Rank by Prefix')
    axes[1].set_xscale('log')
    axes[1].invert_yaxis()

    plt.suptitle('Phase 80: The Hash Effect', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase80_hash_effect.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Is '#' unique or do other symbols work?
    hash_acc = prefix_results['hash']['accuracy']
    other_accs = {p: d['accuracy'] for p, d in prefix_results.items() if p not in ('hash', 'natural')}
    matching = [p for p, a in other_accs.items() if a >= hash_acc and hash_acc > 0]

    output = {
        'phase': 80, 'name': 'The Hash Effect',
        'prefix_results': prefix_results,
        'hash_is_unique': len(matching) == 0,
        'matching_prefixes': matching,
        'sorted_order': sorted_prefixes,
    }
    with open(os.path.join(RESULTS_DIR, 'phase80_hash_effect.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    if len(matching) == 0 and hash_acc > 0:
        print(f"  '#' IS UNIQUE: no other prefix matches its {hash_acc:.0%} accuracy")
    elif len(matching) > 0:
        print(f"  '#' is NOT unique: {matching} also achieve {hash_acc:.0%}")
    else:
        print(f"  '#' did not outperform in this test")
    print("=" * 70)
    phase_complete(80)

if __name__ == '__main__':
    main()
