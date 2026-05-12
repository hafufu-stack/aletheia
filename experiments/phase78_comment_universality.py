# -*- coding: utf-8 -*-
"""
Phase 78: Comment Format Universality Test
P76 showed "# Japan capital:" gets 100% accuracy on 4 facts.
Does it scale to 20+ diverse facts? P72 showed that "anti-GSF prompts"
failed at scale. Is the comment format truly universal or a fluke?
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
    print("[P78] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 78: Comment Format Universality Test")
    print("  Does '# prompt' work on 25 facts?")
    print("=" * 70)

    model, tok = load_model()

    # 25 facts with natural and comment format
    facts = [
        ("Japan capital", 11790, "Tokyo"),
        ("France capital", 6342, "Paris"),
        ("Germany capital", 11307, "Berlin"),
        ("Italy capital", 8394, "Rome"),
        ("Spain capital", 11621, "Madrid"),
        ("Russia capital", 9587, "Moscow"),
        ("China capital", 19214, "Beijing"),
        ("largest planet", 22721, "Jupiter"),
        ("smallest planet", 8086, "Mercury"),
        ("Earth orbits", 4252, "Sun"),
        ("water freezing point", 657, "0"),
        ("water boiling point", 1802, "100"),
        ("gold symbol", 7591, "Au"),
        ("oxygen atomic number", 807, "8"),
        ("Einstein theory", 44449, "relativity"),
        ("Shakespeare play", 13483, "Hamlet"),
        ("speed of light", 22626, "299"),
        ("pi value", 513, "3"),
        ("hydrogen symbol", 367, "H"),
        ("iron symbol", 3096, "Fe"),
        ("helium atomic number", 362, "2"),
        ("newton unit", 1174, "force"),
        ("DNA full form", 390, "de"),
        ("CO2 name", 17432, "carbon"),
        ("Mars nickname", 2266, "Red"),
    ]

    results = []
    comment_correct = 0
    natural_correct = 0
    comment_top5 = 0
    natural_top5 = 0

    for desc, fact_id, expected in facts:
        comment_prompt = f"# {desc}:"
        natural_prompt = f"The {desc} is"

        for fmt_name, prompt in [('comment', comment_prompt), ('natural', natural_prompt)]:
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
            with torch.no_grad():
                out = model(inp)
            logits = out.logits[0, -1, :]
            rank = get_fact_rank(logits, fact_id)
            top1_id = torch.argmax(logits).item()
            top1 = tok.decode([top1_id]).encode('ascii', 'replace').decode().strip()
            correct = top1_id == fact_id
            in_top5 = rank <= 5

            if fmt_name == 'comment':
                if correct: comment_correct += 1
                if in_top5: comment_top5 += 1
            else:
                if correct: natural_correct += 1
                if in_top5: natural_top5 += 1

            results.append({
                'desc': desc, 'expected': expected, 'format': fmt_name,
                'rank': rank, 'top1': top1, 'correct': correct,
            })

        c_rank = results[-2]['rank']
        n_rank = results[-1]['rank']
        winner = 'COMMENT' if c_rank < n_rank else 'NATURAL' if n_rank < c_rank else 'TIE'
        print(f"  {expected:>12s}: comment=r{c_rank:>5d} natural=r{n_rank:>5d} [{winner}]")

    n = len(facts)
    print(f"\n  COMMENT: {comment_correct}/{n} correct ({comment_correct/n:.0%}), "
          f"{comment_top5}/{n} top-5 ({comment_top5/n:.0%})")
    print(f"  NATURAL: {natural_correct}/{n} correct ({natural_correct/n:.0%}), "
          f"{natural_top5}/{n} top-5 ({natural_top5/n:.0%})")

    # Head-to-head: how many facts does comment beat natural?
    comment_wins = 0
    natural_wins = 0
    ties = 0
    for i in range(0, len(results), 2):
        c_r = results[i]['rank']
        n_r = results[i+1]['rank']
        if c_r < n_r: comment_wins += 1
        elif n_r < c_r: natural_wins += 1
        else: ties += 1

    print(f"\n  Head-to-head: comment wins {comment_wins}, natural wins {natural_wins}, ties {ties}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Accuracy comparison
    methods = ['Comment\n(# format)', 'Natural\n(The...is)']
    accs = [comment_correct/n*100, natural_correct/n*100]
    top5s = [comment_top5/n*100, natural_top5/n*100]
    x = [0, 1]
    axes[0].bar([i-0.15 for i in x], accs, 0.3, label='Exact', color=['green','red'], alpha=0.7)
    axes[0].bar([i+0.15 for i in x], top5s, 0.3, label='Top-5', color=['lime','salmon'], alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel('Rate (%)')
    axes[0].set_title(f'Comment vs Natural on {n} Facts')
    axes[0].legend()

    # 2. Per-fact rank comparison
    c_ranks = [results[i]['rank'] for i in range(0, len(results), 2)]
    n_ranks = [results[i+1]['rank'] for i in range(0, len(results), 2)]
    axes[1].scatter(c_ranks, n_ranks, c='blue', alpha=0.6, edgecolors='black', s=60)
    max_r = max(max(c_ranks), max(n_ranks))
    axes[1].plot([1, max_r], [1, max_r], 'k--', alpha=0.3, label='Equal')
    axes[1].set_xlabel('Comment Format Rank')
    axes[1].set_ylabel('Natural Format Rank')
    axes[1].set_title('Per-Fact Rank Comparison')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend()

    # 3. Win distribution
    labels = ['Comment\nWins', 'Natural\nWins', 'Ties']
    vals = [comment_wins, natural_wins, ties]
    colors = ['green', 'red', 'gray']
    axes[2].bar(range(3), vals, color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel('Count')
    axes[2].set_title('Head-to-Head Wins')

    plt.suptitle(f'Phase 78: Comment Format on {n} Facts', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase78_comment_universality.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 78, 'name': 'Comment Format Universality',
        'n_facts': n,
        'comment_accuracy': comment_correct / n,
        'natural_accuracy': natural_correct / n,
        'comment_top5': comment_top5 / n,
        'natural_top5': natural_top5 / n,
        'comment_wins': comment_wins, 'natural_wins': natural_wins, 'ties': ties,
        'results': results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase78_comment_universality.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  Comment: {comment_correct/n:.0%} exact, {comment_top5/n:.0%} top-5")
    print(f"  Natural: {natural_correct/n:.0%} exact, {natural_top5/n:.0%} top-5")
    print(f"  Head-to-head: {comment_wins}-{natural_wins}-{ties}")
    print("=" * 70)
    phase_complete(78)

if __name__ == '__main__':
    main()
