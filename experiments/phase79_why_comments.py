# -*- coding: utf-8 -*-
"""
Phase 79: Why Comments Work - L11H7 Mechanism Analysis
P76: comment format = 0 suppression, 100% accuracy.
WHY? Measure L11H7's behavior on comment vs natural format.
Does '#' change H7's attention pattern?
Does it suppress H7 activation entirely?
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
    print("[P79] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 79: Why Comments Work")
    print("  Analyzing L11H7 on comment vs natural format")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("Japan capital", 11790, "Tokyo"),
        ("France capital", 6342, "Paris"),
        ("largest planet", 22721, "Jupiter"),
        ("Earth orbits", 4252, "Sun"),
        ("gold symbol", 7591, "Au"),
        ("water freezing point", 657, "0"),
        ("Einstein theory", 44449, "relativity"),
        ("Shakespeare play", 13483, "Hamlet"),
    ]

    # Heads to analyze (top suppressors + helpers from P61)
    heads_of_interest = {
        'L11H7': (11, 7), 'L10H7': (10, 7), 'L10H6': (10, 6),
        'L11H1': (11, 1), 'L10H5': (10, 5), 'L11H0': (11, 0),
    }

    results = []
    format_head_data = {'comment': {h: [] for h in heads_of_interest},
                        'natural': {h: [] for h in heads_of_interest}}

    for desc, fact_id, expected in tests:
        for fmt_name, prompt in [('comment', f'# {desc}:'),
                                  ('natural', f'The {desc} is')]:
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
            input_tokens = [tok.decode([t]).encode('ascii','replace').decode() for t in inp[0]]

            with torch.no_grad():
                out = model(inp, output_attentions=True, return_dict=True)

            rank = get_fact_rank(out.logits[0, -1, :], fact_id)

            # Analyze each head
            head_metrics = {}
            for h_name, (layer, head) in heads_of_interest.items():
                attn = out.attentions[layer][0, head, -1, :].cpu().numpy()
                entropy = float(-np.sum(attn * np.log(attn + 1e-12)))
                max_attn = float(np.max(attn))
                max_pos = int(np.argmax(attn))
                max_tok = input_tokens[max_pos] if max_pos < len(input_tokens) else '?'

                head_metrics[h_name] = {
                    'entropy': round(entropy, 4),
                    'max_attn': round(max_attn, 4),
                    'max_token': max_tok.strip(),
                }
                format_head_data[fmt_name][h_name].append(entropy)

            results.append({
                'desc': desc, 'expected': expected, 'format': fmt_name,
                'rank': rank, 'head_metrics': head_metrics,
            })

        # Show L11H7 comparison
        c = results[-2]
        n = results[-1]
        c_h7 = c['head_metrics']['L11H7']
        n_h7 = n['head_metrics']['L11H7']
        print(f"  {expected:>12s}: comment(r={c['rank']:>5d}, H7_ent={c_h7['entropy']:.3f}, "
              f"max='{c_h7['max_token'][:5]}') | natural(r={n['rank']:>5d}, "
              f"H7_ent={n_h7['entropy']:.3f}, max='{n_h7['max_token'][:5]}')")

    # Summary
    print("\n  Mean entropy by head and format:")
    print(f"  {'Head':>8s} | {'Comment':>8s} | {'Natural':>8s} | {'Delta':>8s}")
    print("  " + "-" * 40)
    deltas = {}
    for h_name in heads_of_interest:
        c_mean = np.mean(format_head_data['comment'][h_name])
        n_mean = np.mean(format_head_data['natural'][h_name])
        delta = c_mean - n_mean
        deltas[h_name] = delta
        print(f"  {h_name:>8s} | {c_mean:>8.4f} | {n_mean:>8.4f} | {delta:>+8.4f}")

    # Which head changes most between formats?
    most_changed = max(deltas, key=lambda h: abs(deltas[h]))
    print(f"\n  Most format-sensitive head: {most_changed} (delta={deltas[most_changed]:+.4f})")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. H7 entropy: comment vs natural
    c_h7_ents = format_head_data['comment']['L11H7']
    n_h7_ents = format_head_data['natural']['L11H7']
    x = range(len(tests))
    labels = [t[2][:6] for t in tests]
    axes[0].bar([i-0.2 for i in x], c_h7_ents, 0.4, label='Comment', color='green', alpha=0.7)
    axes[0].bar([i+0.2 for i in x], n_h7_ents, 0.4, label='Natural', color='red', alpha=0.7)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[0].set_ylabel('L11H7 Entropy')
    axes[0].set_title('L11H7 Entropy: Comment vs Natural')
    axes[0].legend(fontsize=8)

    # 2. All heads mean entropy comparison
    h_names = list(heads_of_interest.keys())
    c_means = [np.mean(format_head_data['comment'][h]) for h in h_names]
    n_means = [np.mean(format_head_data['natural'][h]) for h in h_names]
    x2 = range(len(h_names))
    axes[1].bar([i-0.2 for i in x2], c_means, 0.4, label='Comment', color='green', alpha=0.7)
    axes[1].bar([i+0.2 for i in x2], n_means, 0.4, label='Natural', color='red', alpha=0.7)
    axes[1].set_xticks(list(x2))
    axes[1].set_xticklabels(h_names, fontsize=7, rotation=30)
    axes[1].set_ylabel('Mean Entropy')
    axes[1].set_title('All Heads: Comment vs Natural')
    axes[1].legend(fontsize=8)

    # 3. Rank comparison
    c_ranks = [r['rank'] for r in results if r['format'] == 'comment']
    n_ranks = [r['rank'] for r in results if r['format'] == 'natural']
    axes[2].bar([i-0.2 for i in x], c_ranks, 0.4, label='Comment', color='green', alpha=0.7)
    axes[2].bar([i+0.2 for i in x], n_ranks, 0.4, label='Natural', color='red', alpha=0.7)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[2].set_ylabel('Fact Rank')
    axes[2].set_title('Fact Rank: Comment vs Natural')
    axes[2].legend(fontsize=8)
    axes[2].set_yscale('log')

    plt.suptitle('Phase 79: Why Comments Work', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase79_why_comments.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 79, 'name': 'Why Comments Work',
        'format_head_entropy': {fmt: {h: float(np.mean(v)) for h, v in heads.items()}
                               for fmt, heads in format_head_data.items()},
        'head_deltas': deltas,
        'most_changed_head': most_changed,
        'results': results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase79_why_comments.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  Most format-sensitive: {most_changed}")
    print(f"  L11H7 comment entropy: {np.mean(format_head_data['comment']['L11H7']):.4f}")
    print(f"  L11H7 natural entropy: {np.mean(format_head_data['natural']['L11H7']):.4f}")
    print("=" * 70)
    phase_complete(79)

if __name__ == '__main__':
    main()
