# -*- coding: utf-8 -*-
"""
Phase 71: The "The" Hypothesis
P65: L11H7 (top suppressor) attends to "The" with weight 4.79.
P69: Possessive ("Japan's capital:") has lowest GSF (1.5).
Hypothesis: "The" token ACTIVATES the Grammar Police (L11H7).
Removing "The" from prompts reduces suppression.
Direct test: compare L11H7 activation WITH vs WITHOUT "The".
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
    print("[P71] Loading GPT-2 (eager)...")
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
    print("  Phase 71: The 'The' Hypothesis")
    print("  Does 'The' activate L11H7 and trigger GSF?")
    print("=" * 70)

    model, tok = load_model()

    # Paired prompts: WITH "The" vs WITHOUT
    pairs = [
        {
            'fact': 'Tokyo', 'fact_id': 11790,
            'with_the': "The capital of Japan is",
            'without_the': "Japan's capital is",
        },
        {
            'fact': 'Paris', 'fact_id': 6342,
            'with_the': "The capital of France is",
            'without_the': "France's capital is",
        },
        {
            'fact': 'Jupiter', 'fact_id': 22721,
            'with_the': "The largest planet is",
            'without_the': "Largest planet:",
        },
        {
            'fact': 'Sun', 'fact_id': 4252,
            'with_the': "The Earth orbits the",
            'without_the': "Earth orbits:",
        },
        {
            'fact': '100', 'fact_id': 1802,
            'with_the': "The boiling point of water is",
            'without_the': "Water's boiling point:",
        },
        {
            'fact': 'relativity', 'fact_id': 44449,
            'with_the': "Albert Einstein developed the theory of",
            'without_the': "Einstein's theory:",
        },
        {
            'fact': 'Au', 'fact_id': 7591,
            'with_the': "The chemical symbol for gold is",
            'without_the': "Gold's chemical symbol:",
        },
        {
            'fact': '0', 'fact_id': 657,
            'with_the': "Water freezes at",
            'without_the': "Water's freezing point:",
        },
    ]

    n_layers = 12
    results = []

    for pair in pairs:
        fact = pair['fact']
        fid = pair['fact_id']

        for condition, prompt in [('with_the', pair['with_the']),
                                   ('without_the', pair['without_the'])]:
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
            input_tokens = [tok.decode([t]) for t in inp[0]]

            # Check if "The" or " The" is in the tokens
            has_the_token = any('The' in t or 'the' in t for t in input_tokens)

            # Get layer-10 hidden for Logit Lens
            h10 = {}
            def hook10(m, a, o):
                h10['h'] = o[0][0, -1, :].detach()
            handle10 = model.transformer.h[10].register_forward_hook(hook10)

            with torch.no_grad():
                out = model(inp, output_attentions=True, return_dict=True)
            handle10.remove()

            # L11H7 attention to "The" tokens
            l11_attn = out.attentions[11]  # (1, 12, seq, seq)
            l11h7_attn = l11_attn[0, 7, -1, :].cpu().numpy()

            # Attention weight on "The" positions
            the_weight = 0.0
            for i, t in enumerate(input_tokens):
                if 'The' in t or 'the' in t:
                    the_weight += l11h7_attn[i]

            # L11H7 entropy
            l11h7_entropy = float(-np.sum(l11h7_attn * np.log(l11h7_attn + 1e-12)))

            # L11H7 max attention
            l11h7_max = float(np.max(l11h7_attn))

            # Fact rank at L10 vs L12
            normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
            l10_logits = model.lm_head(normed).squeeze(0)
            l10_rank = get_fact_rank(l10_logits, fid)
            l12_rank = get_fact_rank(out.logits[0, -1, :], fid)
            suppression = l12_rank - l10_rank

            result = {
                'fact': fact, 'condition': condition, 'prompt': prompt,
                'has_the_token': has_the_token,
                'l11h7_the_weight': round(the_weight, 4),
                'l11h7_entropy': round(l11h7_entropy, 4),
                'l11h7_max': round(l11h7_max, 4),
                'l10_rank': l10_rank, 'l12_rank': l12_rank,
                'suppression': suppression,
            }
            results.append(result)

    # Display paired comparisons
    print("\n  Paired comparison:")
    print(f"  {'Fact':>10s} | {'Condition':>12s} | {'H7->The':>7s} | {'L10_r':>6s} | {'L12_r':>6s} | {'Supp':>6s}")
    print("  " + "-" * 65)
    for r in results:
        print(f"  {r['fact']:>10s} | {r['condition']:>12s} | {r['l11h7_the_weight']:>7.3f} | "
              f"{r['l10_rank']:>6d} | {r['l12_rank']:>6d} | {r['suppression']:>+6d}")

    # Aggregate stats
    with_the = [r for r in results if r['condition'] == 'with_the']
    without_the = [r for r in results if r['condition'] == 'without_the']

    wt_supp = [r['suppression'] for r in with_the]
    wo_supp = [r['suppression'] for r in without_the]
    wt_h7 = [r['l11h7_the_weight'] for r in with_the]
    wo_h7 = [r['l11h7_the_weight'] for r in without_the]

    print(f"\n  WITH 'The':    mean_supp={np.mean(wt_supp):>8.1f}  mean_H7_the={np.mean(wt_h7):.4f}")
    print(f"  WITHOUT 'The': mean_supp={np.mean(wo_supp):>8.1f}  mean_H7_the={np.mean(wo_h7):.4f}")

    # Correlation: H7->"The" weight vs suppression
    all_h7 = [r['l11h7_the_weight'] for r in results]
    all_supp = [r['suppression'] for r in results]
    if len(all_h7) > 2:
        correlation = np.corrcoef(all_h7, all_supp)[0, 1]
    else:
        correlation = 0.0

    print(f"\n  Correlation(H7_the_weight, suppression) = {correlation:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Paired suppression comparison
    facts = [p['fact'] for p in pairs]
    wt_vals = [r['suppression'] for r in with_the]
    wo_vals = [r['suppression'] for r in without_the]
    x = range(len(facts))
    axes[0].bar([i-0.2 for i in x], wt_vals, 0.4, label='With "The"', color='red', alpha=0.7)
    axes[0].bar([i+0.2 for i in x], wo_vals, 0.4, label='Without "The"', color='green', alpha=0.7)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(facts, fontsize=7, rotation=45)
    axes[0].set_ylabel('Suppression (rank drop)')
    axes[0].set_title('GSF With vs Without "The"')
    axes[0].legend(fontsize=8)
    axes[0].axhline(y=0, color='black', linewidth=0.5)

    # 2. L11H7 attention to "The"
    wt_h7_vals = [r['l11h7_the_weight'] for r in with_the]
    wo_h7_vals = [r['l11h7_the_weight'] for r in without_the]
    axes[1].bar([i-0.2 for i in x], wt_h7_vals, 0.4, label='With "The"', color='red', alpha=0.7)
    axes[1].bar([i+0.2 for i in x], wo_h7_vals, 0.4, label='Without "The"', color='green', alpha=0.7)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(facts, fontsize=7, rotation=45)
    axes[1].set_ylabel('L11H7 Attention on "The"')
    axes[1].set_title('Grammar Police Activation')
    axes[1].legend(fontsize=8)

    # 3. Scatter: H7 attention vs suppression
    colors_scatter = ['red' if r['condition'] == 'with_the' else 'green' for r in results]
    axes[2].scatter(all_h7, all_supp, c=colors_scatter, alpha=0.7, edgecolors='black', s=80)
    axes[2].set_xlabel('L11H7 Attention on "The"')
    axes[2].set_ylabel('Suppression')
    axes[2].set_title(f'Correlation: r={correlation:.3f}')
    # Add trend line
    if len(all_h7) > 2 and np.std(all_h7) > 0:
        z = np.polyfit(all_h7, all_supp, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(all_h7), max(all_h7), 50)
        axes[2].plot(x_line, p(x_line), 'k--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 71: The "The" Hypothesis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase71_the_hypothesis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 71, 'name': 'The "The" Hypothesis',
        'with_the_mean_suppression': float(np.mean(wt_supp)),
        'without_the_mean_suppression': float(np.mean(wo_supp)),
        'with_the_mean_h7_activation': float(np.mean(wt_h7)),
        'without_the_mean_h7_activation': float(np.mean(wo_h7)),
        'correlation_h7_suppression': float(correlation),
        'hypothesis_confirmed': float(np.mean(wt_supp)) > float(np.mean(wo_supp)),
        'results': results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase71_the_hypothesis.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 71: The 'The' Hypothesis")
    confirmed = float(np.mean(wt_supp)) > float(np.mean(wo_supp))
    print(f"  Hypothesis confirmed: {confirmed}")
    print(f"  With 'The':    supp={np.mean(wt_supp):.1f}, H7={np.mean(wt_h7):.4f}")
    print(f"  Without 'The': supp={np.mean(wo_supp):.1f}, H7={np.mean(wo_h7):.4f}")
    print(f"  Correlation(H7, suppression) = {correlation:.4f}")
    if confirmed:
        print("  -> 'The' ACTIVATES the Grammar Police (L11H7)!")
        print("  -> Removing articles reduces fact suppression!")
    print("=" * 70)
    phase_complete(71)

if __name__ == '__main__':
    main()
