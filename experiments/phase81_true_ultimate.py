# -*- coding: utf-8 -*-
"""
Phase 81: Symbol Prefix + L10 Bypass (The TRUE Ultimate Decoder)
P72's anti-GSF pipeline failed at scale. But that was BEFORE P76-P80.
Now we know: symbol prefix scatters suppressors + L10 has truth.
Combine them: symbol prefix prompt + L10 logit lens extraction.
Two ORTHOGONAL improvements should stack multiplicatively.
Test on 25 facts (same set as P78).
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
    print("[P81] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 81: Symbol + L10 Bypass (TRUE Ultimate)")
    print("  Two orthogonal improvements combined")
    print("=" * 70)

    model, tok = load_model()

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

    methods = {
        'A_natural_L12': {'prompt_fmt': 'The {desc} is', 'use_l10': False},
        'B_natural_L10': {'prompt_fmt': 'The {desc} is', 'use_l10': True},
        'C_symbol_L12':  {'prompt_fmt': '# {desc}:', 'use_l10': False},
        'D_symbol_L10':  {'prompt_fmt': '# {desc}:', 'use_l10': True},
        'E_symbol_mix':  {'prompt_fmt': '# {desc}:', 'use_l10': 'mix'},
    }

    method_results = {}
    for method_name, config in methods.items():
        correct = 0
        top5 = 0
        ranks = []

        for desc, fact_id, expected in facts:
            prompt = config['prompt_fmt'].format(desc=desc)
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

            h10 = {}
            def hook(m, a, o):
                h10['h'] = o[0][0, -1, :].detach()
            handle = model.transformer.h[10].register_forward_hook(hook)
            with torch.no_grad():
                out = model(inp)
            handle.remove()

            if config['use_l10'] == True:
                normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
                logits = model.lm_head(normed).squeeze(0)
            elif config['use_l10'] == 'mix':
                normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
                l10_logits = model.lm_head(normed).squeeze(0)
                logits = 0.3 * out.logits[0, -1, :] + 0.7 * l10_logits
            else:
                logits = out.logits[0, -1, :]

            rank = get_fact_rank(logits, fact_id)
            ranks.append(rank)
            if torch.argmax(logits).item() == fact_id:
                correct += 1
            if rank <= 5:
                top5 += 1

        method_results[method_name] = {
            'accuracy': correct / len(facts),
            'top5_rate': top5 / len(facts),
            'median_rank': float(np.median(ranks)),
            'mean_rank': float(np.mean(ranks)),
        }
        print(f"  {method_name:>20s}: acc={correct/len(facts):.0%} "
              f"top5={top5/len(facts):.0%} med={np.median(ranks):.0f}")

    # Additional: sweep alpha for symbol+mix
    print("\n[P81b] Alpha sweep for symbol+mix...")
    alpha_sweep = {}
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        correct = 0
        ranks = []
        for desc, fact_id, expected in facts:
            prompt = f"# {desc}:"
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
            h10 = {}
            def hook(m, a, o):
                h10['h'] = o[0][0, -1, :].detach()
            handle = model.transformer.h[10].register_forward_hook(hook)
            with torch.no_grad():
                out = model(inp)
            handle.remove()
            normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
            l10_logits = model.lm_head(normed).squeeze(0)
            mixed = (1-alpha) * out.logits[0, -1, :] + alpha * l10_logits
            rank = get_fact_rank(mixed, fact_id)
            ranks.append(rank)
            if torch.argmax(mixed).item() == fact_id:
                correct += 1
        alpha_sweep[alpha] = {
            'accuracy': correct/len(facts),
            'median_rank': float(np.median(ranks)),
        }
        print(f"  alpha={alpha:.1f}: acc={correct/len(facts):.0%} "
              f"med={np.median(ranks):.0f}")

    best_alpha = max(alpha_sweep, key=lambda a: (alpha_sweep[a]['accuracy'],
                                                  -alpha_sweep[a]['median_rank']))

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Method comparison
    names = list(method_results.keys())
    display = ['Natural\nL12', 'Natural\nL10', 'Symbol\nL12', 'Symbol\nL10', 'Symbol\nMix']
    accs = [method_results[n]['accuracy']*100 for n in names]
    colors = ['red', 'orange', 'green', 'blue', 'gold']
    axes[0].bar(range(len(names)), accs, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(display, fontsize=8)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title(f'5 Methods on {len(facts)} Facts')

    # 2. Top-5 comparison
    top5s = [method_results[n]['top5_rate']*100 for n in names]
    axes[1].bar(range(len(names)), top5s, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(display, fontsize=8)
    axes[1].set_ylabel('Top-5 Rate (%)')
    axes[1].set_title('Facts in Top-5')

    # 3. Alpha sweep
    alphas = sorted(alpha_sweep.keys())
    sweep_accs = [alpha_sweep[a]['accuracy']*100 for a in alphas]
    axes[2].plot(alphas, sweep_accs, 'g.-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Alpha (L10 weight)')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title(f'Symbol + Mix: Alpha Sweep\n(best={best_alpha:.1f})')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 81: TRUE Ultimate Decoder', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase81_true_ultimate.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 81, 'name': 'Symbol + L10 Bypass (TRUE Ultimate)',
        'n_facts': len(facts),
        'method_results': method_results,
        'alpha_sweep': {str(k): v for k, v in alpha_sweep.items()},
        'best_alpha': best_alpha,
    }
    with open(os.path.join(RESULTS_DIR, 'phase81_true_ultimate.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    for n in names:
        d = method_results[n]
        print(f"  {n:>20s}: acc={d['accuracy']:.0%} top5={d['top5_rate']:.0%} "
              f"med={d['median_rank']:.0f}")
    print(f"  Best alpha for symbol+mix: {best_alpha}")
    print("=" * 70)
    phase_complete(81)

if __name__ == '__main__':
    main()
