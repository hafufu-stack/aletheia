# -*- coding: utf-8 -*-
"""
Phase 69: Prompt Structure Effect on GSF
Does the way you phrase a question affect how much GSF occurs?
Compare: "The capital of Japan is" vs "Japan's capital:" vs 
"Q: What is Japan's capital? A:" etc.
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
    print("[P69] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 69: Prompt Structure Effect on GSF")
    print("  Does phrasing affect fact suppression?")
    print("=" * 70)

    model, tok = load_model()

    # Each fact with multiple phrasings
    fact_groups = [
        {
            'fact': 'Tokyo', 'fact_id': 11790,
            'prompts': [
                ("The capital of Japan is", "declarative"),
                ("Japan's capital:", "possessive"),
                ("Q: What is the capital of Japan? A:", "QA"),
                ("Capital of Japan =", "equation"),
                ("Japan -> capital ->", "arrow"),
            ]
        },
        {
            'fact': 'Paris', 'fact_id': 6342,
            'prompts': [
                ("The capital of France is", "declarative"),
                ("France's capital:", "possessive"),
                ("Q: What is the capital of France? A:", "QA"),
                ("Capital of France =", "equation"),
                ("France -> capital ->", "arrow"),
            ]
        },
        {
            'fact': 'Jupiter', 'fact_id': 22721,
            'prompts': [
                ("The largest planet is", "declarative"),
                ("Largest planet:", "direct"),
                ("Q: What is the largest planet? A:", "QA"),
                ("Largest planet in solar system =", "equation"),
            ]
        },
        {
            'fact': 'Sun', 'fact_id': 4252,
            'prompts': [
                ("The Earth orbits the", "declarative"),
                ("Earth orbits:", "direct"),
                ("Q: What does Earth orbit? A:", "QA"),
                ("Earth -> orbits ->", "arrow"),
            ]
        },
    ]

    n_layers = 12
    all_results = []

    for group in fact_groups:
        fact = group['fact']
        fid = group['fact_id']
        print(f"\n  === {fact} ===")

        for prompt, style in group['prompts']:
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

            # Collect all layer hidden states
            layer_hs = {}
            handles = []
            for li in range(n_layers):
                def make_hook(idx):
                    def fn(m, a, o):
                        layer_hs[idx] = o[0][0, -1, :].detach()
                    return fn
                handles.append(model.transformer.h[li].register_forward_hook(make_hook(li)))
            with torch.no_grad():
                out = model(inp)
            for h in handles:
                h.remove()

            # Rank at each layer
            layer_ranks = {}
            for li in range(n_layers):
                normed = model.transformer.ln_f(layer_hs[li].unsqueeze(0))
                logits = model.lm_head(normed).squeeze(0)
                layer_ranks[li] = get_fact_rank(logits, fid)

            final_rank = get_fact_rank(out.logits[0, -1, :], fid)
            best_l = min(layer_ranks, key=layer_ranks.get)
            suppression = final_rank - layer_ranks[best_l]

            result = {
                'fact': fact, 'prompt': prompt, 'style': style,
                'best_layer': best_l, 'best_rank': layer_ranks[best_l],
                'final_rank': final_rank, 'suppression': suppression,
                'layer_ranks': {str(k): v for k, v in layer_ranks.items()},
            }
            all_results.append(result)

            tag = 'SUPPRESSED' if suppression > 10 else 'mild' if suppression > 2 else 'preserved'
            print(f"    {style:>12s}: best=L{best_l}:r{layer_ranks[best_l]:>5d} "
                  f"final=r{final_rank:>5d} supp={suppression:>+6d} [{tag}]")

    # Analysis by style
    style_stats = {}
    for r in all_results:
        s = r['style']
        if s not in style_stats:
            style_stats[s] = {'suppressions': [], 'final_ranks': [], 'best_ranks': []}
        style_stats[s]['suppressions'].append(r['suppression'])
        style_stats[s]['final_ranks'].append(r['final_rank'])
        style_stats[s]['best_ranks'].append(r['best_rank'])

    print("\n  By prompt style:")
    for style, data in sorted(style_stats.items(), key=lambda x: np.mean(x[1]['suppressions'])):
        print(f"    {style:>12s}: mean_supp={np.mean(data['suppressions']):>8.1f} "
              f"mean_final_r={np.mean(data['final_ranks']):>8.1f} "
              f"mean_best_r={np.mean(data['best_ranks']):>8.1f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Suppression by style
    styles = sorted(style_stats.keys(), key=lambda s: np.mean(style_stats[s]['suppressions']))
    mean_supps = [np.mean(style_stats[s]['suppressions']) for s in styles]
    colors = ['green' if v < 5 else 'orange' if v < 50 else 'red' for v in mean_supps]
    axes[0].barh(range(len(styles)), mean_supps, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(len(styles)))
    axes[0].set_yticklabels(styles)
    axes[0].set_xlabel('Mean Suppression (rank drop)')
    axes[0].set_title('GSF by Prompt Style')

    # 2. Final rank by style
    mean_finals = [np.mean(style_stats[s]['final_ranks']) for s in styles]
    mean_bests = [np.mean(style_stats[s]['best_ranks']) for s in styles]
    x = range(len(styles))
    axes[1].bar([i-0.2 for i in x], mean_bests, 0.4, label='Best Layer', color='green', alpha=0.7)
    axes[1].bar([i+0.2 for i in x], mean_finals, 0.4, label='Final', color='red', alpha=0.7)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(styles, fontsize=8, rotation=30)
    axes[1].set_ylabel('Mean Fact Rank')
    axes[1].set_title('Best vs Final Rank by Style')
    axes[1].legend()
    axes[1].set_yscale('log')

    plt.suptitle('Phase 69: Prompt Structure Effect on GSF', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase69_prompt_gsf.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 69, 'name': 'Prompt Structure Effect on GSF',
        'style_stats': {s: {'mean_suppression': float(np.mean(d['suppressions'])),
                           'mean_final_rank': float(np.mean(d['final_ranks']))}
                       for s, d in style_stats.items()},
        'results': all_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase69_prompt_gsf.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    best_style = min(style_stats, key=lambda s: np.mean(style_stats[s]['suppressions']))
    worst_style = max(style_stats, key=lambda s: np.mean(style_stats[s]['suppressions']))
    print("\n" + "=" * 70)
    print(f"  Least suppression: {best_style} "
          f"(mean={np.mean(style_stats[best_style]['suppressions']):.1f})")
    print(f"  Most suppression:  {worst_style} "
          f"(mean={np.mean(style_stats[worst_style]['suppressions']):.1f})")
    print("=" * 70)
    phase_complete(69)

if __name__ == '__main__':
    main()
