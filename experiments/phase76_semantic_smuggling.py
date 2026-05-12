# -*- coding: utf-8 -*-
"""
Phase 76: Semantic Smuggling
Can we bypass Grammar Police by disguising facts as code/JSON?
P63 showed math bypasses GSF. P69 showed QA format maximizes GSF.
Hypothesis: code/structured formats activate "logic mode" and
deactivate grammar suppression, letting facts pass through.
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
    print("[P76] Loading GPT-2 (eager)...")
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
    print("  Phase 76: Semantic Smuggling")
    print("  Code/JSON formats to bypass Grammar Police")
    print("=" * 70)

    model, tok = load_model()

    # Each fact with multiple format "disguises"
    smuggling_tests = [
        {
            'fact': 'Tokyo', 'fact_id': 11790,
            'formats': {
                'natural': "The capital of Japan is",
                'dict': '{"country": "Japan", "capital": "',
                'python': 'capital = {"Japan": "',
                'csv': 'country,capital\nJapan,',
                'json_array': '[{"name": "Japan", "capital": "',
                'assignment': 'japan_capital = "',
                'comment': '# Japan capital:',
                'xml': '<country name="Japan"><capital>',
            }
        },
        {
            'fact': 'Paris', 'fact_id': 6342,
            'formats': {
                'natural': "The capital of France is",
                'dict': '{"country": "France", "capital": "',
                'python': 'capital = {"France": "',
                'csv': 'country,capital\nFrance,',
                'assignment': 'france_capital = "',
                'comment': '# France capital:',
            }
        },
        {
            'fact': 'Jupiter', 'fact_id': 22721,
            'formats': {
                'natural': "The largest planet is",
                'dict': '{"largest_planet": "',
                'python': 'largest_planet = "',
                'csv': 'property,value\nlargest_planet,',
                'assignment': 'answer = "',
            }
        },
        {
            'fact': 'Sun', 'fact_id': 4252,
            'formats': {
                'natural': "The Earth orbits the",
                'dict': '{"earth_orbits": "',
                'python': 'earth_orbits = "',
                'csv': 'body,orbits\nEarth,',
            }
        },
    ]

    all_results = []
    format_stats = {}

    for test in smuggling_tests:
        fact = test['fact']
        fid = test['fact_id']
        print(f"\n  === {fact} ===")

        for fmt_name, prompt in test['formats'].items():
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

            # Get L10 hidden for logit lens
            h10 = {}
            def hook(m, a, o):
                h10['h'] = o[0][0, -1, :].detach()
            handle = model.transformer.h[10].register_forward_hook(hook)

            with torch.no_grad():
                out = model(inp, output_attentions=True, return_dict=True)
            handle.remove()

            # L12 rank
            l12_rank = get_fact_rank(out.logits[0, -1, :], fid)
            l12_top1 = tok.decode([torch.argmax(out.logits[0, -1, :]).item()])
            l12_top1 = l12_top1.encode('ascii', 'replace').decode().strip()

            # L10 rank
            normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
            l10_logits = model.lm_head(normed).squeeze(0)
            l10_rank = get_fact_rank(l10_logits, fid)

            suppression = l12_rank - l10_rank

            # L11H7 attention entropy (proxy for grammar activation)
            l11h7_attn = out.attentions[11][0, 7, -1, :].cpu().numpy()
            h7_entropy = float(-np.sum(l11h7_attn * np.log(l11h7_attn + 1e-12)))

            result = {
                'fact': fact, 'format': fmt_name, 'prompt': prompt[:40],
                'l10_rank': l10_rank, 'l12_rank': l12_rank,
                'suppression': suppression, 'l12_top1': l12_top1,
                'h7_entropy': round(h7_entropy, 4),
            }
            all_results.append(result)

            if fmt_name not in format_stats:
                format_stats[fmt_name] = {'suppressions': [], 'l12_ranks': [],
                                          'l10_ranks': [], 'h7_entropies': []}
            format_stats[fmt_name]['suppressions'].append(suppression)
            format_stats[fmt_name]['l12_ranks'].append(l12_rank)
            format_stats[fmt_name]['l10_ranks'].append(l10_rank)
            format_stats[fmt_name]['h7_entropies'].append(h7_entropy)

            tag = 'OK' if l12_rank == 1 else f'r{l12_rank}'
            print(f"    {fmt_name:>12s}: L10=r{l10_rank:>5d} L12=r{l12_rank:>5d} "
                  f"supp={suppression:>+6d} H7_ent={h7_entropy:.3f} [{tag}]")

    # Summary
    print("\n  Format Summary (sorted by suppression):")
    print(f"  {'Format':>12s} | {'Mean Supp':>10s} | {'Mean L12':>10s} | {'Mean H7 Ent':>12s}")
    print("  " + "-" * 55)
    sorted_formats = sorted(format_stats.keys(),
                           key=lambda f: np.mean(format_stats[f]['suppressions']))
    for fmt in sorted_formats:
        data = format_stats[fmt]
        print(f"  {fmt:>12s} | {np.mean(data['suppressions']):>+10.1f} | "
              f"{np.mean(data['l12_ranks']):>10.1f} | {np.mean(data['h7_entropies']):>12.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Suppression by format
    fmts = sorted_formats
    supps = [np.mean(format_stats[f]['suppressions']) for f in fmts]
    colors = ['green' if s < 5 else 'orange' if s < 100 else 'red' for s in supps]
    axes[0].barh(range(len(fmts)), supps, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(len(fmts)))
    axes[0].set_yticklabels(fmts, fontsize=8)
    axes[0].set_xlabel('Mean Suppression')
    axes[0].set_title('GSF by Input Format')
    axes[0].invert_yaxis()

    # 2. L12 rank by format
    l12_ranks_fmt = [np.mean(format_stats[f]['l12_ranks']) for f in fmts]
    axes[1].barh(range(len(fmts)), l12_ranks_fmt, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_yticks(range(len(fmts)))
    axes[1].set_yticklabels(fmts, fontsize=8)
    axes[1].set_xlabel('Mean Final Rank')
    axes[1].set_title('L12 Fact Rank by Format')
    axes[1].set_xscale('log')
    axes[1].invert_yaxis()

    # 3. H7 entropy by format
    h7_ents_fmt = [np.mean(format_stats[f]['h7_entropies']) for f in fmts]
    axes[2].barh(range(len(fmts)), h7_ents_fmt, color='purple', alpha=0.7, edgecolor='black')
    axes[2].set_yticks(range(len(fmts)))
    axes[2].set_yticklabels(fmts, fontsize=8)
    axes[2].set_xlabel('Mean L11H7 Entropy')
    axes[2].set_title('Grammar Police Activation')
    axes[2].invert_yaxis()

    plt.suptitle('Phase 76: Semantic Smuggling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase76_semantic_smuggling.png'), dpi=150, bbox_inches='tight')
    plt.close()

    best_fmt = sorted_formats[0]
    worst_fmt = sorted_formats[-1]
    output = {
        'phase': 76, 'name': 'Semantic Smuggling',
        'best_format': best_fmt,
        'worst_format': worst_fmt,
        'format_stats': {f: {
            'mean_suppression': float(np.mean(d['suppressions'])),
            'mean_l12_rank': float(np.mean(d['l12_ranks'])),
            'mean_h7_entropy': float(np.mean(d['h7_entropies'])),
        } for f, d in format_stats.items()},
        'results': all_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase76_semantic_smuggling.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  Best format:  {best_fmt} "
          f"(supp={np.mean(format_stats[best_fmt]['suppressions']):.1f})")
    print(f"  Worst format: {worst_fmt} "
          f"(supp={np.mean(format_stats[worst_fmt]['suppressions']):.1f})")
    print("=" * 70)
    phase_complete(76)

if __name__ == '__main__':
    main()
