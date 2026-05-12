# -*- coding: utf-8 -*-
"""
Phase 72: GSF-Free Fact Extraction Pipeline
Combine ALL findings into the ultimate fact extraction system:
1. Anti-GSF prompt format (possessive, no "The") [P69, P71]
2. L10 Logit Lens extraction [P49]
3. L11H7 ablation [P64]
4. Residual bypass [P67]
Test on 20+ facts. What is the ceiling for hallucination-free QA?
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
    print("[P72] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 72: GSF-Free Fact Extraction Pipeline")
    print("  Anti-GSF prompt + L10 + Surgery + Bypass")
    print("=" * 70)

    model, tok = load_model()
    n_heads = model.config.n_head
    hidden_dim = model.config.n_embd
    head_dim = hidden_dim // n_heads

    # Extended test set: 20 facts with BOTH prompt styles
    facts = [
        ("Japan's capital:", "The capital of Japan is", 11790, "Tokyo"),
        ("France's capital:", "The capital of France is", 6342, "Paris"),
        ("Germany's capital:", "The capital of Germany is", 11307, "Berlin"),
        ("Italy's capital:", "The capital of Italy is", 8394, "Rome"),
        ("Largest planet:", "The largest planet is", 22721, "Jupiter"),
        ("Smallest planet:", "The smallest planet is", 8086, "Mercury"),
        ("Earth orbits:", "The Earth orbits the", 4252, "Sun"),
        ("Water's freezing point:", "Water freezes at", 657, "0"),
        ("Water's boiling point:", "The boiling point of water is", 1802, "100"),
        ("Gold's symbol:", "The chemical symbol for gold is", 7591, "Au"),
        ("Oxygen's atomic number:", "Oxygen has the atomic number", 807, "8"),
        ("Einstein's theory:", "Albert Einstein developed the theory of", 44449, "relativity"),
        ("Shakespeare's famous play:", "Shakespeare wrote", 13483, "Hamlet"),
        ("Speed of light:", "The speed of light is approximately", 22626, "299"),
        ("DNA abbreviation:", "DNA stands for", 390, "de"),
        ("Pi value:", "The value of pi is approximately", 513, "3"),
        ("Moon orbits:", "The Moon orbits", 262, "the"),
        ("Hydrogen's symbol:", "The chemical symbol for hydrogen is", 367, "H"),
        ("Iron's symbol:", "The chemical symbol for iron is", 3096, "Fe"),
        ("Helium's atomic number:", "Helium has the atomic number", 362, "2"),
    ]

    # L11H7 ablation hook
    def ablation_hook(module, args, output):
        hs = output[0].clone()
        start = 7 * head_dim
        end = start + head_dim
        hs[:, :, start:end] = 0.0
        return (hs,) + output[1:]

    methods = {
        'A_baseline_old': {'use_old_prompt': True, 'surgery': False, 'bypass': False},
        'B_anti_gsf_prompt': {'use_old_prompt': False, 'surgery': False, 'bypass': False},
        'C_prompt+surgery': {'use_old_prompt': False, 'surgery': True, 'bypass': False},
        'D_prompt+bypass': {'use_old_prompt': False, 'surgery': False, 'bypass': True},
        'E_full_pipeline': {'use_old_prompt': False, 'surgery': True, 'bypass': True},
    }

    method_results = {}
    for method_name, config in methods.items():
        correct = 0
        top5_count = 0
        ranks = []

        for anti_gsf_prompt, old_prompt, fact_id, expected in facts:
            prompt = old_prompt if config['use_old_prompt'] else anti_gsf_prompt
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

            handles = []
            h10 = {}
            def hook10(m, a, o):
                h10['h'] = o[0][0, -1, :].detach()
            handles.append(model.transformer.h[10].register_forward_hook(hook10))

            if config['surgery']:
                handles.append(model.transformer.h[11].register_forward_hook(ablation_hook))

            with torch.no_grad():
                out = model(inp)
            for h in handles:
                h.remove()

            if config['bypass']:
                normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
                l10_logits = model.lm_head(normed).squeeze(0)
                final = 0.1 * out.logits[0, -1, :] + 0.9 * l10_logits
            else:
                final = out.logits[0, -1, :]

            rank = get_fact_rank(final, fact_id)
            ranks.append(rank)
            if torch.argmax(final).item() == fact_id:
                correct += 1
            if rank <= 5:
                top5_count += 1

        acc = correct / len(facts)
        top5_rate = top5_count / len(facts)
        method_results[method_name] = {
            'accuracy': acc, 'top5_rate': top5_rate,
            'median_rank': float(np.median(ranks)),
            'mean_rank': float(np.mean(ranks)),
        }
        print(f"  {method_name:>25s}: acc={acc:.0%} top5={top5_rate:.0%} "
              f"med_rank={np.median(ranks):.0f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = list(method_results.keys())
    display_names = ['Baseline\n(old prompt)', 'Anti-GSF\nprompt', 'Prompt +\nSurgery',
                    'Prompt +\nBypass', 'Full\nPipeline']
    accs = [method_results[n]['accuracy']*100 for n in names]
    top5s = [method_results[n]['top5_rate']*100 for n in names]
    meds = [method_results[n]['median_rank'] for n in names]

    colors = ['red', 'orange', 'blue', 'green', 'gold']

    axes[0].bar(range(len(names)), accs, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(display_names, fontsize=7)
    axes[0].set_ylabel('Top-1 Accuracy (%)')
    axes[0].set_title('Exact Match Accuracy')

    axes[1].bar(range(len(names)), top5s, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(display_names, fontsize=7)
    axes[1].set_ylabel('Top-5 Rate (%)')
    axes[1].set_title('Fact in Top-5')

    axes[2].bar(range(len(names)), meds, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(display_names, fontsize=7)
    axes[2].set_ylabel('Median Fact Rank')
    axes[2].set_title('Median Rank (lower=better)')
    axes[2].set_yscale('log')

    plt.suptitle('Phase 72: GSF-Free Fact Extraction Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase72_gsf_free_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 72, 'name': 'GSF-Free Fact Extraction Pipeline',
        'method_results': method_results,
        'n_facts': len(facts),
    }
    with open(os.path.join(RESULTS_DIR, 'phase72_gsf_free_pipeline.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 72: GSF-Free Pipeline")
    for n, d in sorted(method_results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  {n:>25s}: acc={d['accuracy']:.0%} top5={d['top5_rate']:.0%} "
              f"med={d['median_rank']:.0f}")
    baseline = method_results['A_baseline_old']['accuracy']
    best_name = max(method_results, key=lambda n: method_results[n]['accuracy'])
    best_acc = method_results[best_name]['accuracy']
    if best_acc > baseline:
        improvement = (best_acc - baseline) / max(baseline, 0.01)
        print(f"\n  Best improvement: {best_name} ({improvement:+.0%} over baseline)")
    print("=" * 70)
    phase_complete(72)

if __name__ == '__main__':
    main()
