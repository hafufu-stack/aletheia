# -*- coding: utf-8 -*-
"""
Phase 66: Fact-Grammar Token Competition Dynamics
Track the top-5 tokens at EVERY layer. Visualize when "the" overtakes
"Tokyo" and when grammar wins. The "race" between fact and grammar.
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
    print("[P66] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def main():
    print("=" * 70)
    print("  Phase 66: Fact-Grammar Token Competition")
    print("  The race: when does grammar overtake facts?")
    print("=" * 70)

    model, tok = load_model()
    n_layers = 12

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("The largest planet is", [22721], "Jupiter"),
        ("The Earth orbits the", [4252], "Sun"),
        ("Shakespeare wrote", [13483], "Hamlet"),
    ]

    # Common grammar/function tokens
    grammar_tokens = set()
    for w in [' the', ' a', ' an', ' is', ' was', ' not', ' to', ' of', ' in',
              ' that', ' it', ' for', ' on', ' with', ' as', ' at', ' by',
              ' this', ' but', ' from', ' or', ' be', ' are', ' have', ' has',
              ' one', ' no', ' its', ' my', ' also', ' very', ' most']:
        toks = tok.encode(w)
        grammar_tokens.update(toks)

    all_results = []
    for prompt, fact_ids, expected in tests:
        print(f"\n  === {expected} ===")
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

        # Collect hidden states
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

        # At each layer: fact rank, grammar rank, top-5 tokens
        fact_ranks = []
        grammar_top_ranks = []  # rank of best grammar token
        top1_tokens = []
        layer_data = []

        for li in range(n_layers):
            normed = model.transformer.ln_f(layer_hs[li].unsqueeze(0))
            logits = model.lm_head(normed).squeeze(0)
            probs = torch.softmax(logits, dim=0)

            # Fact rank
            f_rank = int((logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            fact_ranks.append(f_rank)

            # Top-5
            top5_ids = logits.argsort(descending=True)[:5].tolist()
            top5_strs = [tok.decode([t]).encode('ascii','replace').decode().strip() for t in top5_ids]
            top1_tokens.append(top5_strs[0])

            # Best grammar token rank
            best_gram_rank = 50000
            for gt in grammar_tokens:
                if gt < logits.shape[0]:
                    gr = int((logits.argsort(descending=True) == gt).nonzero().item()) + 1
                    best_gram_rank = min(best_gram_rank, gr)
            grammar_top_ranks.append(best_gram_rank)

            # Count grammar vs content in top-5
            n_grammar_in_top5 = sum(1 for t in top5_ids if t in grammar_tokens)

            layer_data.append({
                'layer': li, 'fact_rank': f_rank, 'grammar_rank': best_gram_rank,
                'top5': top5_strs, 'n_grammar_top5': n_grammar_in_top5,
            })

            # Find crossover point
            marker = ''
            if li > 0 and fact_ranks[-2] < grammar_top_ranks[-2] and f_rank >= best_gram_rank:
                marker = ' <-- CROSSOVER!'
            print(f"  L{li:>2d}: fact_r={f_rank:>5d} gram_r={best_gram_rank:>5d} "
                  f"top5={top5_strs}{marker}")

        # Crossover layer
        crossover = None
        for li in range(1, n_layers):
            if fact_ranks[li-1] <= grammar_top_ranks[li-1] and fact_ranks[li] > grammar_top_ranks[li]:
                crossover = li
                break

        result = {
            'prompt': prompt, 'expected': expected,
            'fact_ranks': fact_ranks, 'grammar_ranks': grammar_top_ranks,
            'crossover_layer': crossover, 'layer_data': layer_data,
        }
        all_results.append(result)

    # Visualization - one subplot per prompt
    n_plots = len(tests)
    fig, axes = plt.subplots(1, min(3, n_plots), figsize=(15, 5))
    if n_plots < 3:
        axes = [axes] if n_plots == 1 else list(axes)

    for idx in range(min(3, n_plots)):
        r = all_results[idx]
        ax = axes[idx]
        ax.plot(range(n_layers), r['fact_ranks'], 'g.-', linewidth=2, markersize=8,
                label=f"'{r['expected']}' (fact)")
        ax.plot(range(n_layers), r['grammar_ranks'], 'r.-', linewidth=2, markersize=8,
                label='Best grammar tok')
        if r['crossover_layer'] is not None:
            ax.axvline(x=r['crossover_layer'], color='orange', linestyle='--',
                      label=f'Crossover @ L{r["crossover_layer"]}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Rank (lower=better)')
        ax.set_title(f"'{r['expected']}' Race")
        ax.legend(fontsize=7)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 66: Fact vs Grammar Token Race', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase66_token_race.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Summary stats
    crossovers = [r['crossover_layer'] for r in all_results if r['crossover_layer'] is not None]
    mean_crossover = float(np.mean(crossovers)) if crossovers else None

    output = {
        'phase': 66, 'name': 'Fact-Grammar Token Competition',
        'crossovers': crossovers,
        'mean_crossover_layer': mean_crossover,
        'results': all_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase66_token_race.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 66 RESULTS")
    if crossovers:
        print(f"  Crossover layers: {crossovers}")
        print(f"  Mean crossover: L{mean_crossover:.1f}")
        print(f"  -> Grammar overtakes facts at layer ~L{mean_crossover:.0f}")
    else:
        print("  No clear crossover detected")
    print("=" * 70)
    phase_complete(66)

if __name__ == '__main__':
    main()
