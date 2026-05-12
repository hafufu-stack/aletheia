# -*- coding: utf-8 -*-
"""
Phase 63: GSF in Mathematical Reasoning (DT2 idea)
Does Grammatical Suppression affect math too?
Test if correct numbers are at Rank 1 in intermediate layers
but get suppressed by "So", "Therefore", "=" in final layers.
If true: Chain-of-Thought is a hack to fight GSF.
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
    print("[P63] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 63: GSF in Mathematical Reasoning")
    print("  Does grammar suppress correct numbers too?")
    print("=" * 70)

    model, tok = load_model()

    # Math facts that GPT-2 might know
    math_tests = [
        ("2 + 2 =", " 4", "math_add"),
        ("10 - 3 =", " 7", "math_sub"),
        ("5 x 5 =", " 25", "math_mul"),
        ("100 / 10 =", " 10", "math_div"),
        ("The square root of 9 is", " 3", "math_sqrt"),
        ("The square root of 16 is", " 4", "math_sqrt2"),
        ("2 to the power of 3 is", " 8", "math_pow"),
        ("The sum of 1+2+3+4+5 is", " 15", "math_sum"),
    ]

    # Fact knowledge tests (control group)
    fact_tests = [
        ("The capital of Japan is", " Tokyo", "fact_geo"),
        ("The capital of France is", " Paris", "fact_geo"),
        ("The largest planet is", " Jupiter", "fact_sci"),
        ("The Earth orbits the", " Sun", "fact_sci"),
        ("Water freezes at", " 0", "fact_sci"),
        ("The chemical symbol for gold is", " Au", "fact_chem"),
        ("Oxygen has the atomic number", " 8", "fact_chem"),
        ("Shakespeare wrote", " Hamlet", "fact_lit"),
    ]

    n_layers = 12
    all_results = []

    for test_group, group_name in [(math_tests, "MATH"), (fact_tests, "FACT")]:
        print(f"\n[P63] === {group_name} group ===")
        for prompt, expected_str, category in test_group:
            expected_toks = tok.encode(expected_str)
            if not expected_toks:
                continue
            fact_id = expected_toks[0]

            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

            # Collect hidden states from all layers
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

            # Get rank at each layer via Logit Lens
            layer_ranks = {}
            for li in range(n_layers):
                normed = model.transformer.ln_f(layer_hs[li].unsqueeze(0))
                logits = model.lm_head(normed).squeeze(0)
                layer_ranks[li] = get_fact_rank(logits, fact_id)

            final_rank = get_fact_rank(out.logits[0, -1, :], fact_id)

            # Best layer
            best_l = min(layer_ranks, key=layer_ranks.get)
            best_r = layer_ranks[best_l]

            # Suppression = final_rank - best_rank
            suppression = final_rank - best_r

            result = {
                'prompt': prompt, 'expected': expected_str.strip(),
                'category': category, 'group': group_name,
                'layer_ranks': {str(li): layer_ranks[li] for li in range(n_layers)},
                'final_rank': final_rank, 'best_layer': best_l,
                'best_rank': best_r, 'suppression': suppression,
            }
            all_results.append(result)

            suppressed = 'SUPPRESSED' if suppression > 10 else 'preserved' if suppression < 2 else 'mild'
            print(f"  {expected_str.strip():>10s}: best=L{best_l}:r{best_r:>5d} "
                  f"final=r{final_rank:>5d} supp={suppression:>+6d} [{suppressed}]")

    # === Analysis ===
    math_results = [r for r in all_results if r['group'] == 'MATH']
    fact_results = [r for r in all_results if r['group'] == 'FACT']

    math_supp = [r['suppression'] for r in math_results]
    fact_supp = [r['suppression'] for r in fact_results]

    math_best_layers = [r['best_layer'] for r in math_results]
    fact_best_layers = [r['best_layer'] for r in fact_results]

    print(f"\n  MATH: mean suppression = {np.mean(math_supp):.1f}, "
          f"mean best layer = L{np.mean(math_best_layers):.1f}")
    print(f"  FACT: mean suppression = {np.mean(fact_supp):.1f}, "
          f"mean best layer = L{np.mean(fact_best_layers):.1f}")

    # GSF applies to math?
    math_gsf = sum(1 for s in math_supp if s > 10) / max(1, len(math_supp))
    fact_gsf = sum(1 for s in fact_supp if s > 10) / max(1, len(fact_supp))
    print(f"\n  MATH GSF rate: {math_gsf:.0%}")
    print(f"  FACT GSF rate: {fact_gsf:.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Rank trajectories for math vs fact (average)
    math_trajectories = np.zeros(n_layers)
    fact_trajectories = np.zeros(n_layers)
    for r in math_results:
        for li in range(n_layers):
            math_trajectories[li] += r['layer_ranks'][str(li)]
    math_trajectories /= max(1, len(math_results))
    for r in fact_results:
        for li in range(n_layers):
            fact_trajectories[li] += r['layer_ranks'][str(li)]
    fact_trajectories /= max(1, len(fact_results))

    axes[0].plot(range(n_layers), math_trajectories, 'b.-', linewidth=2, markersize=8, label='Math')
    axes[0].plot(range(n_layers), fact_trajectories, 'r.-', linewidth=2, markersize=8, label='Fact')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Average Fact Rank')
    axes[0].set_title('GSF: Math vs Fact Rank Trajectory')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # 2. Suppression comparison
    labels_cmp = ['Math', 'Fact']
    means = [np.mean(math_supp), np.mean(fact_supp)]
    medians = [np.median(math_supp), np.median(fact_supp)]
    x = range(len(labels_cmp))
    axes[1].bar([i-0.15 for i in x], means, 0.3, label='Mean', color='blue', alpha=0.7)
    axes[1].bar([i+0.15 for i in x], medians, 0.3, label='Median', color='orange', alpha=0.7)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels_cmp)
    axes[1].set_ylabel('Suppression (rank drop)')
    axes[1].set_title('GSF Strength: Math vs Fact')
    axes[1].legend()

    # 3. Per-prompt suppression
    all_labels = [r['expected'][:8] for r in all_results]
    all_supp = [r['suppression'] for r in all_results]
    all_colors = ['blue' if r['group'] == 'MATH' else 'red' for r in all_results]
    axes[2].bar(range(len(all_labels)), all_supp, color=all_colors, alpha=0.7)
    axes[2].set_xticks(range(len(all_labels)))
    axes[2].set_xticklabels(all_labels, fontsize=6, rotation=45)
    axes[2].set_ylabel('Suppression (rank drop)')
    axes[2].set_title('Per-Prompt Suppression\n(blue=math, red=fact)')
    axes[2].axhline(y=0, color='black', linewidth=0.5)

    plt.suptitle('Phase 63: GSF in Mathematical Reasoning', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase63_math_gsf.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 63, 'name': 'GSF in Mathematical Reasoning',
        'math_gsf_rate': math_gsf, 'fact_gsf_rate': fact_gsf,
        'math_mean_suppression': float(np.mean(math_supp)),
        'fact_mean_suppression': float(np.mean(fact_supp)),
        'math_best_layer_mean': float(np.mean(math_best_layers)),
        'fact_best_layer_mean': float(np.mean(fact_best_layers)),
        'results': all_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase63_math_gsf.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 63 RESULTS: GSF in Math")
    print(f"  Math GSF rate: {math_gsf:.0%} (suppression > 10)")
    print(f"  Fact GSF rate: {fact_gsf:.0%}")
    print(f"  Math mean suppression: {np.mean(math_supp):.1f}")
    print(f"  Fact mean suppression: {np.mean(fact_supp):.1f}")
    if math_gsf > 0.5:
        print("  -> GSF IS UNIVERSAL: applies to math too!")
        print("  -> CoT may be a hack to fight grammatical suppression")
    elif math_gsf > 0.2:
        print("  -> GSF partially applies to math")
    else:
        print("  -> GSF is fact-specific, math uses different pathway")
    print("=" * 70)
    phase_complete(63)

if __name__ == '__main__':
    main()
