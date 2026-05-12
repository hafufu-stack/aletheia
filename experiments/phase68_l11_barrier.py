# -*- coding: utf-8 -*-
"""
Phase 68: The L11 Barrier - Precise Crossover Mechanics
P66 found grammar overtakes facts at EXACTLY L11.
Dissect L11: what happens WITHIN this single layer?
Split L11 into sublayer operations (Attention, MLP, LayerNorm).
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
    print("[P68] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 68: The L11 Barrier")
    print("  Dissecting the crossover layer: Attention vs MLP")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("The largest planet is", [22721], "Jupiter"),
        ("The Earth orbits the", [4252], "Sun"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("Water freezes at", [657], "0"),
        ("The boiling point of water is", [1802], "100"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("Oxygen has the atomic number", [807], "8"),
        ("The chemical symbol for gold is", [7591], "Au"),
    ]

    all_results = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

        # Collect hidden states at key points within L10 and L11
        checkpoints = {}

        # Hook after L10 (before L11)
        def hook_l10_out(m, a, o):
            checkpoints['after_L10'] = o[0][0, -1, :].detach().clone()
        h1 = model.transformer.h[10].register_forward_hook(hook_l10_out)

        # Hook into L11's attention and MLP sublayers
        # GPT-2 block: x -> LN1 -> Attn -> +x -> LN2 -> MLP -> +x
        def hook_l11_attn(m, a, o):
            # Attention output (before residual)
            checkpoints['l11_attn_out'] = o[0][0, -1, :].detach().clone()
        h2 = model.transformer.h[11].attn.register_forward_hook(hook_l11_attn)

        def hook_l11_mlp(m, a, o):
            checkpoints['l11_mlp_out'] = o[0, -1, :].detach().clone()
        h3 = model.transformer.h[11].mlp.register_forward_hook(hook_l11_mlp)

        def hook_l11_out(m, a, o):
            checkpoints['after_L11'] = o[0][0, -1, :].detach().clone()
        h4 = model.transformer.h[11].register_forward_hook(hook_l11_out)

        with torch.no_grad():
            out = model(inp)
        for h in [h1, h2, h3, h4]:
            h.remove()

        # Compute fact rank at each checkpoint via Logit Lens
        ranks = {}
        for name, hs in checkpoints.items():
            normed = model.transformer.ln_f(hs.unsqueeze(0))
            logits = model.lm_head(normed).squeeze(0)
            ranks[name] = get_fact_rank(logits, fact_ids[0])

        # Also compute what L11 attention vs MLP contribute
        # after_L10 + attn_contribution = after_attn (before MLP)
        # after_attn + mlp_contribution = after_L11
        attn_contribution = checkpoints['l11_attn_out']
        mlp_contribution = checkpoints['l11_mlp_out']

        # Test: what if we skip L11 attention but keep MLP?
        skip_attn = checkpoints['after_L10'] + mlp_contribution
        normed = model.transformer.ln_f(skip_attn.unsqueeze(0))
        logits = model.lm_head(normed).squeeze(0)
        ranks['skip_l11_attn'] = get_fact_rank(logits, fact_ids[0])

        # Test: what if we skip L11 MLP but keep attention?
        skip_mlp = checkpoints['after_L10'] + attn_contribution
        normed = model.transformer.ln_f(skip_mlp.unsqueeze(0))
        logits = model.lm_head(normed).squeeze(0)
        ranks['skip_l11_mlp'] = get_fact_rank(logits, fact_ids[0])

        result = {
            'expected': expected, 'ranks': ranks,
        }
        all_results.append(result)

        print(f"  {expected:>12s}: L10_out=r{ranks['after_L10']:>5d} "
              f"L11_out=r{ranks['after_L11']:>5d} "
              f"skip_attn=r{ranks['skip_l11_attn']:>5d} "
              f"skip_mlp=r{ranks['skip_l11_mlp']:>5d}")

    # Analysis: is it attention or MLP that kills facts?
    avg_ranks = {k: np.mean([r['ranks'][k] for r in all_results])
                 for k in all_results[0]['ranks']}

    print("\n  Average ranks:")
    for k, v in sorted(avg_ranks.items(), key=lambda x: x[1]):
        print(f"    {k:>20s}: {v:.1f}")

    # The culprit: if skip_l11_attn has better rank than after_L11,
    # then attention is the suppressor
    attn_is_culprit = avg_ranks.get('skip_l11_attn', 99999) < avg_ranks.get('after_L11', 0)
    mlp_is_culprit = avg_ranks.get('skip_l11_mlp', 99999) < avg_ranks.get('after_L11', 0)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Average ranks at each checkpoint
    checkpoint_names = ['after_L10', 'after_L11', 'skip_l11_attn', 'skip_l11_mlp']
    display_names = ['After L10\n(before L11)', 'After L11\n(full)', 'Skip L11\nAttention', 'Skip L11\nMLP']
    avg_vals = [avg_ranks[k] for k in checkpoint_names]
    colors = ['green', 'red', 'blue', 'orange']
    axes[0].bar(range(len(checkpoint_names)), avg_vals, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_xticks(range(len(checkpoint_names)))
    axes[0].set_xticklabels(display_names, fontsize=8)
    axes[0].set_ylabel('Average Fact Rank (lower=better)')
    axes[0].set_title('L11 Sublayer Analysis')
    axes[0].set_yscale('log')

    # 2. Per-prompt comparison
    labels = [r['expected'][:6] for r in all_results]
    l10_vals = [r['ranks']['after_L10'] for r in all_results]
    l11_vals = [r['ranks']['after_L11'] for r in all_results]
    x = range(len(labels))
    axes[1].bar([i-0.2 for i in x], l10_vals, 0.4, label='After L10', color='green', alpha=0.7)
    axes[1].bar([i+0.2 for i in x], l11_vals, 0.4, label='After L11', color='red', alpha=0.7)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[1].set_ylabel('Fact Rank')
    axes[1].set_title('L10 vs L11: Per-Prompt')
    axes[1].legend()
    axes[1].set_yscale('log')

    plt.suptitle('Phase 68: The L11 Barrier', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase68_l11_barrier.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 68, 'name': 'The L11 Barrier',
        'avg_ranks': {k: round(v, 1) for k, v in avg_ranks.items()},
        'attn_is_culprit': attn_is_culprit,
        'mlp_is_culprit': mlp_is_culprit,
        'results': all_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase68_l11_barrier.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 68 RESULTS: The L11 Barrier")
    print(f"  Attention is culprit: {attn_is_culprit}")
    print(f"  MLP is culprit:      {mlp_is_culprit}")
    for k, v in sorted(avg_ranks.items(), key=lambda x: x[1]):
        print(f"    {k}: avg_rank={v:.1f}")
    print("=" * 70)
    phase_complete(68)

if __name__ == '__main__':
    main()
