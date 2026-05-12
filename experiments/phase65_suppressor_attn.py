# -*- coding: utf-8 -*-
"""
Phase 65: Suppressor Head Attention Analysis
P61 found L11H7 is the top fact suppressor.
WHAT is L11H7 attending to? Is it attending to grammar cue tokens?
Visualize attention patterns of suppressors vs helpers.
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
    print("[P65] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def main():
    print("=" * 70)
    print("  Phase 65: Suppressor Head Attention Analysis")
    print("  What are the Grammar Police heads looking at?")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", "Tokyo"),
        ("The capital of France is", "Paris"),
        ("The largest planet is", "Jupiter"),
        ("Albert Einstein developed the theory of", "relativity"),
        ("The Earth orbits the", "Sun"),
        ("Shakespeare wrote", "Hamlet"),
        ("Water freezes at", "0"),
        ("The chemical symbol for gold is", "Au"),
        ("The boiling point of water is", "100"),
        ("Oxygen has the atomic number", "8"),
    ]

    # From P61: top suppressors and helpers
    suppressors = [(11, 7), (10, 7), (10, 6), (11, 1), (11, 4)]
    helpers = [(10, 5), (11, 11), (10, 11), (11, 0), (11, 5)]

    all_attn_data = []
    suppressor_entropy = {k: [] for k in ['suppressors', 'helpers', 'others']}
    suppressor_position_focus = {k: [] for k in ['suppressors', 'helpers']}

    for prompt, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        input_tokens = [tok.decode([t]).encode('ascii','replace').decode() for t in inp[0]]

        with torch.no_grad():
            out = model(inp, output_attentions=True, return_dict=True)

        # Analyze attention of each head type
        for layer_idx in [10, 11]:
            attn = out.attentions[layer_idx]  # (1, n_heads, seq, seq)
            for head_idx in range(12):
                # Attention from last token to all positions
                a = attn[0, head_idx, -1, :].cpu().numpy()
                entropy = float(-np.sum(a * np.log(a + 1e-12)))

                # Position of maximum attention
                max_pos = int(np.argmax(a))
                max_val = float(a[max_pos])
                # Relative position (0=first, 1=last)
                rel_pos = max_pos / max(1, len(a) - 1) if len(a) > 1 else 0

                # Categorize
                pair = (layer_idx, head_idx)
                if pair in suppressors:
                    suppressor_entropy['suppressors'].append(entropy)
                    suppressor_position_focus['suppressors'].append(rel_pos)
                elif pair in helpers:
                    suppressor_entropy['helpers'].append(entropy)
                    suppressor_position_focus['helpers'].append(rel_pos)
                else:
                    suppressor_entropy['others'].append(entropy)

        # Detailed attention for L11H7 (top suppressor)
        l11h7_attn = out.attentions[11][0, 7, -1, :].cpu().numpy()
        top3_pos = np.argsort(l11h7_attn)[-3:][::-1]
        top3_tokens = [(int(p), input_tokens[p] if p < len(input_tokens) else '?',
                       float(l11h7_attn[p])) for p in top3_pos]

        all_attn_data.append({
            'prompt': prompt, 'expected': expected,
            'l11h7_top3': top3_tokens,
            'input_tokens': input_tokens,
        })
        print(f"  {expected:>12s}: L11H7 attends to: "
              + ", ".join([f"'{t[1]}' ({t[2]:.2f})" for t in top3_tokens]))

    # === Analysis ===
    print("\n[P65] Entropy analysis:")
    for cat in ['suppressors', 'helpers', 'others']:
        vals = suppressor_entropy[cat]
        print(f"  {cat:>12s}: mean_H={np.mean(vals):.3f} std={np.std(vals):.3f}")

    print("\n[P65] Position focus (where does attention peak?):")
    for cat in ['suppressors', 'helpers']:
        vals = suppressor_position_focus[cat]
        print(f"  {cat:>12s}: mean_rel_pos={np.mean(vals):.3f} "
              f"(0=start, 1=end)")

    # What tokens do suppressors focus on?
    print("\n[P65] L11H7 most-attended tokens across all prompts:")
    token_freq = {}
    for d in all_attn_data:
        for pos, tok_str, weight in d['l11h7_top3']:
            tok_str = tok_str.strip()
            if tok_str not in token_freq:
                token_freq[tok_str] = 0
            token_freq[tok_str] += weight
    sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    for t, w in sorted_tokens:
        print(f"  '{t}': total_weight={w:.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Entropy distribution
    for cat, color in [('suppressors', 'red'), ('helpers', 'green'), ('others', 'gray')]:
        vals = suppressor_entropy[cat]
        axes[0].hist(vals, bins=15, alpha=0.5, color=color, label=f'{cat} (n={len(vals)})',
                    density=True)
    axes[0].set_xlabel('Attention Entropy')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Entropy: Suppressors vs Helpers')
    axes[0].legend(fontsize=8)

    # 2. Position focus
    for cat, color in [('suppressors', 'red'), ('helpers', 'green')]:
        vals = suppressor_position_focus[cat]
        axes[1].hist(vals, bins=10, alpha=0.5, color=color, label=cat, density=True)
    axes[1].set_xlabel('Relative Position (0=start, 1=end)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Where Do Heads Focus?')
    axes[1].legend(fontsize=8)

    # 3. Top attended tokens by L11H7
    top_toks = [t for t, w in sorted_tokens[:8]]
    top_weights = [w for t, w in sorted_tokens[:8]]
    axes[2].barh(range(len(top_toks)), top_weights, color='red', alpha=0.7)
    axes[2].set_yticks(range(len(top_toks)))
    axes[2].set_yticklabels([f"'{t}'" for t in top_toks], fontsize=8)
    axes[2].set_xlabel('Total Attention Weight')
    axes[2].set_title("L11H7's Favorite Tokens")
    axes[2].invert_yaxis()

    plt.suptitle('Phase 65: Suppressor Head Attention', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase65_suppressor_attn.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 65, 'name': 'Suppressor Head Attention Analysis',
        'entropy': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                   for k, v in suppressor_entropy.items()},
        'position_focus': {k: {'mean': float(np.mean(v))}
                          for k, v in suppressor_position_focus.items()},
        'l11h7_top_tokens': sorted_tokens,
        'attention_data': all_attn_data,
    }
    with open(os.path.join(RESULTS_DIR, 'phase65_suppressor_attn.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 65 RESULTS")
    print("  Suppressor entropy: " + f"{np.mean(suppressor_entropy['suppressors']):.3f}")
    print("  Helper entropy:     " + f"{np.mean(suppressor_entropy['helpers']):.3f}")
    print("  L11H7 top token:    " + f"'{sorted_tokens[0][0]}'")
    print("=" * 70)
    phase_complete(65)

if __name__ == '__main__':
    main()
