# -*- coding: utf-8 -*-
"""
Phase 62: Cross-Layer Token Agreement Map (Original idea)
For all 12 layers, what is the top-1 predicted token?
When do layers agree (facts) vs disagree (hallucination)?
Quantify the "consensus" vs "suppression" dynamics.
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
    print("[P62] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def main():
    print("=" * 70)
    print("  Phase 62: Cross-Layer Token Agreement Map")
    print("  When do layers agree vs disagree?")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The Earth orbits the", [4252], "Sun"),
        ("The boiling point of water is", [1802], "100"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("Oxygen has the atomic number", [807], "8"),
        # Non-fact prompts (should have different patterns)
        ("I think that the best way to", [], "non-fact"),
        ("Once upon a time, there was a", [], "non-fact"),
        ("In my opinion, the most important", [], "non-fact"),
    ]

    n_layers = 12
    all_results = []
    agreement_matrices = []

    for prompt, fact_ids, expected in tests:
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

        # Get top-1 token from each layer via Logit Lens
        layer_tokens = {}
        layer_top5 = {}
        for li in range(n_layers):
            normed = model.transformer.ln_f(layer_hs[li].unsqueeze(0))
            logits = model.lm_head(normed).squeeze(0)
            top5 = logits.argsort(descending=True)[:5].tolist()
            layer_tokens[li] = top5[0]
            layer_top5[li] = top5

        final_token = torch.argmax(out.logits[0, -1, :]).item()

        # Agreement matrix: for each pair of layers, do they agree on top-1?
        agreement = np.zeros((n_layers, n_layers))
        for i in range(n_layers):
            for j in range(n_layers):
                # Check overlap in top-5
                overlap = len(set(layer_top5[i]) & set(layer_top5[j]))
                agreement[i, j] = overlap / 5.0

        agreement_matrices.append(agreement)

        # Count how many layers agree with the final output
        agree_with_final = sum(1 for li in range(n_layers) if layer_tokens[li] == final_token)
        # Count how many layers have the fact token as top-1
        fact_layers = sum(1 for li in range(n_layers) if layer_tokens[li] in fact_ids) if fact_ids else 0

        # Decode tokens for display
        layer_tok_strs = []
        for li in range(n_layers):
            t = tok.decode([layer_tokens[li]]).encode('ascii', 'replace').decode().strip()
            layer_tok_strs.append(t)

        result = {
            'prompt': prompt, 'expected': expected,
            'layer_tokens': {str(li): layer_tok_strs[li] for li in range(n_layers)},
            'final_token': tok.decode([final_token]).encode('ascii', 'replace').decode().strip(),
            'agree_with_final': agree_with_final,
            'fact_layers': fact_layers,
            'is_fact': bool(fact_ids),
        }
        all_results.append(result)

        tok_str = ' | '.join([f"L{li}:{layer_tok_strs[li][:5]}" for li in [0, 4, 8, 10, 11]])
        print(f"  {expected:>12s}: {tok_str} final={result['final_token'][:5]} "
              f"agree={agree_with_final} fact_l={fact_layers}")

    # Analysis: fact vs non-fact agreement patterns
    fact_agree = [r['agree_with_final'] for r in all_results if r['is_fact']]
    nonfact_agree = [r['agree_with_final'] for r in all_results if not r['is_fact']]
    print(f"\n  Fact prompts: mean agree={np.mean(fact_agree):.1f}/12")
    print(f"  Non-fact prompts: mean agree={np.mean(nonfact_agree):.1f}/12")

    # Average agreement matrix for facts vs non-facts
    fact_matrices = [m for m, r in zip(agreement_matrices, all_results) if r['is_fact']]
    nonfact_matrices = [m for m, r in zip(agreement_matrices, all_results) if not r['is_fact']]
    avg_fact_matrix = np.mean(fact_matrices, axis=0)
    avg_nonfact_matrix = np.mean(nonfact_matrices, axis=0) if nonfact_matrices else np.zeros((12,12))

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Fact agreement matrix
    im1 = axes[0].imshow(avg_fact_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Layer')
    axes[0].set_title('Fact Prompts: Top-5 Overlap')
    axes[0].set_xticks(range(0, 12, 2))
    axes[0].set_yticks(range(0, 12, 2))
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # 2. Non-fact agreement matrix
    im2 = axes[1].imshow(avg_nonfact_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Layer')
    axes[1].set_title('Non-Fact Prompts: Top-5 Overlap')
    axes[1].set_xticks(range(0, 12, 2))
    axes[1].set_yticks(range(0, 12, 2))
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # 3. Per-prompt fact layer count
    fact_results = [r for r in all_results if r['is_fact']]
    labels = [r['expected'][:8] for r in fact_results]
    fact_l = [r['fact_layers'] for r in fact_results]
    agree_l = [r['agree_with_final'] for r in fact_results]
    x = range(len(labels))
    axes[2].bar([i-0.2 for i in x], fact_l, 0.4, label='Fact Layers', color='green', alpha=0.7)
    axes[2].bar([i+0.2 for i in x], agree_l, 0.4, label='Agree w/ Final', color='red', alpha=0.7)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[2].set_ylabel('Count (out of 12)')
    axes[2].set_title('Per-Prompt Layer Analysis')
    axes[2].legend(fontsize=8)

    plt.suptitle('Phase 62: Cross-Layer Token Agreement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase62_agreement_map.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 62, 'name': 'Cross-Layer Token Agreement Map',
        'results': all_results,
        'fact_mean_agree': float(np.mean(fact_agree)),
        'nonfact_mean_agree': float(np.mean(nonfact_agree)) if nonfact_agree else None,
    }
    with open(os.path.join(RESULTS_DIR, 'phase62_agreement_map.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 62 RESULTS")
    print(f"  Fact prompts: {np.mean(fact_agree):.1f}/12 layers agree with final")
    print(f"  Non-fact prompts: {np.mean(nonfact_agree):.1f}/12 layers agree with final")
    print("=" * 70)
    phase_complete(62)

if __name__ == '__main__':
    main()
