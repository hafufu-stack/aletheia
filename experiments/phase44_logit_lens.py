# -*- coding: utf-8 -*-
"""
Phase 44: Logit Lens Fact Extraction
P43 proved facts are cleanest at L3-L6. Instead of rejecting output tokens,
project INTERMEDIATE hidden states through LM head to salvage facts
before Skill heads corrupt them at deeper layers.
"""
import os, json, sys
import numpy as np
import torch
import torch.nn.functional as F
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
    print("[P44] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def logit_lens(model, tok, prompt):
    """Project every layer's hidden state through LM head (Logit Lens)."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    layer_hs = {}
    handles = []
    for li in range(12):
        def make_hook(layer_idx):
            def hook_fn(module, args, output):
                layer_hs[layer_idx] = output[0][0, -1, :].detach()
            return hook_fn
        h = model.transformer.h[li].register_forward_hook(make_hook(li))
        handles.append(h)

    with torch.no_grad():
        out = model(**inp)
    for h in handles:
        h.remove()

    # Project each layer through ln_f + lm_head
    layer_logits = {}
    for li, hs in layer_hs.items():
        normed = model.transformer.ln_f(hs.unsqueeze(0))
        logits = model.lm_head(normed).squeeze(0)
        layer_logits[li] = logits

    final_logits = out.logits[:, -1, :].squeeze(0)
    return layer_logits, final_logits

def main():
    print("=" * 70)
    print("  Phase 44: Logit Lens Fact Extraction")
    print("  Salvage facts from intermediate layers before Skill corruption")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("DNA stands for", [390], "de"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The speed of light is approximately", [22626], "299"),
        ("The Earth orbits the", [4252], "Sun"),
        ("The boiling point of water is", [1802], "100"),
    ]

    # === P44a: Logit Lens per-layer analysis ===
    print("\n[P44a] Logit Lens: fact token rank per layer...")
    all_results = []

    for prompt, fact_ids, expected in tests:
        layer_logits, final_logits = logit_lens(model, tok, prompt)

        ranks = {}
        top1s = {}
        for li in range(12):
            logits = layer_logits[li]
            rank = int((logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            top1 = tok.decode([torch.argmax(logits).item()]).encode('ascii','replace').decode().strip()
            ranks[li] = rank
            top1s[li] = top1

        final_rank = int((final_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1

        # Find best layer
        best_layer = min(ranks, key=ranks.get)
        best_rank = ranks[best_layer]

        all_results.append({
            'prompt': prompt[:35], 'expected': expected,
            'final_rank': final_rank, 'best_layer': best_layer,
            'best_rank': best_rank, 'ranks': ranks, 'top1s': top1s,
        })

        # Pretty print
        rank_bar = ' '.join([f'{ranks[l]:>4d}' for l in range(12)])
        improved = 'IMPROVED' if best_rank < final_rank else 'same'
        print(f"  {expected:>8s}: final=r{final_rank:>5d} best=L{best_layer}:r{best_rank:>4d} [{improved}]")
        print(f"           L0   L1   L2   L3   L4   L5   L6   L7   L8   L9  L10  L11")
        print(f"           {rank_bar}")

    # === P44b: Intermediate layer ensemble ===
    print("\n[P44b] Layer ensemble: average logits from L4-L8...")
    ensemble_results = []
    for prompt, fact_ids, expected in tests:
        layer_logits, final_logits = logit_lens(model, tok, prompt)

        # Ensemble: average L4-L8
        ensemble_logits = torch.zeros_like(final_logits)
        for li in range(4, 9):
            ensemble_logits += layer_logits[li]
        ensemble_logits /= 5

        ens_rank = int((ensemble_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
        final_rank = int((final_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1

        # Weighted ensemble: prefer earlier layers
        weighted = torch.zeros_like(final_logits)
        weights = {4: 1.0, 5: 0.8, 6: 0.6, 7: 0.4, 8: 0.2}
        for li, w in weights.items():
            weighted += w * layer_logits[li]
        weighted /= sum(weights.values())
        w_rank = int((weighted.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1

        # Combined: final + best intermediate
        best_layer = min(range(12), key=lambda l: int((layer_logits[l].argsort(descending=True) == fact_ids[0]).nonzero().item()))
        combined = final_logits + layer_logits[best_layer]
        c_rank = int((combined.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1

        ensemble_results.append({
            'expected': expected, 'final_rank': final_rank,
            'ensemble_rank': ens_rank, 'weighted_rank': w_rank,
            'combined_rank': c_rank,
        })
        print(f"  {expected:>8s}: final=r{final_rank:>5d} ens=r{ens_rank:>5d} "
              f"wt=r{w_rank:>5d} comb=r{c_rank:>5d}")

    # === P44c: Oracle-guided layer selection ===
    print("\n[P44c] Oracle-guided: use entropy to select best layer...")
    # When entropy is high -> use earlier layer; when low -> use final
    oracle_results = []
    for i, (prompt, fact_ids, expected) in enumerate(tests):
        layer_logits, final_logits = logit_lens(model, tok, prompt)
        r = all_results[i]

        # Simple heuristic: use L6 always (P43 showed facts cleanest there)
        l6_rank = r['ranks'][6]
        l6_correct = torch.argmax(layer_logits[6]).item() in fact_ids
        final_correct = torch.argmax(final_logits).item() in fact_ids

        oracle_results.append({
            'expected': expected, 'l6_correct': l6_correct,
            'final_correct': final_correct, 'l6_rank': l6_rank,
        })

    l6_acc = sum(1 for r in oracle_results if r['l6_correct']) / len(tests)
    final_acc = sum(1 for r in oracle_results if r['final_correct']) / len(tests)
    print(f"  Final layer accuracy: {final_acc:.0%}")
    print(f"  L6 accuracy:          {l6_acc:.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Heatmap: fact rank per layer per prompt
    labels = [r['expected'] for r in all_results]
    rank_matrix = [[min(r['ranks'][l], 500) for l in range(12)] for r in all_results]
    im = axes[0].imshow(rank_matrix, aspect='auto', cmap='RdYlGn_r',
                        norm=matplotlib.colors.LogNorm(vmin=1, vmax=500))
    axes[0].set_xticks(range(12))
    axes[0].set_xticklabels([f'L{l}' for l in range(12)], fontsize=7)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_title('Fact Token Rank (Logit Lens)')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Best layer histogram
    best_layers = [r['best_layer'] for r in all_results]
    axes[1].hist(best_layers, bins=range(13), color='teal', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Count (best for fact)')
    axes[1].set_title('Which Layer Holds Facts Best?')
    axes[1].set_xticks(range(12))

    # Ensemble comparison
    methods = ['Final', 'Ensemble', 'Weighted', 'Combined']
    method_ranks = [
        [r['final_rank'] for r in ensemble_results],
        [r['ensemble_rank'] for r in ensemble_results],
        [r['weighted_rank'] for r in ensemble_results],
        [r['combined_rank'] for r in ensemble_results],
    ]
    medians = [np.median(r) for r in method_ranks]
    axes[2].bar(methods, medians, color=['red','blue','green','orange'], alpha=0.7)
    axes[2].set_ylabel('Median Fact Rank')
    axes[2].set_title('Ensemble Methods')

    plt.suptitle('Phase 44: Logit Lens Fact Extraction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase44_logit_lens.png'), dpi=150, bbox_inches='tight')
    plt.close()

    improved_count = sum(1 for r in all_results if r['best_rank'] < r['final_rank'])
    results = {
        'phase': 44, 'name': 'Logit Lens Fact Extraction',
        'improved': improved_count, 'total': len(tests),
        'l6_accuracy': l6_acc, 'final_accuracy': final_acc,
        'per_case': all_results,
        'ensemble_results': ensemble_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase44_logit_lens.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 44 RESULTS: Logit Lens Fact Extraction")
    print("=" * 70)
    print(f"  Improved by Logit Lens: {improved_count}/{len(tests)}")
    print(f"  Final layer accuracy:   {final_acc:.0%}")
    print(f"  L6 accuracy:            {l6_acc:.0%}")
    print("=" * 70)
    phase_complete(44)

if __name__ == '__main__':
    main()
