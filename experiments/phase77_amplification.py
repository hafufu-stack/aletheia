# -*- coding: utf-8 -*-
"""
Phase 77: Suppressor Amplification (Opus Original)
What if we BOOST L11H7 instead of ablating it?
If GSF is proportional to H7 output magnitude, boosting should
make facts disappear even further. Test the dose-response curve.
This tests linearity vs nonlinearity of the suppression mechanism.
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
    print("[P77] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def make_scaling_hook(head_idx, scale, n_heads, hidden_dim):
    head_dim = hidden_dim // n_heads
    def hook_fn(module, args, output):
        hs = output[0].clone()
        start = head_idx * head_dim
        end = start + head_dim
        hs[:, :, start:end] = hs[:, :, start:end] * scale
        return (hs,) + output[1:]
    return hook_fn

def main():
    print("=" * 70)
    print("  Phase 77: Suppressor Amplification")
    print("  Dose-response: what happens when we BOOST L11H7?")
    print("=" * 70)

    model, tok = load_model()
    n_heads = model.config.n_head
    hidden_dim = model.config.n_embd

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("The Earth orbits the", [4252], "Sun"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The boiling point of water is", [1802], "100"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("Oxygen has the atomic number", [807], "8"),
        ("The chemical symbol for gold is", [7591], "Au"),
    ]

    # Scale factors: 0 (ablation) to 3 (3x boost)
    scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    scale_results = {}
    for scale in scales:
        correct = 0
        ranks = []
        for prompt, fact_ids, expected in tests:
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

            handle = model.transformer.h[11].register_forward_hook(
                make_scaling_hook(7, scale, n_heads, hidden_dim))
            with torch.no_grad():
                out = model(inp)
            handle.remove()

            rank = get_fact_rank(out.logits[0, -1, :], fact_ids[0])
            ranks.append(rank)
            if torch.argmax(out.logits[0, -1, :]).item() in fact_ids:
                correct += 1

        scale_results[scale] = {
            'accuracy': correct / len(tests),
            'median_rank': float(np.median(ranks)),
            'mean_rank': float(np.mean(ranks)),
            'ranks': ranks,
        }
        print(f"  scale={scale:.2f}: acc={correct/len(tests):.0%} "
              f"med_rank={np.median(ranks):.0f} mean_rank={np.mean(ranks):.1f}")

    # Also test: boost L10H5 (top helper) -- does it help?
    print("\n[P77b] Boosting L10H5 (top fact helper)...")
    helper_results = {}
    for scale in [0.5, 1.0, 1.5, 2.0, 3.0]:
        correct = 0
        ranks = []
        for prompt, fact_ids, expected in tests:
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
            handle = model.transformer.h[10].register_forward_hook(
                make_scaling_hook(5, scale, n_heads, hidden_dim))
            with torch.no_grad():
                out = model(inp)
            handle.remove()
            rank = get_fact_rank(out.logits[0, -1, :], fact_ids[0])
            ranks.append(rank)
            if torch.argmax(out.logits[0, -1, :]).item() in fact_ids:
                correct += 1
        helper_results[scale] = {
            'accuracy': correct / len(tests),
            'median_rank': float(np.median(ranks)),
        }
        print(f"  L10H5 scale={scale:.1f}: acc={correct/len(tests):.0%} "
              f"med_rank={np.median(ranks):.0f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Dose-response curve: scale vs accuracy
    sc = sorted(scale_results.keys())
    accs = [scale_results[s]['accuracy']*100 for s in sc]
    axes[0].plot(sc, accs, 'b.-', linewidth=2, markersize=8)
    axes[0].set_xlabel('L11H7 Scale Factor')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Dose-Response: H7 Scale vs Accuracy')
    axes[0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Normal (1.0)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Scale vs median rank
    meds = [scale_results[s]['median_rank'] for s in sc]
    axes[1].plot(sc, meds, 'r.-', linewidth=2, markersize=8)
    axes[1].set_xlabel('L11H7 Scale Factor')
    axes[1].set_ylabel('Median Fact Rank')
    axes[1].set_title('Dose-Response: H7 Scale vs Rank')
    axes[1].axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    # 3. Suppressor boost vs Helper boost
    h_scales = sorted(helper_results.keys())
    h_accs = [helper_results[s]['accuracy']*100 for s in h_scales]
    s_accs_subset = [scale_results[s]['accuracy']*100 for s in h_scales if s in scale_results]
    axes[2].plot(h_scales, h_accs, 'g.-', linewidth=2, markersize=8, label='Boost L10H5 (Helper)')
    if len(s_accs_subset) == len(h_scales):
        axes[2].plot(h_scales, s_accs_subset, 'r.-', linewidth=2, markersize=8, label='Boost L11H7 (Suppress)')
    axes[2].set_xlabel('Scale Factor')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Boost Helper vs Boost Suppressor')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 77: Suppressor Amplification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase77_amplification.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Is dose-response linear?
    baseline_med = scale_results[1.0]['median_rank']
    ablated_med = scale_results[0.0]['median_rank']
    boosted_med = scale_results[2.0]['median_rank']
    # Linear prediction for 2.0: baseline + (baseline - ablated) = 2*baseline - ablated
    linear_prediction = 2 * baseline_med - ablated_med
    actual = boosted_med
    linearity = 1.0 - abs(actual - linear_prediction) / max(abs(linear_prediction), 1)

    output = {
        'phase': 77, 'name': 'Suppressor Amplification',
        'suppressor_dose_response': {str(k): v for k, v in scale_results.items()},
        'helper_dose_response': {str(k): v for k, v in helper_results.items()},
        'linearity_score': linearity,
    }
    with open(os.path.join(RESULTS_DIR, 'phase77_amplification.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 77: Suppressor Amplification")
    print(f"  Ablation (0x): med_rank={ablated_med}")
    print(f"  Normal  (1x):  med_rank={baseline_med}")
    print(f"  Boosted (2x):  med_rank={boosted_med}")
    print(f"  Linearity: {linearity:.2f}")
    print("=" * 70)
    phase_complete(77)

if __name__ == '__main__':
    main()
