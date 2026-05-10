# -*- coding: utf-8 -*-
"""
Phase 42: Stochastic Truth Resonance
From SNN-Synthesis: noise injection (sigma=0.15) boosted Hanoi solve 9%->32%.
Apply stochastic resonance to Aletheia: inject calibrated noise into
hidden states to enhance factual retrieval via "noise-assisted tunneling."
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
    print("[P42] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def noisy_forward(model, tok, prompt, layer_idx, sigma, n_samples=10):
    """Forward pass with Gaussian noise injected at specified layer."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    results = []

    for _ in range(n_samples):
        handles = []
        def make_hook(s):
            def hook_fn(module, args, output):
                noise = torch.randn_like(output[0]) * s
                return (output[0] + noise,) + output[1:]
            return hook_fn
        h = model.transformer.h[layer_idx].register_forward_hook(make_hook(sigma))
        handles.append(h)

        with torch.no_grad():
            out = model(**inp)
        for hh in handles:
            hh.remove()

        logits = out.logits[:, -1, :].squeeze(0)
        results.append(logits.cpu())

    # Majority vote: most common top-1 across samples
    top1s = [torch.argmax(r).item() for r in results]
    from collections import Counter
    majority = Counter(top1s).most_common(1)[0][0]

    # Average logits
    avg_logits = torch.mean(torch.stack(results), dim=0)

    return majority, avg_logits, top1s

def main():
    print("=" * 70)
    print("  Phase 42: Stochastic Truth Resonance")
    print("  Noise injection for fact retrieval (from SNN-Synthesis SR)")
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
    ]

    # === P42a: Sigma sweep (classic SR curve) ===
    print("\n[P42a] Stochastic resonance sigma sweep (L10, n=20)...")
    sigma_sweep = {}
    for sigma in [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]:
        correct_majority = 0
        correct_avg = 0
        for prompt, fact_ids, expected in tests:
            if sigma == 0:
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    out = model(**inp)
                logits = out.logits[:, -1, :].squeeze(0)
                if torch.argmax(logits).item() in fact_ids:
                    correct_majority += 1
                    correct_avg += 1
            else:
                majority, avg_logits, _ = noisy_forward(
                    model, tok, prompt, layer_idx=10, sigma=sigma, n_samples=20)
                if majority in fact_ids:
                    correct_majority += 1
                if torch.argmax(avg_logits).item() in fact_ids:
                    correct_avg += 1

        sigma_sweep[sigma] = {
            'majority': correct_majority / len(tests),
            'avg': correct_avg / len(tests),
        }
        print(f"  sigma={sigma:.2f}: majority={correct_majority}/{len(tests)}, "
              f"avg={correct_avg}/{len(tests)}")

    # === P42b: Layer sweep (which layer benefits most?) ===
    print("\n[P42b] Layer sweep (sigma=0.15, n=20)...")
    layer_sweep = {}
    for layer in [4, 6, 8, 10, 11]:
        correct = 0
        for prompt, fact_ids, _ in tests:
            majority, _, _ = noisy_forward(model, tok, prompt, layer, 0.15, 20)
            if majority in fact_ids:
                correct += 1
        layer_sweep[layer] = correct / len(tests)
        print(f"  L{layer:>2d}: {correct}/{len(tests)} = {correct/len(tests):.0%}")

    # === P42c: SR + spike combo ===
    print("\n[P42c] Stochastic resonance + spike combo...")
    combo_results = {}
    for sigma in [0, 0.1, 0.15, 0.2]:
        for spike in [0, 3, 5]:
            correct = 0
            for prompt, fact_ids, _ in tests:
                if sigma == 0:
                    inp = tok(prompt, return_tensors='pt').to(DEVICE)
                    with torch.no_grad():
                        out = model(**inp)
                    logits = out.logits[:, -1, :].squeeze(0)
                else:
                    _, logits, _ = noisy_forward(model, tok, prompt, 10, sigma, 20)
                for fid in fact_ids:
                    logits[fid] += spike
                if torch.argmax(logits).item() in fact_ids:
                    correct += 1
            combo_results[f"s{sigma}_sp{spike}"] = correct / len(tests)
            print(f"  sigma={sigma:.2f} + spike={spike}: {correct}/{len(tests)}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Classic SR curve
    sigmas = sorted(sigma_sweep.keys())
    maj_accs = [sigma_sweep[s]['majority']*100 for s in sigmas]
    avg_accs = [sigma_sweep[s]['avg']*100 for s in sigmas]
    axes[0].plot(sigmas, maj_accs, 'g.-', linewidth=2, label='Majority vote')
    axes[0].plot(sigmas, avg_accs, 'b.--', linewidth=2, label='Avg logits')
    axes[0].set_xlabel('Noise sigma')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Stochastic Resonance Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Layer sweep
    layers = sorted(layer_sweep.keys())
    l_accs = [layer_sweep[l]*100 for l in layers]
    axes[1].bar([f'L{l}' for l in layers], l_accs, color='teal', alpha=0.7)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('SR by Layer (sigma=0.15)')

    # Combo heatmap
    ss = [0, 0.1, 0.15, 0.2]
    sps = [0, 3, 5]
    grid = [[combo_results.get(f"s{s}_sp{sp}", 0)*100 for sp in sps] for s in ss]
    im = axes[2].imshow(grid, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    axes[2].set_xticks(range(len(sps)))
    axes[2].set_xticklabels([f'sp={s}' for s in sps])
    axes[2].set_yticks(range(len(ss)))
    axes[2].set_yticklabels([f's={s}' for s in ss])
    axes[2].set_title('SR + Spike Combo')
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    plt.suptitle('Phase 42: Stochastic Truth Resonance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase42_sr.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Find optimal sigma (SR peak)
    best_sigma = max(sigma_sweep, key=lambda s: sigma_sweep[s]['majority'])
    results = {
        'phase': 42, 'name': 'Stochastic Truth Resonance',
        'inspiration': 'SNN-Synthesis SR (sigma=0.15, 9%->32%)',
        'sigma_sweep': {str(k): v for k, v in sigma_sweep.items()},
        'layer_sweep': {str(k): v for k, v in layer_sweep.items()},
        'combo': combo_results,
        'best_sigma': best_sigma,
    }
    with open(os.path.join(RESULTS_DIR, 'phase42_sr.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 42 RESULTS: Stochastic Truth Resonance")
    print("=" * 70)
    print(f"  Best sigma: {best_sigma} ({sigma_sweep[best_sigma]['majority']:.0%})")
    print(f"  Baseline (sigma=0): {sigma_sweep[0]['majority']:.0%}")
    print("=" * 70)
    phase_complete(42)

if __name__ == '__main__':
    main()
