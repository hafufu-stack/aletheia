# -*- coding: utf-8 -*-
"""
Phase 19: Truth Scaling Law
- Since gpt2-medium/large not cached, use virtual scaling:
  simulate larger models by varying effective hidden dimension
- Measure how critical spike scales with model capacity
- Derive scaling equation: spike_c ~ f(d_model)
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_model():
    print("[P19] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def main():
    print("=" * 70)
    print("  Phase 19: Truth Scaling Law")
    print("  How does critical spike scale with model capacity?")
    print("=" * 70)

    model, tok = load_model()
    d_model = model.config.n_embd  # 768
    n_layers = model.config.n_layer  # 12
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  GPT-2: d={d_model}, L={n_layers}, params={n_params:,}")

    qa_pairs = [
        ("The capital of Japan is", [11790]),
        ("The capital of France is", [6342]),
        ("Water freezes at", [657]),
        ("The largest planet is", [22721]),
        ("DNA stands for", [390]),
    ]

    magnitudes = list(range(0, 21))

    # === Baseline: full model ===
    print("\n[P19a] Full model critical spike measurement...")
    full_model_acc = {}
    for mag in magnitudes:
        correct = 0
        for prompt, fact_ids in qa_pairs:
            inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[:, -1, :].squeeze(0)
            for tid in fact_ids:
                logits[tid] += mag
            if torch.argmax(logits).item() in fact_ids:
                correct += 1
        full_model_acc[mag] = correct / len(qa_pairs)

    full_crit = next((m for m in magnitudes if full_model_acc[m] >= 1.0), None)
    print(f"  Full model critical spike: {full_crit}")

    # === Virtual scaling: ablate layers to simulate smaller models ===
    print("\n[P19b] Virtual scaling: ablating layers to simulate smaller models...")
    # By disabling upper layers, we simulate models with fewer layers
    class LayerKillHook:
        def __call__(self, module, input, output):
            # Replace output with input (skip this layer)
            return (input[0],) + output[1:] if isinstance(output, tuple) else input[0]

    scale_results = {}

    for n_active in [2, 4, 6, 8, 10, 12]:
        # Kill layers beyond n_active
        handles = []
        for li in range(n_active, n_layers):
            h = model.transformer.h[li].register_forward_hook(LayerKillHook())
            handles.append(h)

        # Find critical spike
        crit_mag = None
        for mag in magnitudes:
            correct = 0
            for prompt, fact_ids in qa_pairs:
                inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
                with torch.no_grad():
                    out = model(**inp)
                logits = out.logits[:, -1, :].squeeze(0)
                for tid in fact_ids:
                    logits[tid] += mag
                if torch.argmax(logits).item() in fact_ids:
                    correct += 1
            if correct == len(qa_pairs):
                crit_mag = mag
                break

        for h in handles:
            h.remove()

        effective_params = n_active * (n_params // n_layers)
        scale_results[n_active] = {
            'critical_spike': crit_mag,
            'effective_params': effective_params,
            'effective_d': d_model,  # same d, fewer layers
        }
        print(f"  {n_active:>2d} layers ({effective_params:>12,} params): "
              f"critical spike = {crit_mag}")

    # === Analyze scaling law ===
    print("\n[P19c] Deriving scaling law...")
    valid_pts = [(r['effective_params'], r['critical_spike'])
                 for r in scale_results.values()
                 if r['critical_spike'] is not None and r['critical_spike'] > 0]

    if len(valid_pts) >= 3:
        params = np.array([p for p, _ in valid_pts])
        spikes = np.array([s for _, s in valid_pts])

        # Fit: spike_c = a * params^beta
        log_p = np.log(params)
        log_s = np.log(spikes + 0.1)
        beta, log_a = np.polyfit(log_p, log_s, 1)
        a = np.exp(log_a)

        print(f"  Scaling law: spike_c = {a:.4f} * N^{beta:.4f}")
        print(f"  (negative beta = larger models need smaller spikes)")

        # Predict for hypothetical model sizes
        for target_params in [345e6, 774e6, 1.5e9, 175e9]:
            predicted = a * target_params ** beta
            print(f"    {target_params/1e9:.1f}B params -> predicted spike = {predicted:.2f}")
    else:
        beta = None
        a = None
        print("  Not enough valid data points for scaling law")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Accuracy curves for different model sizes
    for n_active in [2, 6, 12]:
        if n_active == 12:
            accs = [full_model_acc[m]*100 for m in magnitudes]
            axes[0].plot(magnitudes, accs, '.-', label=f'{n_active}L (full)', linewidth=2)

    axes[0].set_xlabel('Spike Magnitude')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy vs Model Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Critical spike vs model size (scaling law)
    if valid_pts:
        p_vals = [p/1e6 for p, _ in valid_pts]
        s_vals = [s for _, s in valid_pts]
        axes[1].loglog(p_vals, s_vals, 'ro', markersize=10, label='Measured')
        if beta is not None:
            p_fit = np.linspace(min(params), max(params)*10, 50)
            s_fit = a * p_fit ** beta
            axes[1].loglog(p_fit/1e6, s_fit, 'b--', alpha=0.5,
                           label=f'Fit: spike ~ N^{beta:.3f}')
        axes[1].set_xlabel('Parameters (M)')
        axes[1].set_ylabel('Critical Spike')
        axes[1].set_title('Truth Scaling Law')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Plot 3: Layer count vs critical spike
    layers = sorted(scale_results.keys())
    crits = [scale_results[l]['critical_spike'] or 0 for l in layers]
    axes[2].bar(layers, crits, color='teal', alpha=0.7)
    axes[2].set_xlabel('Active Layers')
    axes[2].set_ylabel('Critical Spike')
    axes[2].set_title('Depth vs Critical Spike')

    plt.suptitle('Phase 19: Truth Scaling Law', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase19_scaling.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 19, 'name': 'Truth Scaling Law',
        'gpt2_params': n_params,
        'gpt2_critical_spike': full_crit,
        'scale_results': {str(k): v for k, v in scale_results.items()},
        'scaling_beta': float(beta) if beta else None,
        'scaling_a': float(a) if a else None,
    }
    with open(os.path.join(RESULTS_DIR, 'phase19_scaling.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 19 RESULTS: Truth Scaling Law")
    print("=" * 70)
    if beta:
        print(f"  spike_c = {a:.4f} * N^{beta:.4f}")
    print(f"  Full model critical: {full_crit}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
