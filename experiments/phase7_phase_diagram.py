# -*- coding: utf-8 -*-
"""
Phase 7: Hallucination Phase Diagram
- 2D grid: temperature x spike magnitude
- Map accuracy and entropy across full parameter space
- Find critical exponents of the hallucination phase transition
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
    print("[P7] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def main():
    print("=" * 70)
    print("  Phase 7: Hallucination Phase Diagram")
    print("  Temperature x Spike Magnitude -> Accuracy/Entropy Map")
    print("=" * 70)

    model, tok = load_model()

    qa_pairs = [
        ("The capital of Japan is", [11790]),
        ("Water freezes at", [657]),
        ("The chemical formula for water is", [367]),
        ("The largest planet is", [22721]),
        ("DNA stands for", [390]),
    ]

    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    spike_mags = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

    acc_grid = np.zeros((len(temperatures), len(spike_mags)))
    ent_grid = np.zeros((len(temperatures), len(spike_mags)))

    print(f"\n[P7a] Scanning {len(temperatures)}x{len(spike_mags)} = "
          f"{len(temperatures)*len(spike_mags)} grid points...")

    for ti, temp in enumerate(temperatures):
        for si, mag in enumerate(spike_mags):
            correct = 0
            entropies = []

            for prompt, fact_ids in qa_pairs:
                inp = tok(prompt, return_tensors='pt', truncation=True,
                          max_length=128).to(DEVICE)
                with torch.no_grad():
                    out = model(**inp)
                logits = out.logits[:, -1, :]

                # Spike inject
                spiked = logits.clone()
                for tid in fact_ids:
                    if tid < spiked.shape[-1]:
                        spiked[..., tid] += mag

                # Apply temperature
                probs = F.softmax(spiked / max(temp, 0.01), dim=-1).squeeze(0)
                h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
                entropies.append(h)

                if torch.argmax(probs).item() in fact_ids:
                    correct += 1

            acc_grid[ti, si] = correct / len(qa_pairs)
            ent_grid[ti, si] = np.mean(entropies)

        print(f"  T={temp:.1f}: acc=[{', '.join(f'{a:.0%}' for a in acc_grid[ti])}]")

    # === Find critical line (phase boundary) ===
    print("\n[P7b] Finding phase boundary...")
    critical_line = []
    for ti, temp in enumerate(temperatures):
        for si, mag in enumerate(spike_mags):
            if acc_grid[ti, si] >= 1.0:
                critical_line.append((temp, mag))
                break
        else:
            critical_line.append((temp, None))

    for temp, mag in critical_line:
        print(f"  T={temp:.1f} -> critical spike = {mag}")

    # === Critical exponent estimation ===
    valid_points = [(t, m) for t, m in critical_line if m is not None and m > 0]
    if len(valid_points) >= 3:
        temps_c = np.array([p[0] for p in valid_points])
        mags_c = np.array([p[1] for p in valid_points])
        # Fit: mag_c ~ T^gamma (power law)
        log_t = np.log(temps_c + 0.01)
        log_m = np.log(mags_c + 0.01)
        gamma, intercept = np.polyfit(log_t, log_m, 1)
        print(f"\n  Critical exponent gamma = {gamma:.3f}")
        print(f"  Phase boundary: spike_c ~ T^{gamma:.2f}")
    else:
        gamma = None

    # === Visualization ===
    print("\n[P7] Generating phase diagram...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Accuracy phase diagram (heatmap)
    im1 = axes[0].imshow(acc_grid, aspect='auto', origin='lower',
                          extent=[spike_mags[0], spike_mags[-1],
                                  temperatures[0], temperatures[-1]],
                          cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_xlabel('Spike Magnitude')
    axes[0].set_ylabel('Temperature')
    axes[0].set_title('Accuracy Phase Diagram')
    plt.colorbar(im1, ax=axes[0], label='Accuracy')
    # Draw phase boundary
    valid_pts = [(m, t) for t, m in critical_line if m is not None]
    if valid_pts:
        axes[0].plot([p[0] for p in valid_pts], [p[1] for p in valid_pts],
                     'k--', linewidth=2, label='Phase Boundary')
        axes[0].legend()

    # Plot 2: Entropy phase diagram
    im2 = axes[1].imshow(ent_grid, aspect='auto', origin='lower',
                          extent=[spike_mags[0], spike_mags[-1],
                                  temperatures[0], temperatures[-1]],
                          cmap='hot_r')
    axes[1].set_xlabel('Spike Magnitude')
    axes[1].set_ylabel('Temperature')
    axes[1].set_title('Entropy Phase Diagram')
    plt.colorbar(im2, ax=axes[1], label='Entropy')

    # Plot 3: Critical line
    if valid_points:
        axes[2].loglog([p[0] for p in valid_points],
                       [p[1] for p in valid_points],
                       'ro-', linewidth=2, markersize=8)
        if gamma is not None:
            t_fit = np.linspace(0.1, 3.0, 50)
            m_fit = np.exp(intercept) * t_fit ** gamma
            axes[2].loglog(t_fit, m_fit, 'b--', alpha=0.5,
                           label=f'Fit: spike ~ T^{gamma:.2f}')
            axes[2].legend()
        axes[2].set_xlabel('Temperature')
        axes[2].set_ylabel('Critical Spike Magnitude')
        axes[2].set_title('Phase Boundary (log-log)')
        axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 7: Hallucination Phase Diagram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase7_phase_diagram.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 7, 'name': 'Hallucination Phase Diagram',
        'temperatures': temperatures,
        'spike_magnitudes': spike_mags,
        'accuracy_grid': acc_grid.tolist(),
        'entropy_grid': ent_grid.tolist(),
        'critical_line': [(t, m) for t, m in critical_line],
        'critical_exponent_gamma': gamma,
    }
    with open(os.path.join(RESULTS_DIR, 'phase7_phase_diagram.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 7 RESULTS: Hallucination Phase Diagram")
    print("=" * 70)
    if gamma:
        print(f"  Critical exponent: gamma = {gamma:.3f}")
    print(f"  Grid: {len(temperatures)} temps x {len(spike_mags)} spikes")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
