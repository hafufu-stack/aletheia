# -*- coding: utf-8 -*-
"""
Phase 35: Grand Unified Truth Equation
Extend spike_c = alpha * N^beta to include vocab size V and
architecture topology C for cross-model prediction.
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import phase_complete

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("  Phase 35: Grand Unified Truth Equation")
    print("  Multi-variable scaling law: spike_c(N, V, d, L)")
    print("=" * 70)

    # Known data points
    models = {
        'GPT-2': {
            'N': 124e6, 'V': 50257, 'd': 768, 'L': 12, 'n_heads': 12,
            'spike_actual': 6, 'spike_p19': 9.10,
        },
        'GPT-2-Med': {
            'N': 345e6, 'V': 50257, 'd': 1024, 'L': 24, 'n_heads': 16,
            'spike_actual': None, 'spike_p19': 5.52,
        },
        'GPT-2-Large': {
            'N': 774e6, 'V': 50257, 'd': 1280, 'L': 36, 'n_heads': 20,
            'spike_actual': None, 'spike_p19': 3.72,
        },
        'GPT-2-XL': {
            'N': 1.5e9, 'V': 50257, 'd': 1600, 'L': 48, 'n_heads': 25,
            'spike_actual': None, 'spike_p19': 2.70,
        },
        'Qwen2.5-7B': {
            'N': 7e9, 'V': 152064, 'd': 3584, 'L': 28, 'n_heads': 28,
            'spike_actual': 10, 'spike_p19': 1.26,
        },
        'Mistral-7B': {
            'N': 7e9, 'V': 32000, 'd': 4096, 'L': 32, 'n_heads': 32,
            'spike_actual': 10, 'spike_p19': 1.26,
        },
    }

    # === P35a: Analyze why 7B models deviate ===
    print("\n[P35a] Analyzing scaling law deviation...")
    print(f"  Original law: spike_c = 85846 * N^(-0.491)")
    print(f"\n  Model comparison:")
    for name, m in models.items():
        ratio = m['V'] / m['N']
        d_ratio = m['d'] / np.sqrt(m['N'])
        print(f"  {name:>15s}: N={m['N']:.0e} V={m['V']:>6d} V/N={ratio:.2e} "
              f"d/sqrt(N)={d_ratio:.4f} actual={m['spike_actual']}")

    # === P35b: Proposed correction: vocab dilution factor ===
    print("\n[P35b] Vocab dilution hypothesis...")
    # Hypothesis: larger vocab dilutes the spike effect
    # spike_c = alpha * N^beta * (V/V_ref)^gamma
    V_ref = 50257  # GPT-2 vocab
    ALPHA = 85845.7508
    BETA = -0.491

    # The 7B models have spike_actual=10, N=7e9
    # Original prediction: 1.26
    # Actual: 10
    # Correction factor needed: 10 / 1.26 = 7.94x
    # Qwen vocab: 152064 -> ratio = 152064/50257 = 3.03
    # Mistral vocab: 32000 -> ratio = 32000/50257 = 0.64

    # Try different gamma values
    best_gamma = 0
    best_error = float('inf')
    gamma_sweep = {}
    for gamma in np.arange(0, 3, 0.1):
        total_error = 0
        n_points = 0
        for name, m in models.items():
            if m['spike_actual'] is not None:
                predicted = ALPHA * (m['N'] ** BETA) * ((m['V'] / V_ref) ** gamma)
                error = abs(predicted - m['spike_actual'])
                total_error += error ** 2
                n_points += 1
        rmse = np.sqrt(total_error / n_points) if n_points > 0 else float('inf')
        gamma_sweep[gamma] = rmse
        if rmse < best_error:
            best_error = rmse
            best_gamma = gamma

    print(f"  Best gamma (vocab correction): {best_gamma:.1f} (RMSE={best_error:.2f})")

    # === P35c: Full multi-variable fit ===
    print("\n[P35c] Multi-variable equation...")
    # spike_c = alpha * N^beta * (V/V_ref)^gamma * (L/L_ref)^delta
    L_ref = 12  # GPT-2 layers

    best_params = {'gamma': 0, 'delta': 0}
    best_rmse = float('inf')
    param_sweep = []

    for gamma in np.arange(0, 3, 0.2):
        for delta in np.arange(-1, 2, 0.2):
            total_error = 0
            n_pts = 0
            for name, m in models.items():
                if m['spike_actual'] is not None:
                    pred = ALPHA * (m['N'] ** BETA) * \
                           ((m['V'] / V_ref) ** gamma) * \
                           ((m['L'] / L_ref) ** delta)
                    total_error += (pred - m['spike_actual']) ** 2
                    n_pts += 1
            rmse = np.sqrt(total_error / n_pts) if n_pts > 0 else float('inf')
            param_sweep.append({'gamma': round(gamma, 1), 'delta': round(delta, 1), 'rmse': round(rmse, 3)})
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'gamma': gamma, 'delta': delta}

    g_opt = best_params['gamma']
    d_opt = best_params['delta']
    print(f"  Optimal: gamma={g_opt:.1f}, delta={d_opt:.1f} (RMSE={best_rmse:.2f})")
    print(f"\n  GRAND UNIFIED EQUATION:")
    print(f"  spike_c = {ALPHA:.0f} * N^({BETA:.3f}) * (V/{V_ref})^({g_opt:.1f}) * (L/{L_ref})^({d_opt:.1f})")

    # Predictions with new equation
    print(f"\n  Predictions with unified equation:")
    for name, m in sorted(models.items(), key=lambda x: x[1]['N']):
        pred_old = ALPHA * (m['N'] ** BETA)
        pred_new = ALPHA * (m['N'] ** BETA) * \
                   ((m['V'] / V_ref) ** g_opt) * \
                   ((m['L'] / L_ref) ** d_opt)
        actual = m['spike_actual'] if m['spike_actual'] else '?'
        print(f"  {name:>15s}: old={pred_old:.2f} new={pred_new:.2f} actual={actual}")

    # Extrapolations
    print(f"\n  Extrapolations:")
    extrapolations = {
        'GPT-4 (est)': {'N': 1.8e12, 'V': 100000, 'L': 120},
        'Gemini Ultra': {'N': 1.5e12, 'V': 256000, 'L': 64},
        'LLaMA-3-70B': {'N': 70e9, 'V': 128256, 'L': 80},
    }
    for name, m in extrapolations.items():
        pred = ALPHA * (m['N'] ** BETA) * \
               ((m['V'] / V_ref) ** g_opt) * \
               ((m['L'] / L_ref) ** d_opt)
        print(f"  {name:>15s}: spike_c = {pred:.3f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Gamma sweep
    gammas = sorted(gamma_sweep.keys())
    rmses = [gamma_sweep[g] for g in gammas]
    axes[0].plot(gammas, rmses, 'b.-', linewidth=2)
    axes[0].axvline(x=best_gamma, color='red', linestyle='--')
    axes[0].set_xlabel('Gamma (vocab exponent)')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title(f'Vocab Correction (best gamma={best_gamma:.1f})')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Old vs New predictions
    model_names = [n for n in models if models[n]['spike_actual'] is not None]
    actuals = [models[n]['spike_actual'] for n in model_names]
    old_preds = [ALPHA * (models[n]['N'] ** BETA) for n in model_names]
    new_preds = [ALPHA * (models[n]['N'] ** BETA) *
                 ((models[n]['V'] / V_ref) ** g_opt) *
                 ((models[n]['L'] / L_ref) ** d_opt) for n in model_names]
    x = range(len(model_names))
    axes[1].bar([i-0.25 for i in x], actuals, 0.25, label='Actual', color='green', alpha=0.7)
    axes[1].bar([i for i in x], old_preds, 0.25, label='Old eq', color='red', alpha=0.7)
    axes[1].bar([i+0.25 for i in x], new_preds, 0.25, label='New eq', color='blue', alpha=0.7)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(model_names, fontsize=7, rotation=30)
    axes[1].set_ylabel('Critical Spike')
    axes[1].set_title('Prediction Comparison')
    axes[1].legend(fontsize=7)

    # Plot 3: Extrapolation
    N_range = np.logspace(8, 13, 50)
    V_vals = {'V=50K': 50257, 'V=100K': 100000, 'V=256K': 256000}
    for vname, vval in V_vals.items():
        preds = [ALPHA * (n ** BETA) * ((vval / V_ref) ** g_opt) * ((48 / L_ref) ** d_opt)
                for n in N_range]
        axes[2].loglog(N_range, preds, linewidth=2, label=vname)
    axes[2].set_xlabel('Parameters (N)')
    axes[2].set_ylabel('Critical Spike')
    axes[2].set_title('Unified Extrapolation')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 35: Grand Unified Truth Equation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase35_unified.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 35, 'name': 'Grand Unified Truth Equation',
        'original_law': {'alpha': ALPHA, 'beta': BETA},
        'unified_law': {
            'alpha': ALPHA, 'beta': BETA,
            'gamma': round(g_opt, 2), 'delta': round(d_opt, 2),
            'V_ref': V_ref, 'L_ref': L_ref,
        },
        'rmse_original': round(float(gamma_sweep[0]), 3),
        'rmse_unified': round(best_rmse, 3),
        'model_data': {k: {kk: vv for kk, vv in v.items()} for k, v in models.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase35_unified.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 35 RESULTS: Grand Unified Truth Equation")
    print("=" * 70)
    print(f"  spike_c = {ALPHA:.0f} * N^({BETA:.3f}) * (V/{V_ref})^({g_opt:.1f}) * (L/{L_ref})^({d_opt:.1f})")
    print(f"  RMSE: {gamma_sweep[0]:.2f} -> {best_rmse:.2f}")
    print("=" * 70)
    phase_complete(35)
    return results

if __name__ == '__main__':
    main()
