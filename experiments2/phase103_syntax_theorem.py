# -*- coding: utf-8 -*-
"""
Phase 103: The Absolute Syntax Theorem
Prove that best layer = L_total - c (constant offset) rather than
alpha * L_total (proportional). Deep Think's hypothesis: c ~ 1-3.
Pure math on existing data, no model loading needed.
"""
import json, os, numpy as np
from scipy.optimize import curve_fit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

def proportional_model(L, alpha):
    """L_best = alpha * L"""
    return alpha * L

def offset_model(L, c):
    """L_best = L - c"""
    return L - c

def main():
    print("[P103] The Absolute Syntax Theorem")

    # Collect all data points from P96
    data = [
        ('GPT2-Small',  'GPT-2', 12, 10),
        ('GPT2-Medium', 'GPT-2', 24, 22),
        ('GPT2-Large',  'GPT-2', 36, 34),
        ('GPT2-XL',     'GPT-2', 48, 45),
        ('Qwen-0.5B',   'Qwen',  24, 23),
        ('Qwen-1.5B',   'Qwen',  28, 27),
    ]

    # Also add P100 data
    p100_path = os.path.join(RESULTS_DIR, 'phase100_grand_validation.json')
    if os.path.exists(p100_path):
        p100 = json.load(open(p100_path))
        data.append(('Qwen-7B', 'Qwen', p100['n_layers'], p100['best_layer']))

    names = [d[0] for d in data]
    archs = [d[1] for d in data]
    L_total = np.array([d[2] for d in data], dtype=float)
    L_best = np.array([d[3] for d in data], dtype=float)

    # Compute c = L_total - L_best for each model
    c_values = L_total - L_best
    print("\n  Raw data:")
    print(f"  {'Model':<15} {'Arch':<6} {'L_total':<8} {'L_best':<8} {'c = L-Lbest':<12} {'alpha=Lb/Lt':<10}")
    for i in range(len(data)):
        alpha = L_best[i] / L_total[i]
        print(f"  {names[i]:<15} {archs[i]:<6} {int(L_total[i]):<8} {int(L_best[i]):<8} {c_values[i]:<12.0f} {alpha:<10.3f}")

    # Fit 1: Proportional model (L_best = alpha * L)
    popt_prop, _ = curve_fit(proportional_model, L_total, L_best)
    alpha_fit = popt_prop[0]
    ss_res_prop = np.sum((L_best - proportional_model(L_total, alpha_fit))**2)
    ss_tot = np.sum((L_best - L_best.mean())**2)
    r2_prop = 1 - ss_res_prop / ss_tot

    # Fit 2: Offset model (L_best = L - c)
    popt_off, _ = curve_fit(offset_model, L_total, L_best, p0=[2.0])
    c_fit = popt_off[0]
    ss_res_off = np.sum((L_best - offset_model(L_total, c_fit))**2)
    r2_off = 1 - ss_res_off / ss_tot

    # AIC comparison (with 1 parameter each, same k, compare residuals directly)
    rmse_prop = np.sqrt(ss_res_prop / len(L_total))
    rmse_off = np.sqrt(ss_res_off / len(L_total))

    print(f"\n  === MODEL COMPARISON ===")
    print(f"  Proportional: L_best = {alpha_fit:.4f} * L  (R2={r2_prop:.4f}, RMSE={rmse_prop:.3f})")
    print(f"  Offset:       L_best = L - {c_fit:.2f}      (R2={r2_off:.4f}, RMSE={rmse_off:.3f})")
    print(f"\n  WINNER: {'OFFSET (Absolute Syntax Boundary)' if r2_off > r2_prop else 'PROPORTIONAL (Aletheia Constant)'}")
    print(f"  c (Absolute Syntax Layers) = {c_fit:.2f}")
    print(f"  Mean c (raw): {c_values.mean():.2f} +/- {c_values.std():.2f}")

    results = {
        'phase': 103, 'name': 'Absolute Syntax Theorem',
        'data': [{'model': names[i], 'arch': archs[i], 'L_total': int(L_total[i]),
                  'L_best': int(L_best[i]), 'c': float(c_values[i])} for i in range(len(data))],
        'proportional_fit': {'alpha': float(alpha_fit), 'R2': float(r2_prop), 'RMSE': float(rmse_prop)},
        'offset_fit': {'c': float(c_fit), 'R2': float(r2_off), 'RMSE': float(rmse_off)},
        'winner': 'offset' if r2_off > r2_prop else 'proportional',
        'mean_c': float(c_values.mean()),
        'std_c': float(c_values.std()),
    }
    with open(os.path.join(RESULTS_DIR, 'phase103_syntax_theorem.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. c values per model
    colors = ['#3498db' if a == 'GPT-2' else '#e74c3c' for a in archs]
    axes[0].bar(names, c_values, color=colors)
    axes[0].axhline(y=c_fit, color='gold', linestyle='--', linewidth=2, label=f'Mean c={c_fit:.1f}')
    axes[0].set_ylabel('c = L_total - L_best')
    axes[0].set_title('Absolute Syntax Layers per Model')
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].legend()

    # 2. Proportional vs Offset fit
    L_range = np.linspace(10, 55, 100)
    axes[1].scatter(L_total, L_best, c=colors, s=150, edgecolors='black', zorder=5)
    axes[1].plot(L_range, proportional_model(L_range, alpha_fit), '--',
                color='#e74c3c', label=f'Proportional: {alpha_fit:.3f}*L (R2={r2_prop:.3f})')
    axes[1].plot(L_range, offset_model(L_range, c_fit), '-',
                color='#2ecc71', linewidth=2, label=f'Offset: L-{c_fit:.1f} (R2={r2_off:.3f})')
    axes[1].plot(L_range, L_range, ':', color='gray', alpha=0.3, label='L_best = L')
    for i, name in enumerate(names):
        axes[1].annotate(name, (L_total[i], L_best[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[1].set_xlabel('Total Layers (L)')
    axes[1].set_ylabel('Best Layer (L_best)')
    axes[1].set_title('Proportional vs Offset Model')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3. Residuals comparison
    res_prop = L_best - proportional_model(L_total, alpha_fit)
    res_off = L_best - offset_model(L_total, c_fit)
    x = np.arange(len(names))
    w = 0.35
    axes[2].bar(x - w/2, np.abs(res_prop), w, label=f'Proportional (RMSE={rmse_prop:.2f})',
               color='#e74c3c', alpha=0.7)
    axes[2].bar(x + w/2, np.abs(res_off), w, label=f'Offset (RMSE={rmse_off:.2f})',
               color='#2ecc71', alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    axes[2].set_ylabel('|Residual| (layers)')
    axes[2].set_title('Residual Comparison')
    axes[2].legend()

    fig.suptitle('Phase 103: The Absolute Syntax Theorem', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase103_syntax_theorem.png'), dpi=150)
    plt.close()
    print("[Phase 103] Complete.")

if __name__ == '__main__':
    main()
