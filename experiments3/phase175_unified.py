# -*- coding: utf-8 -*-
"""
Phase 175: Grand Unified Sword Equation
Fit g* = alpha * d^beta * L^gamma across ALL tested models.

Data points:
  Qwen-0.5B:  d=896,  L=24, g*=5
  Qwen-1.5B:  d=1536, L=28, g*=15
  Qwen-14B:   d=5120, L=48, g*=50
  GPT-2 Small: d=768,  L=12, g*=3
  GPT-2 Medium: d=1024, L=24, g*=? (from P168 sweep)
  GPT-2 Large: d=1280, L=36, g*=? (from P168 sweep)

Model: CPU only (regression analysis)
"""
import json, os, numpy as np, time
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    print("[P175] Grand Unified Sword Equation")
    start_time = time.time()

    # Known data points: (model_name, d, L, heads, g_star)
    # g* = gain that achieves peak accuracy in P159/P168
    known_data = [
        ("GPT-2 Small",  768,  12, 12,  3),
        ("Qwen-0.5B",    896,  24, 14,  5),
        ("Qwen-1.5B",    1536, 28, 12, 15),
        ("Qwen-14B",     5120, 48, 40, 50),
    ]

    # Load P168 sweep results to find best g* for GPT-2 Medium/Large
    try:
        p168 = json.load(open(os.path.join(RESULTS_DIR, 'phase168_excalibur.json')))
        for model_key, model_data in p168['results'].items():
            sweep = model_data.get('sweep', {})
            if sweep:
                best_g = max(sweep.keys(), key=lambda g: sweep[g]['num'])
                best_num = sweep[best_g]['num']
                d = model_data['d']
                n_layers = model_data['n_layers']
                name = model_data['model']
                # Only add if accuracy is decent
                if best_num >= 0.4:
                    # Convert string key to int
                    g_val = int(best_g) if isinstance(best_g, str) else best_g
                    known_data.append((name, d, n_layers, 0, g_val))
                    print(f"  Added {name}: d={d}, L={n_layers}, g*={g_val} (acc={best_num:.0%})")
    except Exception as e:
        print(f"  P168 data not available: {e}")

    print(f"\n  Data points: {len(known_data)}")
    for name, d, L, h, g in known_data:
        print(f"    {name:20s}: d={d:5d}, L={L:3d}, g*={g:3d}")

    names = [x[0] for x in known_data]
    ds = np.array([x[1] for x in known_data], dtype=float)
    Ls = np.array([x[2] for x in known_data], dtype=float)
    gs = np.array([x[4] for x in known_data], dtype=float)

    results = {}

    # Fit 1: g* = alpha * d^beta (original P159 equation)
    print("\n  === Fit 1: g* = alpha * d^beta ===")
    def model_d(d, alpha, beta):
        return alpha * d**beta
    try:
        popt1, pcov1 = curve_fit(model_d, ds, gs, p0=[0.1, 0.5], maxfev=5000)
        pred1 = model_d(ds, *popt1)
        rmse1 = np.sqrt(np.mean((pred1 - gs)**2))
        r2_1 = 1 - np.sum((gs - pred1)**2) / np.sum((gs - gs.mean())**2)
        print(f"    alpha={popt1[0]:.4f}, beta={popt1[1]:.4f}")
        print(f"    RMSE={rmse1:.2f}, R2={r2_1:.4f}")
        results['fit_d'] = {'alpha': popt1[0], 'beta': popt1[1],
                           'rmse': rmse1, 'r2': r2_1}
    except Exception as e:
        print(f"    Fit failed: {e}")
        popt1 = [0.13, 0.45]
        results['fit_d'] = {'alpha': 0.13, 'beta': 0.45, 'rmse': 0, 'r2': 0}

    # Fit 2: g* = alpha * d^beta * L^gamma (generalized)
    print("\n  === Fit 2: g* = alpha * d^beta * L^gamma ===")
    def model_dL(X, alpha, beta, gamma):
        d, L = X
        return alpha * d**beta * L**gamma
    try:
        popt2, pcov2 = curve_fit(model_dL, (ds, Ls), gs, p0=[0.01, 0.3, 0.3], maxfev=5000)
        pred2 = model_dL((ds, Ls), *popt2)
        rmse2 = np.sqrt(np.mean((pred2 - gs)**2))
        r2_2 = 1 - np.sum((gs - pred2)**2) / np.sum((gs - gs.mean())**2)
        print(f"    alpha={popt2[0]:.6f}, beta={popt2[1]:.4f}, gamma={popt2[2]:.4f}")
        print(f"    RMSE={rmse2:.2f}, R2={r2_2:.4f}")
        results['fit_dL'] = {'alpha': popt2[0], 'beta': popt2[1], 'gamma': popt2[2],
                            'rmse': rmse2, 'r2': r2_2}
    except Exception as e:
        print(f"    Fit failed: {e}")
        results['fit_dL'] = {'error': str(e)}

    # Fit 3: g* = alpha * (d*L)^beta (combined scale)
    print("\n  === Fit 3: g* = alpha * (d*L)^beta ===")
    def model_dxL(dL, alpha, beta):
        return alpha * dL**beta
    try:
        dxL = ds * Ls
        popt3, pcov3 = curve_fit(model_dxL, dxL, gs, p0=[0.001, 0.5], maxfev=5000)
        pred3 = model_dxL(dxL, *popt3)
        rmse3 = np.sqrt(np.mean((pred3 - gs)**2))
        r2_3 = 1 - np.sum((gs - pred3)**2) / np.sum((gs - gs.mean())**2)
        print(f"    alpha={popt3[0]:.6f}, beta={popt3[1]:.4f}")
        print(f"    RMSE={rmse3:.2f}, R2={r2_3:.4f}")
        results['fit_dxL'] = {'alpha': popt3[0], 'beta': popt3[1],
                             'rmse': rmse3, 'r2': r2_3}
    except Exception as e:
        print(f"    Fit failed: {e}")

    # Predictions for unseen models
    print("\n  === Predictions for Larger Models ===")
    pred_models = [
        ("Qwen-7B",     3584, 28),
        ("Qwen-32B",    5120, 64),
        ("Qwen-72B",    8192, 80),
        ("Llama-3-8B",  4096, 32),
        ("Llama-3-70B", 8192, 80),
    ]
    predictions = {}
    for name, d, L in pred_models:
        g_d = model_d(d, *popt1) if 'fit_d' in results else 0
        try:
            g_dL = model_dL((d, L), *popt2) if 'fit_dL' in results else 0
        except:
            g_dL = 0
        print(f"    {name:15s}: d={d:5d}, L={L:2d} -> g*(d)={g_d:.1f}, g*(d,L)={g_dL:.1f}")
        predictions[name] = {'d': d, 'L': L, 'g_d': round(g_d, 1), 'g_dL': round(g_dL, 1)}

    results['predictions'] = predictions

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase175_unified.json'), 'w') as f:
        json.dump({'phase': '175', 'name': 'Grand Unified Sword Equation',
                   'data_points': [{'name': n, 'd': d, 'L': L, 'g': g}
                                   for n, d, L, _, g in known_data],
                   'fits': results}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: g* vs d with fit
    ax = axes[0]
    d_range = np.linspace(500, 9000, 200)
    ax.scatter(ds, gs, c='blue', s=120, zorder=10, label='Measured')
    for i, name in enumerate(names):
        ax.annotate(name.split('-')[0], (ds[i], gs[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.plot(d_range, model_d(d_range, *popt1), 'r--', lw=2,
            label=f'$g^* = {popt1[0]:.3f} \\cdot d^{{{popt1[1]:.3f}}}$')
    # Plot predictions
    for name, p in predictions.items():
        ax.scatter([p['d']], [p['g_d']], c='red', marker='*', s=100, zorder=5, alpha=0.5)
    ax.set_xlabel('Hidden Dimension d', fontsize=12)
    ax.set_ylabel('Critical Gain g*', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('Equation 1: g* ~ d^beta', fontsize=13, fontweight='bold')

    # Middle: Log-log plot
    ax = axes[1]
    ax.scatter(np.log(ds), np.log(gs), c='blue', s=120, zorder=10)
    for i, name in enumerate(names):
        ax.annotate(name.split('-')[0], (np.log(ds[i]), np.log(gs[i])),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    log_d_range = np.linspace(np.log(500), np.log(9000), 100)
    ax.plot(log_d_range, np.log(popt1[0]) + popt1[1] * log_d_range, 'r--', lw=2)
    ax.set_xlabel('log(d)', fontsize=12)
    ax.set_ylabel('log(g*)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('Log-Log Linearity Check', fontsize=13, fontweight='bold')

    # Right: R2 comparison
    ax = axes[2]
    fit_names = []
    r2_vals = []
    for k in ['fit_d', 'fit_dL', 'fit_dxL']:
        if k in results and 'r2' in results[k]:
            label = {'fit_d': '$d^\\beta$', 'fit_dL': '$d^\\beta L^\\gamma$',
                     'fit_dxL': '$(dL)^\\beta$'}[k]
            fit_names.append(label)
            r2_vals.append(results[k]['r2'])
    colors = ['#e74c3c' if v > 0.95 else '#f39c12' if v > 0.9 else '#3498db' for v in r2_vals]
    ax.bar(fit_names, r2_vals, color=colors, alpha=0.8)
    for i, v in enumerate(r2_vals):
        ax.text(i, v+0.01, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('R-squared', fontsize=12)
    ax.set_ylim(0.8, 1.05); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Model Comparison', fontsize=13, fontweight='bold')

    plt.suptitle('Phase 175: Grand Unified Sword Equation\n'
                 'One equation to predict g* for ANY Transformer',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase175_unified.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    best_fit = max([(k, results[k].get('r2', 0)) for k in ['fit_d', 'fit_dL', 'fit_dxL']
                    if k in results], key=lambda x: x[1])
    print(f"  -> Best model: {best_fit[0]} (R2={best_fit[1]:.4f})")
    if best_fit[1] > 0.95:
        print("  -> THE EQUATION IS UNIVERSAL!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 175] Complete.")


if __name__ == '__main__':
    main()
