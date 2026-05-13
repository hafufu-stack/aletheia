# -*- coding: utf-8 -*-
"""
Phase 99: Scaling Prediction to 90%
Fit a scaling law curve to all data points and predict
at what model size factual accuracy reaches 90%.
Deep Think's prediction: 7B-14B range.
"""
import json, os, sys, numpy as np
from scipy.optimize import curve_fit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

def log_model(N, alpha, beta):
    """Accuracy = alpha * log(N) + beta"""
    return alpha * np.log(N) + beta

def power_model(N, a, b):
    """Accuracy = a * N^b"""
    return a * np.power(N, b)

def main():
    print("[P99] Scaling Prediction to 90%")

    # Load P95 + P96 data
    data_points = []

    # P95 data (GPT-2 family)
    p95_path = os.path.join(RESULTS_DIR, 'phase95_grand_scaling.json')
    if os.path.exists(p95_path):
        p95 = json.load(open(p95_path))
        for label, r in p95['results'].items():
            data_points.append({
                'model': label,
                'params': r['params'],
                'natural_acc': r['natural_accuracy'],
                'code_acc': r['code_mode_accuracy'],
                'best_acc': r['best_accuracy'],
            })

    # P96 data (Qwen)
    p96_path = os.path.join(RESULTS_DIR, 'phase96_universal_094.json')
    if os.path.exists(p96_path):
        p96 = json.load(open(p96_path))
        for label, r in p96['results'].items():
            if label not in [d['model'] for d in data_points]:
                data_points.append({
                    'model': label,
                    'params': r['params'],
                    'natural_acc': r['natural_accuracy'],
                    'code_acc': r['code_mode_accuracy'],
                    'best_acc': r['best_accuracy'],
                })

    if not data_points:
        print("  No data available. Run P95/P96 first.")
        return

    # Sort by params
    data_points.sort(key=lambda x: x['params'])

    print("\n  Data Points:")
    for d in data_points:
        print(f"    {d['model']:20s}: {d['params']/1e6:.0f}M, nat={d['natural_acc']:.0%}, code={d['code_acc']:.0%}, best={d['best_acc']:.0%}")

    params = np.array([d['params'] for d in data_points])
    code_accs = np.array([d['code_acc'] for d in data_points])
    nat_accs = np.array([d['natural_acc'] for d in data_points])
    best_accs = np.array([d['best_acc'] for d in data_points])

    # Fit log model to Code Mode accuracy
    results = {}

    for acc_name, acc_data in [('code_mode', code_accs), ('natural', nat_accs), ('best_layer', best_accs)]:
        try:
            popt_log, _ = curve_fit(log_model, params, acc_data, p0=[0.1, -1.0], maxfev=10000)
            alpha_log, beta_log = popt_log

            # Predict N for 90% accuracy
            # 0.9 = alpha * log(N) + beta  =>  N = exp((0.9 - beta) / alpha)
            if alpha_log > 0:
                N_90 = np.exp((0.9 - beta_log) / alpha_log)
            else:
                N_90 = float('inf')

            # R-squared
            ss_res = np.sum((acc_data - log_model(params, *popt_log))**2)
            ss_tot = np.sum((acc_data - acc_data.mean())**2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)

            results[acc_name] = {
                'alpha': float(alpha_log),
                'beta': float(beta_log),
                'r_squared': float(r2),
                'N_90_percent': float(N_90),
                'N_90_billions': float(N_90 / 1e9),
            }
            print(f"\n  {acc_name}: Acc = {alpha_log:.4f} * ln(N) + {beta_log:.4f}")
            print(f"    R^2 = {r2:.4f}")
            print(f"    Predicted N for 90%: {N_90/1e9:.1f}B parameters")
        except Exception as e:
            print(f"  {acc_name}: fit failed: {e}")
            results[acc_name] = {'error': str(e)}

    out = {
        'phase': 99, 'name': 'Scaling Prediction',
        'data_points': data_points,
        'fits': results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase99_scaling_prediction.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Data + fit curves
    param_range = np.logspace(np.log10(params.min()*0.5), np.log10(100e9), 200)
    for acc_name, acc_data, color, marker in [
        ('code_mode', code_accs, '#2ecc71', 'o'),
        ('natural', nat_accs, '#e74c3c', 's'),
        ('best_layer', best_accs, '#3498db', '^'),
    ]:
        axes[0].scatter(params, acc_data, c=color, marker=marker, s=100,
                       label=f'{acc_name} (data)', zorder=5, edgecolors='black')
        if acc_name in results and 'alpha' in results[acc_name]:
            r = results[acc_name]
            pred = log_model(param_range, r['alpha'], r['beta'])
            pred = np.clip(pred, 0, 1)
            axes[0].plot(param_range, pred, color=color, linestyle='--', alpha=0.6,
                        label=f'{acc_name} fit (R2={r["r_squared"]:.2f})')

    axes[0].axhline(y=0.9, color='gold', linestyle=':', linewidth=2, label='90% target')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Parameters')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Scaling Law: Accuracy vs Model Size')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # 2. Predictions table
    axes[1].axis('off')
    table_data = [['Metric', 'Formula', 'R2', 'N for 90%']]
    for acc_name in ['natural', 'code_mode', 'best_layer']:
        if acc_name in results and 'alpha' in results[acc_name]:
            r = results[acc_name]
            formula = f"{r['alpha']:.3f}*ln(N) + {r['beta']:.3f}"
            n90 = f"{r['N_90_billions']:.1f}B"
            table_data.append([acc_name, formula, f"{r['r_squared']:.3f}", n90])
    table = axes[1].table(cellText=table_data[1:], colLabels=table_data[0],
                          loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    axes[1].set_title('Scaling Predictions', fontsize=14, fontweight='bold', y=0.85)

    fig.suptitle('Phase 99: When Does Accuracy Reach 90%?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase99_scaling_prediction.png'), dpi=150)
    plt.close()
    print("[Phase 99] Complete.")

if __name__ == '__main__':
    main()
