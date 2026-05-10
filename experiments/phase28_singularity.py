# -*- coding: utf-8 -*-
"""
Phase 28: Macroscopic Singularity - Large Model Validation
Test the Truth Scaling Law (spike_c ~ N^-0.491) on larger models.
Use locally cached 7B models to validate extrapolation from GPT-2 124M.
"""
import os, json, sys, gc
import numpy as np
import torch
import torch.nn.functional as F
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

# GPT-2 scaling law from Phase 19
ALPHA = 85845.7508
BETA = -0.4910

def predicted_spike(n_params):
    return ALPHA * (n_params ** BETA)

# Factual QA tests
TESTS = [
    ("The capital of Japan is", "Tokyo"),
    ("The capital of France is", "Paris"),
    ("Water freezes at", "0"),
    ("The largest planet is", "Jupiter"),
    ("DNA stands for", "de"),
    ("The chemical symbol for gold is", "Au"),
    ("Shakespeare wrote", "Hamlet"),
    ("The speed of light is approximately", "299"),
]

def test_model_gpt2():
    """Test GPT-2 124M (baseline)."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("\n[P28] Testing GPT-2 (124M)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    pred = predicted_spike(n_params)

    # Find actual critical spike
    for spike in range(0, 20):
        correct = 0
        for prompt, expected in TESTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[:, -1, :].squeeze(0)
            fact_ids = tok.encode(' ' + expected)
            if fact_ids:
                logits[fact_ids[0]] += spike
            winner = tok.decode([torch.argmax(logits).item()]).strip()
            if winner.lower() == expected.lower():
                correct += 1
        if correct >= len(TESTS) * 0.75:
            actual = spike
            break
    else:
        actual = 20

    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  GPT-2: {n_params:,} params, predicted={pred:.2f}, actual={actual}")
    return {'model': 'GPT-2', 'params': n_params, 'predicted': pred, 'actual': actual}


def test_model_larger():
    """Try to test a larger model if available locally."""
    results = []

    # Try Qwen2.5-7B (from previous projects)
    model_candidates = [
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
        ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B"),
    ]

    for model_name, short_name in model_candidates:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            print(f"\n[P28] Testing {short_name}...")

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            tok = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, device_map="auto",
                torch_dtype=torch.float16, local_files_only=True,
                trust_remote_code=True)

            n_params = sum(p.numel() for p in model.parameters())
            # For quantized: estimate original param count
            if '7B' in short_name:
                n_params_est = 7_000_000_000
            else:
                n_params_est = n_params
            pred = predicted_spike(n_params_est)

            # Find actual critical spike
            actual = -1
            for spike in [0, 1, 2, 3, 5, 7, 10]:
                correct = 0
                for prompt, expected in TESTS:
                    full_prompt = prompt
                    inp = tok(full_prompt, return_tensors='pt').to(model.device)
                    with torch.no_grad():
                        out = model(**inp)
                    logits = out.logits[:, -1, :].squeeze(0).float()

                    fact_text = ' ' + expected
                    fact_ids = tok.encode(fact_text)
                    if fact_ids:
                        logits[fact_ids[-1]] += spike

                    winner = tok.decode([torch.argmax(logits).item()]).strip()
                    if expected.lower() in winner.lower():
                        correct += 1

                acc = correct / len(TESTS)
                print(f"    spike={spike}: {correct}/{len(TESTS)} = {acc:.0%}")
                if acc >= 0.75 and actual < 0:
                    actual = spike

            if actual < 0:
                actual = 10

            result = {
                'model': short_name, 'params': n_params_est,
                'predicted': round(pred, 2), 'actual': actual,
            }
            results.append(result)
            print(f"  {short_name}: {n_params_est:,} params, predicted={pred:.2f}, actual={actual}")

            del model, tok; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  {short_name}: SKIP ({str(e)[:60]})")
            continue

    return results


def main():
    print("=" * 70)
    print("  Phase 28: Macroscopic Singularity")
    print("  Validating Truth Scaling Law on larger models")
    print("=" * 70)

    print(f"\n  Scaling law: spike_c = {ALPHA:.1f} * N^({BETA:.4f})")
    print(f"  Predictions:")
    for name, n in [("GPT-2", 124e6), ("GPT-2-Med", 345e6), ("GPT-2-XL", 1.5e9),
                     ("7B", 7e9), ("70B", 70e9), ("175B", 175e9)]:
        print(f"    {name:>10s} ({n/1e9:.1f}B): spike_c = {predicted_spike(n):.3f}")

    # Test GPT-2
    gpt2_result = test_model_gpt2()
    all_results = [gpt2_result]

    # Test larger models
    larger_results = test_model_larger()
    all_results.extend(larger_results)

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Scaling law + data points
    n_range = np.logspace(7, 12, 100)
    predicted = [predicted_spike(n) for n in n_range]
    axes[0].loglog(n_range, predicted, 'b-', linewidth=2, label='Scaling Law')
    for r in all_results:
        color = 'green' if abs(r['predicted'] - r['actual']) < 3 else 'red'
        axes[0].scatter(r['params'], r['actual'], c=color, s=100, zorder=5,
                       label=f"{r['model']} (actual={r['actual']})")
        axes[0].scatter(r['params'], r['predicted'], c='blue', s=50, marker='x', zorder=5)
    axes[0].set_xlabel('Parameters')
    axes[0].set_ylabel('Critical Spike')
    axes[0].set_title('Truth Scaling Law Validation')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Prediction table
    axes[1].axis('off')
    table_data = [['Model', 'Params', 'Predicted', 'Actual']]
    for r in all_results:
        table_data.append([r['model'], f"{r['params']/1e6:.0f}M",
                          f"{r['predicted']:.2f}", str(r['actual'])])
    # Add extrapolations
    for name, n in [("70B", 70e9), ("175B", 175e9)]:
        table_data.append([name, f"{n/1e9:.0f}B", f"{predicted_spike(n):.3f}", "?"])
    table = axes[1].table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    axes[1].set_title('Scaling Law Predictions')

    # Plot 3: Prediction error
    if len(all_results) > 1:
        models = [r['model'] for r in all_results]
        errors = [r['actual'] - r['predicted'] for r in all_results]
        axes[2].bar(models, errors, color=['green' if abs(e) < 3 else 'red' for e in errors], alpha=0.7)
        axes[2].axhline(y=0, color='black', linewidth=0.5)
        axes[2].set_ylabel('Actual - Predicted')
        axes[2].set_title('Prediction Error')
    else:
        axes[2].text(0.5, 0.5, 'Need more models\nfor error analysis',
                    ha='center', va='center', fontsize=12, transform=axes[2].transAxes)
        axes[2].set_title('Prediction Error')

    plt.suptitle('Phase 28: Macroscopic Singularity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase28_singularity.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 28, 'name': 'Macroscopic Singularity',
        'scaling_law': {'alpha': ALPHA, 'beta': BETA},
        'model_results': all_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase28_singularity.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 28 RESULTS: Macroscopic Singularity")
    print("=" * 70)
    for r in all_results:
        print(f"  {r['model']:>12s}: predicted={r['predicted']:.2f}, actual={r['actual']}")
    print("=" * 70)

    phase_complete(28)
    return results

if __name__ == '__main__':
    main()
