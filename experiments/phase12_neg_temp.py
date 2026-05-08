# -*- coding: utf-8 -*-
"""
Phase 12: Negative Temperature Thermodynamics
- Apply T < 0 to softmax: inverts probability distribution
- Combine with antimatter CD to extract truth from inverted space
- Explore the forbidden thermodynamic regime
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
    print("[P12] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def neg_temp_decode(logits, temp):
    """Apply temperature (including negative) to logits."""
    if abs(temp) < 1e-8:
        # T=0: argmax
        probs = torch.zeros_like(logits)
        probs[..., torch.argmax(logits, dim=-1)] = 1.0
        return probs
    return F.softmax(logits / temp, dim=-1)


def main():
    print("=" * 70)
    print("  Phase 12: Negative Temperature Thermodynamics")
    print("  Truth Extraction from the Forbidden Regime")
    print("=" * 70)

    model, tok = load_model()

    qa_pairs = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("Water freezes at", [657], "0"),
        ("The chemical formula for water is", [367], "H"),
        ("The largest planet is", [22721], "Jupiter"),
        ("DNA stands for", [390], "de"),
    ]

    # === Temperature sweep including negative ===
    temperatures = [-2.0, -1.0, -0.5, -0.3, -0.1,
                    0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    print("\n[P12a] Temperature sweep (positive and NEGATIVE)...")
    results_by_temp = {}

    for temp in temperatures:
        correct = 0
        entropies = []
        top1_tokens = []

        for prompt, fact_ids, fact_text in qa_pairs:
            inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[:, -1, :].squeeze(0)

            probs = neg_temp_decode(logits, temp)
            h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
            entropies.append(h)

            top1_id = torch.argmax(probs).item()
            top1_text = tok.decode([top1_id]).strip()
            top1_tokens.append(top1_text)

            if top1_id in fact_ids:
                correct += 1

        results_by_temp[temp] = {
            'accuracy': correct / len(qa_pairs),
            'mean_entropy': float(np.mean(entropies)),
            'top1_samples': top1_tokens[:3],
        }
        safe_tokens = [t.encode('ascii', 'replace').decode() for t in top1_tokens[:3]]
        sign = '+' if temp > 0 else ''
        print(f"  T={sign}{temp:.1f}: acc={correct}/5, "
              f"H={np.mean(entropies):.2f}, "
              f"top1={safe_tokens}")

    # === Antimatter + Negative Temperature combo ===
    print("\n[P12b] Antimatter CD + Negative Temperature combo...")
    combo_results = {}
    alphas = [0.0, 0.3, 0.5, 1.0]
    neg_temps = [-0.5, -0.3, -0.1]

    for alpha in alphas:
        for neg_t in neg_temps:
            correct = 0
            for prompt, fact_ids, _ in qa_pairs:
                inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
                with torch.no_grad():
                    out = model(**inp)
                logits = out.logits[:, -1, :].squeeze(0)

                # Antimatter: amplified logits
                anti_logits = logits * 2.0
                # Contrastive
                adjusted = logits - alpha * anti_logits
                # Negative temperature
                probs = neg_temp_decode(adjusted, neg_t)

                if torch.argmax(probs).item() in fact_ids:
                    correct += 1

            key = f"a={alpha}_T={neg_t}"
            combo_results[key] = correct / len(qa_pairs)
            print(f"  alpha={alpha}, T={neg_t}: acc={correct}/5")

    # === Analysis: what does negative temperature select? ===
    print("\n[P12c] Analyzing negative temperature token selection...")
    test_prompt = "The capital of Japan is"
    inp = tok(test_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[:, -1, :].squeeze(0)

    for temp in [1.0, 0.1, -0.1, -1.0]:
        probs = neg_temp_decode(logits, temp)
        top5_ids = torch.topk(probs, 5).indices.tolist()
        top5_tokens = [tok.decode([tid]).strip() for tid in top5_ids]
        top5_probs = [float(probs[tid]) for tid in top5_ids]
        top5_safe = [(t.encode('ascii', 'replace').decode(), f'{p:.3f}') for t, p in zip(top5_tokens, top5_probs)]
        sign = '+' if temp > 0 else ''
        print(f"  T={sign}{temp}: top5 = {top5_safe}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    temps_sorted = sorted(results_by_temp.keys())
    accs = [results_by_temp[t]['accuracy']*100 for t in temps_sorted]
    ents = [results_by_temp[t]['mean_entropy'] for t in temps_sorted]

    # Plot 1: Accuracy vs Temperature (including negative)
    pos_temps = [t for t in temps_sorted if t > 0]
    neg_temps_list = [t for t in temps_sorted if t < 0]
    pos_accs = [results_by_temp[t]['accuracy']*100 for t in pos_temps]
    neg_accs = [results_by_temp[t]['accuracy']*100 for t in neg_temps_list]
    axes[0].plot(pos_temps, pos_accs, 'b.-', label='T > 0 (normal)', linewidth=2)
    axes[0].plot(neg_temps_list, neg_accs, 'r.-', label='T < 0 (forbidden)', linewidth=2)
    axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy Across Temperature Regimes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Entropy
    axes[1].plot(temps_sorted, ents, 'go-', linewidth=2)
    axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Entropy: Normal vs Negative T')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Combo heatmap
    combo_data = np.zeros((len(alphas), len(neg_temps)))
    for i, a in enumerate(alphas):
        for j, nt in enumerate(neg_temps):
            key = f"a={a}_T={nt}"
            combo_data[i, j] = combo_results.get(key, 0) * 100
    im = axes[2].imshow(combo_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    axes[2].set_xticks(range(len(neg_temps)))
    axes[2].set_xticklabels([str(t) for t in neg_temps])
    axes[2].set_yticks(range(len(alphas)))
    axes[2].set_yticklabels([str(a) for a in alphas])
    axes[2].set_xlabel('Negative Temperature')
    axes[2].set_ylabel('Antimatter Alpha')
    axes[2].set_title('Combo: CD + Negative T')
    plt.colorbar(im, ax=axes[2], label='Accuracy %')

    plt.suptitle('Phase 12: Negative Temperature Thermodynamics',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase12_neg_temp.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 12, 'name': 'Negative Temperature Thermodynamics',
        'temp_sweep': {str(k): v for k, v in results_by_temp.items()},
        'combo_results': combo_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase12_neg_temp.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 12 RESULTS: Negative Temperature")
    print("=" * 70)
    print(f"  T=+1.0: {results_by_temp[1.0]['accuracy']:.0%}")
    print(f"  T=-1.0: {results_by_temp[-1.0]['accuracy']:.0%}")
    print(f"  T=-0.1: {results_by_temp[-0.1]['accuracy']:.0%}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
