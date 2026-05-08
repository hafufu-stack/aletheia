# -*- coding: utf-8 -*-
"""
Phase 23: Spike x Prefill Fusion Cannon
- Combine P5 (output spike) + P17 (prefill) for maximum effect
- Test if combined approach achieves even lower entropy / higher accuracy
- The ultimate hallucination eradication method
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
    print("[P23] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def generate_method(model, tok, prompt, fact_ids, method, spike_mag=10, max_tokens=15):
    """Generate using different methods.
    method: 'baseline', 'spike_only', 'prefill_only', 'fusion'
    """
    if method in ('prefill_only', 'fusion'):
        # Add fact token to prompt
        fact_text = tok.decode(fact_ids)
        prompt_used = prompt + fact_text
    else:
        prompt_used = prompt

    ids = tok(prompt_used, return_tensors='pt')['input_ids'].to(DEVICE)
    gen = ids.clone()
    tokens = []
    entropies = []

    for step in range(max_tokens):
        with torch.no_grad():
            out = model(gen)
        logits = out.logits[:, -1, :].squeeze(0)

        # Apply spike at t=0 (for spike_only and fusion)
        if step == 0 and method in ('spike_only', 'fusion'):
            for tid in fact_ids:
                if tid < logits.shape[0]:
                    logits[tid] += spike_mag

        probs = F.softmax(logits, dim=-1)
        h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
        entropies.append(h)

        next_tok = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
        tokens.append(tok.decode([next_tok.item()]).encode('ascii', 'replace').decode())
        if next_tok.item() == tok.eos_token_id:
            break
        gen = torch.cat([gen, next_tok], dim=1)

    return ''.join(tokens), tokens, entropies


def main():
    print("=" * 70)
    print("  Phase 23: Spike x Prefill Fusion Cannon")
    print("  The ultimate hallucination eradication method")
    print("=" * 70)

    model, tok = load_model()

    test_cases = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("DNA stands for", [390], "de"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The speed of light is approximately", [22626], "299"),
    ]

    methods = ['baseline', 'spike_only', 'prefill_only', 'fusion']
    spike_mag = 10

    # === Compare all methods ===
    print(f"\n[P23a] Comparing all methods (spike_mag={spike_mag})...")
    all_results = {m: {'correct': 0, 'entropies': [], 'texts': []} for m in methods}

    for prompt, fact_ids, expected in test_cases:
        print(f"\n  {prompt[:35]}... (expect: {expected})")
        for method in methods:
            text, tokens, ents = generate_method(
                model, tok, prompt, fact_ids, method,
                spike_mag=spike_mag, max_tokens=15)
            # Check if expected appears
            correct = expected.lower() in text.lower()
            all_results[method]['correct'] += int(correct)
            all_results[method]['entropies'].append(np.mean(ents[:5]))
            all_results[method]['texts'].append(text[:50])
            status = 'OK' if correct else 'FAIL'
            print(f"    {method:>15s}: [{status}] {text[:45]}")

    # === Summary ===
    print(f"\n[P23b] Summary:")
    n = len(test_cases)
    for method in methods:
        acc = all_results[method]['correct'] / n
        ent = np.mean(all_results[method]['entropies'])
        print(f"  {method:>15s}: acc={acc:.0%} ({all_results[method]['correct']}/{n}), "
              f"entropy={ent:.2f}")

    # === Minimal spike for fusion ===
    print(f"\n[P23c] Minimum spike with fusion...")
    for test_mag in [0, 1, 2, 3, 5, 7, 10]:
        correct = 0
        for prompt, fact_ids, expected in test_cases:
            text, _, _ = generate_method(
                model, tok, prompt, fact_ids, 'fusion',
                spike_mag=test_mag, max_tokens=10)
            if expected.lower() in text.lower():
                correct += 1
        print(f"  fusion(spike={test_mag:>3d}): {correct}/{n}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Accuracy comparison
    method_labels = ['Baseline', 'Spike Only', 'Prefill Only', 'Fusion']
    accs = [all_results[m]['correct']/n*100 for m in methods]
    colors = ['red', 'orange', 'blue', 'green']
    axes[0].bar(method_labels, accs, color=colors, alpha=0.7)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Method Comparison')
    axes[0].set_ylim(0, 110)
    for i, v in enumerate(accs):
        axes[0].text(i, v+2, f'{v:.0f}%', ha='center', fontweight='bold')

    # Plot 2: Entropy comparison
    ents = [np.mean(all_results[m]['entropies']) for m in methods]
    axes[1].bar(method_labels, ents, color=colors, alpha=0.7)
    axes[1].set_ylabel('Mean Entropy')
    axes[1].set_title('Uncertainty Comparison')

    # Plot 3: Entropy trajectories (first test case)
    color_codes = ['r', 'orange', 'b', 'g']
    for mi, method in enumerate(methods):
        _, _, ents_traj = generate_method(
            model, tok, test_cases[0][0], test_cases[0][1], method,
            spike_mag=spike_mag, max_tokens=12)
        axes[2].plot(ents_traj[:10], '.-', color=color_codes[mi],
                     label=method_labels[mi], linewidth=2)
    axes[2].set_xlabel('Token Position')
    axes[2].set_ylabel('Entropy')
    axes[2].set_title(f'Entropy Trajectory: {test_cases[0][0][:25]}')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 23: Spike x Prefill Fusion Cannon',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase23_fusion.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 23, 'name': 'Spike x Prefill Fusion Cannon',
        'spike_magnitude': spike_mag,
        'accuracy': {m: all_results[m]['correct']/n for m in methods},
        'mean_entropy': {m: float(np.mean(all_results[m]['entropies'])) for m in methods},
        'texts': {m: all_results[m]['texts'] for m in methods},
    }
    with open(os.path.join(RESULTS_DIR, 'phase23_fusion.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 23 RESULTS: Fusion Cannon")
    print("=" * 70)
    for m in methods:
        print(f"  {m:>15s}: {all_results[m]['correct']}/{n}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
