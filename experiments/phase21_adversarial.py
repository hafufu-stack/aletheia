# -*- coding: utf-8 -*-
"""
Phase 21: Adversarial Prompt Robustness
- Test if misleading context can defeat spike injection
- "Many believe the capital of Japan is Osaka" + spike=Tokyo
- Measure spike resilience against adversarial prompts
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
    print("[P21] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def eval_prompt(model, tok, prompt, fact_ids, spike_mag=0):
    inp = tok(prompt, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[:, -1, :].squeeze(0)
    for tid in fact_ids:
        if tid < logits.shape[0]:
            logits[tid] += spike_mag
    probs = F.softmax(logits, dim=-1)
    rank = (logits.argsort(descending=True) == fact_ids[0]).nonzero().item() + 1
    prob = float(probs[fact_ids[0]])
    correct = torch.argmax(probs).item() in fact_ids
    return correct, rank, prob


def main():
    print("=" * 70)
    print("  Phase 21: Adversarial Prompt Robustness")
    print("  Can misleading context defeat the spike?")
    print("=" * 70)

    model, tok = load_model()

    # Adversarial test cases: clean vs poisoned prompts
    cases = [
        {
            'clean': "The capital of Japan is",
            'adversarial': "Contrary to popular belief, many experts argue that the capital of Japan is not what most people think. The real capital of Japan is",
            'fact_ids': [11790], 'answer': 'Tokyo',
        },
        {
            'clean': "The capital of France is",
            'adversarial': "Some historians claim that the true capital of France has always been Lyon, not the commonly believed city. The capital of France is",
            'fact_ids': [6342], 'answer': 'Paris',
        },
        {
            'clean': "The largest planet is",
            'adversarial': "Recent astronomical discoveries have shown that Saturn is actually larger than previously thought. The largest planet is",
            'fact_ids': [22721], 'answer': 'Jupiter',
        },
        {
            'clean': "Water freezes at",
            'adversarial': "Under special quantum conditions, scientists discovered water freezes at a completely different temperature. Water freezes at",
            'fact_ids': [657], 'answer': '0',
        },
        {
            'clean': "DNA stands for",
            'adversarial': "A common misconception exists about what DNA actually stands for. Many textbooks are wrong. DNA stands for",
            'fact_ids': [390], 'answer': 'de',
        },
    ]

    spike_mags = [0, 3, 5, 7, 10, 15, 20]

    # === Sweep: clean vs adversarial at different spike levels ===
    print("\n[P21a] Clean vs Adversarial prompt + spike sweep...")
    clean_results = {}
    adv_results = {}

    for mag in spike_mags:
        clean_correct = 0
        adv_correct = 0
        clean_ranks = []
        adv_ranks = []

        for case in cases:
            c, r_c, _ = eval_prompt(model, tok, case['clean'], case['fact_ids'], mag)
            a, r_a, _ = eval_prompt(model, tok, case['adversarial'], case['fact_ids'], mag)
            clean_correct += int(c)
            adv_correct += int(a)
            clean_ranks.append(r_c)
            adv_ranks.append(r_a)

        clean_results[mag] = {'acc': clean_correct/len(cases),
                              'mean_rank': float(np.mean(clean_ranks))}
        adv_results[mag] = {'acc': adv_correct/len(cases),
                            'mean_rank': float(np.mean(adv_ranks))}
        print(f"  spike={mag:>3d}: clean={clean_correct}/5 (rank={np.mean(clean_ranks):.0f})  "
              f"adv={adv_correct}/5 (rank={np.mean(adv_ranks):.0f})")

    # === Per-case analysis at spike=10 ===
    print("\n[P21b] Per-case analysis (spike=10)...")
    per_case = []
    for case in cases:
        c_c, r_c, p_c = eval_prompt(model, tok, case['clean'], case['fact_ids'], 10)
        c_a, r_a, p_a = eval_prompt(model, tok, case['adversarial'], case['fact_ids'], 10)
        per_case.append({
            'answer': case['answer'],
            'clean_rank': r_c, 'adv_rank': r_a,
            'clean_correct': c_c, 'adv_correct': c_a,
            'rank_degradation': r_a - r_c,
        })
        status_c = 'OK' if c_c else 'FAIL'
        status_a = 'OK' if c_a else 'FAIL'
        print(f"  {case['answer']:>8s}: clean rank={r_c:>3d}({status_c})  "
              f"adv rank={r_a:>3d}({status_a})  "
              f"degradation={r_a - r_c:>+4d}")

    # Find minimum spike to overcome adversarial
    print("\n[P21c] Minimum spike to overcome adversarial...")
    for case in cases:
        for mag in range(0, 30):
            c, _, _ = eval_prompt(model, tok, case['adversarial'], case['fact_ids'], mag)
            if c:
                print(f"  {case['answer']:>8s}: adversarial defeated at spike={mag}")
                break

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    mags = sorted(clean_results.keys())
    c_acc = [clean_results[m]['acc']*100 for m in mags]
    a_acc = [adv_results[m]['acc']*100 for m in mags]
    axes[0].plot(mags, c_acc, 'g.-', label='Clean prompt', linewidth=2)
    axes[0].plot(mags, a_acc, 'r.-', label='Adversarial prompt', linewidth=2)
    axes[0].set_xlabel('Spike Magnitude')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Spike vs Adversarial Attack')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    c_rank = [clean_results[m]['mean_rank'] for m in mags]
    a_rank = [adv_results[m]['mean_rank'] for m in mags]
    axes[1].semilogy(mags, c_rank, 'g.-', label='Clean', linewidth=2)
    axes[1].semilogy(mags, a_rank, 'r.-', label='Adversarial', linewidth=2)
    axes[1].set_xlabel('Spike Magnitude')
    axes[1].set_ylabel('Mean Rank (log)')
    axes[1].set_title('Rank Degradation by Adversarial')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    answers = [p['answer'] for p in per_case]
    degradations = [p['rank_degradation'] for p in per_case]
    colors = ['green' if p['adv_correct'] else 'red' for p in per_case]
    axes[2].barh(answers, degradations, color=colors, alpha=0.7)
    axes[2].set_xlabel('Rank Degradation')
    axes[2].set_title('Per-Fact Adversarial Impact (spike=10)')

    plt.suptitle('Phase 21: Adversarial Robustness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase21_adversarial.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 21, 'name': 'Adversarial Prompt Robustness',
        'clean_sweep': {str(k): v for k, v in clean_results.items()},
        'adv_sweep': {str(k): v for k, v in adv_results.items()},
        'per_case': per_case,
    }
    with open(os.path.join(RESULTS_DIR, 'phase21_adversarial.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 21 RESULTS: Adversarial Robustness")
    print("=" * 70)
    print(f"  Clean spike=10: {clean_results[10]['acc']:.0%}")
    print(f"  Adversarial spike=10: {adv_results[10]['acc']:.0%}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
