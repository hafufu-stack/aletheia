# -*- coding: utf-8 -*-
"""
Phase 27: Cognitive Dissonance Firewall
Detect anti-spike attacks by measuring semantic collision between
logit_bias direction and internal fact representation.
"""
import os, json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import phase_complete

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_model():
    print("[P27] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def main():
    print("=" * 70)
    print("  Phase 27: Cognitive Dissonance Firewall")
    print("  Detect anti-spike attacks via semantic collision")
    print("=" * 70)

    model, tok = load_model()
    lm_head = model.lm_head

    tests = [
        ("The capital of Japan is", 11790, "Tokyo",
         [6342, 3334, 5765], ["Paris", "London", "Beijing"]),
        ("The capital of France is", 6342, "Paris",
         [11790, 3334, 7753], ["Tokyo", "London", "Rome"]),
        ("The largest planet is", 22721, "Jupiter",
         [16309, 7733, 11563], ["Saturn", "Mars", "Earth"]),
        ("Water freezes at", 657, "0",
         [1802, 2167, 1120], ["100", "50", "20"]),
    ]

    # === P27a: Compute dissonance score ===
    print("\n[P27a] Computing dissonance for truth vs anti-spike...")

    all_scores = []
    for prompt, correct_id, correct_name, wrong_ids, wrong_names in tests:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        base_logits = out.logits[:, -1, :].squeeze(0)

        # Get model's internal belief (top-k logit direction)
        belief_vec = base_logits / (base_logits.norm() + 1e-12)

        # Truth spike: add to correct token
        truth_bias = torch.zeros_like(base_logits)
        truth_bias[correct_id] = 10.0
        truth_dir = truth_bias / (truth_bias.norm() + 1e-12)
        truth_cos = F.cosine_similarity(belief_vec.unsqueeze(0), truth_dir.unsqueeze(0)).item()

        # Anti-spike: add to wrong tokens
        for wid, wname in zip(wrong_ids, wrong_names):
            anti_bias = torch.zeros_like(base_logits)
            anti_bias[wid] = 10.0
            anti_dir = anti_bias / (anti_bias.norm() + 1e-12)
            anti_cos = F.cosine_similarity(belief_vec.unsqueeze(0), anti_dir.unsqueeze(0)).item()

            # Dissonance = how much the bias direction conflicts with model belief
            dissonance = truth_cos - anti_cos
            is_attack = anti_cos < truth_cos

            all_scores.append({
                'prompt': prompt[:30], 'correct': correct_name,
                'target': wname, 'type': 'anti-spike',
                'truth_cos': round(truth_cos, 4),
                'anti_cos': round(anti_cos, 4),
                'dissonance': round(dissonance, 4),
                'detected': is_attack,
            })

        # Also score the truth spike
        all_scores.append({
            'prompt': prompt[:30], 'correct': correct_name,
            'target': correct_name, 'type': 'truth-spike',
            'truth_cos': round(truth_cos, 4),
            'anti_cos': round(truth_cos, 4),
            'dissonance': 0.0,
            'detected': False,
        })

    # Print results
    for s in all_scores:
        tag = 'ATTACK' if s['type'] == 'anti-spike' else 'TRUTH'
        det = 'DETECTED' if s['detected'] else 'passed'
        print(f"  [{tag:>6s}] {s['correct']:>7s}->{s['target']:>7s}: "
              f"cos={s['anti_cos']:+.4f} dissonance={s['dissonance']:+.4f} [{det}]")

    # === P27b: Threshold sweep for firewall ===
    print("\n[P27b] Firewall threshold sweep...")
    truth_spikes = [s for s in all_scores if s['type'] == 'truth-spike']
    anti_spikes = [s for s in all_scores if s['type'] == 'anti-spike']

    threshold_results = {}
    for thresh in np.arange(-0.01, 0.02, 0.002):
        tp = sum(1 for s in anti_spikes if s['dissonance'] > thresh)  # correct blocks
        fp = sum(1 for s in truth_spikes if s['dissonance'] > thresh)  # wrong blocks
        fn = len(anti_spikes) - tp
        tn = len(truth_spikes) - fp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        threshold_results[round(thresh, 4)] = {
            'precision': precision, 'recall': recall,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }
        print(f"  thresh={thresh:+.4f}: block {tp}/{len(anti_spikes)} attacks, "
              f"false-block {fp}/{len(truth_spikes)} truths "
              f"(P={precision:.2f} R={recall:.2f})")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Cosine similarity distribution
    truth_cos_vals = [s['truth_cos'] for s in truth_spikes]
    anti_cos_vals = [s['anti_cos'] for s in anti_spikes]
    axes[0].hist(truth_cos_vals, bins=10, alpha=0.7, color='green', label='Truth spike')
    axes[0].hist(anti_cos_vals, bins=10, alpha=0.7, color='red', label='Anti-spike')
    axes[0].set_xlabel('Cosine Similarity with Model Belief')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Truth vs Anti-Spike Distribution')
    axes[0].legend()

    # Plot 2: Dissonance scores
    dissonances = [s['dissonance'] for s in anti_spikes]
    labels = [f"{s['correct']}->{s['target']}" for s in anti_spikes]
    colors_bar = ['red' if s['detected'] else 'orange' for s in anti_spikes]
    axes[1].barh(range(len(dissonances)), dissonances, color=colors_bar, alpha=0.7)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=6)
    axes[1].set_xlabel('Dissonance Score')
    axes[1].set_title('Per-Attack Dissonance')

    # Plot 3: Precision-Recall
    threshs = sorted(threshold_results.keys())
    precs = [threshold_results[t]['precision'] for t in threshs]
    recs = [threshold_results[t]['recall'] for t in threshs]
    axes[2].plot(recs, precs, 'b.-', linewidth=2)
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Firewall Precision-Recall')
    axes[2].set_xlim(0, 1.05)
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 27: Cognitive Dissonance Firewall', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase27_firewall.png'), dpi=150, bbox_inches='tight')
    plt.close()

    detection_rate = sum(1 for s in anti_spikes if s['detected']) / len(anti_spikes) if anti_spikes else 0
    results = {
        'phase': 27, 'name': 'Cognitive Dissonance Firewall',
        'detection_rate': detection_rate,
        'scores': all_scores,
        'threshold_sweep': {str(k): v for k, v in threshold_results.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase27_firewall.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 27 RESULTS: Cognitive Dissonance Firewall")
    print("=" * 70)
    print(f"  Anti-spike detection rate: {detection_rate:.0%}")
    print("=" * 70)

    phase_complete(27)
    return results

if __name__ == '__main__':
    main()
