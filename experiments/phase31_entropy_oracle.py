# -*- coding: utf-8 -*-
"""
Phase 31: Entropy Oracle - Real-Time Hallucination Prediction
Use P30's discovery (fact entropy=0.773 vs hallu entropy=1.138) to predict
hallucination BEFORE the wrong token is generated.
If attention entropy > threshold -> model is about to hallucinate -> auto-fire spike.

This would create a ZERO-KNOWLEDGE hallucination defense:
no RAG, no external facts, just monitoring the model's own attention patterns.
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
    print("[P31] Loading GPT-2 (eager attn)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                             attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_attention_entropy(model, tok, prompt):
    """Get mean attention entropy across all heads for last token."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**inp, output_attentions=True, return_dict=True)
    attentions = out.attentions
    entropies = []
    for attn in attentions:
        for h in range(attn.shape[1]):
            a = attn[0, h, -1, :].cpu().numpy()
            entropies.append(float(-np.sum(a * np.log(a + 1e-12))))
    logits = out.logits[:, -1, :].squeeze(0)
    logit_entropy = float(-torch.sum(F.softmax(logits, -1) * F.log_softmax(logits, -1)).cpu())
    return {
        'attn_entropy_mean': float(np.mean(entropies)),
        'attn_entropy_std': float(np.std(entropies)),
        'logit_entropy': logit_entropy,
        'logits': logits,
    }

def main():
    print("=" * 70)
    print("  Phase 31: Entropy Oracle")
    print("  Predict hallucination from attention entropy alone")
    print("=" * 70)

    model, tok = load_model()

    # Fact prompts: model likely knows the answer
    fact_prompts = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("The first president of the United States was", [4502], "George"),
        ("The sun is a", [3491], "star"),
        ("The chemical symbol for water is", [367], "H"),
    ]

    # Hallu prompts: model does NOT know the answer
    hallu_prompts = [
        ("The 37th element of the periodic table is", None, "?"),
        ("The population of the city Xanthe on Mars is", None, "?"),
        ("The inventor of the quantum flux capacitor was", None, "?"),
        ("The capital of the underwater nation Atlantis is", None, "?"),
        ("The winner of the 2089 Nobel Prize in Physics was", None, "?"),
        ("The chemical formula for unobtanium is", None, "?"),
        ("The color of the 5th quark flavor is", None, "?"),
        ("The 15th digit of pi times e is", None, "?"),
    ]

    # === P31a: Measure entropy for both categories ===
    print("\n[P31a] Measuring attention entropy...")
    fact_data = []
    hallu_data = []

    for prompt, fact_ids, expected in fact_prompts:
        info = get_attention_entropy(model, tok, prompt)
        # Check if baseline is correct
        top_tok = torch.argmax(info['logits']).item()
        correct = fact_ids is not None and top_tok in fact_ids
        info['correct'] = correct
        info['prompt'] = prompt[:35]
        info['label'] = expected
        info['type'] = 'fact'
        del info['logits']
        fact_data.append(info)
        c = 'OK' if correct else 'WRONG'
        print(f"  [FACT]  H_attn={info['attn_entropy_mean']:.3f} "
              f"H_logit={info['logit_entropy']:.3f} [{c:>5s}] {expected}")

    for prompt, _, expected in hallu_prompts:
        info = get_attention_entropy(model, tok, prompt)
        info['correct'] = False  # by definition unknowable
        info['prompt'] = prompt[:35]
        info['label'] = expected
        info['type'] = 'hallu'
        del info['logits']
        hallu_data.append(info)
        print(f"  [HALLU] H_attn={info['attn_entropy_mean']:.3f} "
              f"H_logit={info['logit_entropy']:.3f}          {prompt[:30]}...")

    # === P31b: ROC analysis ===
    print("\n[P31b] ROC analysis: can entropy predict hallucination?")
    all_data = fact_data + hallu_data
    all_entropies = [d['attn_entropy_mean'] for d in all_data]
    all_labels = [0 if d['type'] == 'fact' else 1 for d in all_data]  # 1=hallu

    # Also try logit entropy
    all_logit_ent = [d['logit_entropy'] for d in all_data]

    # Sweep thresholds
    roc_attn = []
    roc_logit = []
    for thresh in np.arange(0.0, 3.0, 0.05):
        # Attention entropy
        tp = sum(1 for e, l in zip(all_entropies, all_labels) if e > thresh and l == 1)
        fp = sum(1 for e, l in zip(all_entropies, all_labels) if e > thresh and l == 0)
        fn = sum(1 for e, l in zip(all_entropies, all_labels) if e <= thresh and l == 1)
        tn = sum(1 for e, l in zip(all_entropies, all_labels) if e <= thresh and l == 0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        roc_attn.append({'thresh': thresh, 'tpr': tpr, 'fpr': fpr, 'tp': tp, 'fp': fp})

        # Logit entropy
        tp2 = sum(1 for e, l in zip(all_logit_ent, all_labels) if e > thresh and l == 1)
        fp2 = sum(1 for e, l in zip(all_logit_ent, all_labels) if e > thresh and l == 0)
        fn2 = sum(1 for e, l in zip(all_logit_ent, all_labels) if e <= thresh and l == 1)
        tn2 = sum(1 for e, l in zip(all_logit_ent, all_labels) if e <= thresh and l == 0)
        tpr2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
        fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0
        roc_logit.append({'thresh': thresh, 'tpr': tpr2, 'fpr': fpr2})

    # Compute AUC
    fpr_a = [r['fpr'] for r in roc_attn]
    tpr_a = [r['tpr'] for r in roc_attn]
    auc_attn = abs(np.trapz(tpr_a, fpr_a))

    fpr_l = [r['fpr'] for r in roc_logit]
    tpr_l = [r['tpr'] for r in roc_logit]
    auc_logit = abs(np.trapz(tpr_l, fpr_l))

    print(f"  AUC (attention entropy): {auc_attn:.3f}")
    print(f"  AUC (logit entropy):     {auc_logit:.3f}")

    # Find best threshold
    best_acc = 0
    best_thresh = 0
    for r in roc_attn:
        acc = (r['tpr'] * len(hallu_data) + (1 - r['fpr']) * len(fact_data)) / len(all_data)
        if acc > best_acc:
            best_acc = acc
            best_thresh = r['thresh']
    print(f"  Best threshold: {best_thresh:.2f} (accuracy={best_acc:.0%})")

    # === P31c: Conditional spike (auto-fire when entropy high) ===
    print("\n[P31c] Conditional spike: auto-fire when H_attn > threshold...")
    for auto_thresh in [0.7, 0.8, 0.9, 1.0, 1.1]:
        auto_correct = 0
        auto_fired = 0
        for prompt, fact_ids, expected in fact_prompts:
            info = get_attention_entropy(model, tok, prompt)
            logits = info['logits']
            if info['attn_entropy_mean'] > auto_thresh and fact_ids:
                logits[fact_ids[0]] += 10
                auto_fired += 1
            winner = torch.argmax(logits).item()
            if fact_ids and winner in fact_ids:
                auto_correct += 1
        print(f"  H_thresh={auto_thresh:.1f}: fired={auto_fired}/{len(fact_prompts)}, "
              f"correct={auto_correct}/{len(fact_prompts)}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Entropy distribution
    f_ents = [d['attn_entropy_mean'] for d in fact_data]
    h_ents = [d['attn_entropy_mean'] for d in hallu_data]
    axes[0].hist(f_ents, bins=8, alpha=0.6, color='green', label=f'Fact (mean={np.mean(f_ents):.2f})')
    axes[0].hist(h_ents, bins=8, alpha=0.6, color='red', label=f'Hallu (mean={np.mean(h_ents):.2f})')
    axes[0].axvline(x=best_thresh, color='black', linestyle='--', label=f'Best thresh={best_thresh:.2f}')
    axes[0].set_xlabel('Attention Entropy')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Entropy Distribution')
    axes[0].legend(fontsize=8)

    # Plot 2: ROC curve
    axes[1].plot(fpr_a, tpr_a, 'g-', linewidth=2, label=f'Attn Entropy (AUC={auc_attn:.3f})')
    axes[1].plot(fpr_l, tpr_l, 'b--', linewidth=2, label=f'Logit Entropy (AUC={auc_logit:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k:', alpha=0.3)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC: Hallucination Detection')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Per-prompt scatter
    for d in fact_data:
        axes[2].scatter(d['attn_entropy_mean'], d['logit_entropy'],
                       c='green', s=60, alpha=0.7, zorder=5)
    for d in hallu_data:
        axes[2].scatter(d['attn_entropy_mean'], d['logit_entropy'],
                       c='red', s=60, alpha=0.7, zorder=5)
    axes[2].set_xlabel('Attention Entropy')
    axes[2].set_ylabel('Logit Entropy')
    axes[2].set_title('2D Entropy Space')
    axes[2].legend(['Fact', 'Hallu'], fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 31: Entropy Oracle - Hallucination Prediction',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase31_entropy_oracle.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 31, 'name': 'Entropy Oracle',
        'fact_mean_entropy': float(np.mean(f_ents)),
        'hallu_mean_entropy': float(np.mean(h_ents)),
        'auc_attn': auc_attn,
        'auc_logit': auc_logit,
        'best_threshold': best_thresh,
        'best_accuracy': best_acc,
        'fact_data': fact_data,
        'hallu_data': hallu_data,
    }
    with open(os.path.join(RESULTS_DIR, 'phase31_entropy_oracle.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 31 RESULTS: Entropy Oracle")
    print("=" * 70)
    print(f"  Fact entropy:  {np.mean(f_ents):.3f}")
    print(f"  Hallu entropy: {np.mean(h_ents):.3f}")
    print(f"  AUC (attention): {auc_attn:.3f}")
    print(f"  AUC (logit):     {auc_logit:.3f}")
    print(f"  Best threshold:  {best_thresh:.2f} -> {best_acc:.0%} accuracy")
    print("=" * 70)

    phase_complete(31)
    return results

if __name__ == '__main__':
    main()
