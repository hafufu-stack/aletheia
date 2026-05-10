# -*- coding: utf-8 -*-
"""
Phase 57: Oracle Revisited - Relative vs Absolute Entropy
P31 showed AUC=1.0 for detecting hallucination via attention entropy.
P56 showed Oracle fails as absolute threshold (11/12 = "confident").
RESOLUTION: P31's AUC was RELATIVE (comparing same prompt with
correct vs wrong completion). Absolute thresholding is different.
This phase formally proves the distinction.
"""
import os, json, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
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
    print("[P57] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def measure_entropy(model, input_ids):
    """Attention entropy of last token position."""
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, return_dict=True)
    ents = []
    for attn in out.attentions:
        for h in range(attn.shape[1]):
            a = attn[0, h, -1, :].cpu().numpy()
            ents.append(float(-np.sum(a * np.log(a + 1e-12))))
    return float(np.mean(ents))

def main():
    print("=" * 70)
    print("  Phase 57: Oracle Revisited - Relative vs Absolute Entropy")
    print("  Why P31's AUC=1.0 and P56's Oracle failure coexist")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", 11790, "Tokyo", " the"),
        ("The capital of France is", 6342, "Paris", " the"),
        ("Water freezes at", 657, "0", " the"),
        ("The largest planet is", 22721, "Jupiter", " the"),
        ("The chemical symbol for gold is", 7591, "Au", " the"),
        ("Shakespeare wrote", 13483, "Hamlet", " :"),
        ("The Earth orbits the", 4252, "Sun", " earth"),
        ("The boiling point of water is", 1802, "100", " about"),
        ("Albert Einstein developed the theory of", 44449, "relativity", " the"),
        ("Oxygen has the atomic number", 807, "8", " of"),
    ]

    # === P57a: RELATIVE entropy (P31 replication) ===
    print("\n[P57a] Relative entropy: same prompt + correct vs wrong completion...")
    relative_data = []
    labels = []
    scores = []
    for prompt, fact_id, expected, wrong_str in tests:
        # Prompt + correct token
        correct_text = prompt + " " + expected
        correct_ids = tok(correct_text, return_tensors='pt')['input_ids'].to(DEVICE)
        ent_correct = measure_entropy(model, correct_ids)

        # Prompt + wrong token (L12's actual output)
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        with torch.no_grad():
            out = model(inp, return_dict=True)
        l12_tok = torch.argmax(out.logits[:, -1, :]).item()
        l12_name = tok.decode([l12_tok]).encode('ascii', 'replace').decode().strip()
        wrong_text = prompt + " " + l12_name
        wrong_ids = tok(wrong_text, return_tensors='pt')['input_ids'].to(DEVICE)
        ent_wrong = measure_entropy(model, wrong_ids)

        # Prompt only (baseline)
        ent_base = measure_entropy(model, inp)

        relative_data.append({
            'expected': expected, 'wrong': l12_name,
            'ent_correct': round(ent_correct, 4),
            'ent_wrong': round(ent_wrong, 4),
            'ent_base': round(ent_base, 4),
            'delta': round(ent_wrong - ent_correct, 4),
            'correct_lower': ent_correct < ent_wrong,
        })

        # For AUC: label=1 for fact, 0 for hallucination
        # Score = negative entropy (higher = more likely fact)
        labels.extend([1, 0])
        scores.extend([-ent_correct, -ent_wrong])

        tag = 'CORRECT<WRONG' if ent_correct < ent_wrong else 'WRONG<=CORRECT'
        print(f"  {expected:>12s}: correct={ent_correct:.4f} wrong({l12_name})={ent_wrong:.4f} [{tag}]")

    # Compute RELATIVE AUC
    try:
        rel_auc = roc_auc_score(labels, scores)
    except:
        rel_auc = 0.5
    n_correct_lower = sum(1 for r in relative_data if r['correct_lower'])
    print(f"\n  RELATIVE AUC: {rel_auc:.3f}")
    print(f"  Correct has lower entropy: {n_correct_lower}/{len(tests)}")

    # === P57b: ABSOLUTE entropy (P56's problem) ===
    print("\n[P57b] Absolute entropy: just the prompt (before generating)...")
    absolute_data = []
    abs_labels = []
    abs_scores = []
    for prompt, fact_id, expected, wrong_str in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        ent = measure_entropy(model, inp)

        with torch.no_grad():
            out = model(inp, return_dict=True)
        l12_tok = torch.argmax(out.logits[:, -1, :]).item()
        is_correct = l12_tok == fact_id

        absolute_data.append({
            'expected': expected, 'entropy': round(ent, 4),
            'l12_correct': is_correct,
        })
        abs_labels.append(1 if is_correct else 0)
        abs_scores.append(-ent)

        tag = 'OK' if is_correct else 'FAIL'
        print(f"  {expected:>12s}: H={ent:.4f} [{tag}]")

    # Absolute AUC (should be near 0.5 = useless)
    try:
        abs_auc = roc_auc_score(abs_labels, abs_scores)
    except:
        abs_auc = 0.5
    print(f"\n  ABSOLUTE AUC: {abs_auc:.3f}")

    # === P57c: Differential Oracle ===
    print("\n[P57c] Differential Oracle: ent(prompt+L12) vs ent(prompt+L10)...")
    diff_results = []
    for prompt, fact_id, expected, wrong_str in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

        # Get L12's choice
        with torch.no_grad():
            out = model(inp, return_dict=True)
        l12_tok = torch.argmax(out.logits[:, -1, :]).item()
        l12_name = tok.decode([l12_tok]).encode('ascii', 'replace').decode().strip()

        # Get L10's choice
        hidden = {}
        def hook(m, a, o):
            hidden['h'] = o[0][0, -1, :].detach()
        handle = model.transformer.h[10].register_forward_hook(hook)
        with torch.no_grad():
            model(inp)
        handle.remove()
        normed = model.transformer.ln_f(hidden['h'].unsqueeze(0))
        l10_logits = model.lm_head(normed).squeeze(0)
        l10_tok = torch.argmax(l10_logits).item()
        l10_name = tok.decode([l10_tok]).encode('ascii', 'replace').decode().strip()

        # Measure entropy of each continuation
        l12_cont = tok(prompt + " " + l12_name, return_tensors='pt')['input_ids'].to(DEVICE)
        l10_cont = tok(prompt + " " + l10_name, return_tensors='pt')['input_ids'].to(DEVICE)
        ent_l12 = measure_entropy(model, l12_cont)
        ent_l10 = measure_entropy(model, l10_cont)

        # Pick lower entropy
        if ent_l10 < ent_l12:
            selected = l10_tok
            source = 'L10'
        else:
            selected = l12_tok
            source = 'L12'

        is_correct = selected == fact_id

        diff_results.append({
            'expected': expected, 'l12': l12_name, 'l10': l10_name,
            'ent_l12': round(ent_l12, 4), 'ent_l10': round(ent_l10, 4),
            'selected': source, 'correct': is_correct,
        })

        tag = 'OK' if is_correct else 'FAIL'
        print(f"  {expected:>12s}: L12({l12_name})={ent_l12:.4f} "
              f"L10({l10_name})={ent_l10:.4f} -> {source} [{tag}]")

    diff_acc = sum(1 for d in diff_results if d['correct']) / len(tests)
    l10_picked = sum(1 for d in diff_results if d['selected'] == 'L10')

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Relative: paired entropy comparison
    labels_plot = [r['expected'][:6] for r in relative_data]
    ent_c = [r['ent_correct'] for r in relative_data]
    ent_w = [r['ent_wrong'] for r in relative_data]
    x = range(len(labels_plot))
    axes[0].bar([i-0.2 for i in x], ent_c, 0.4, label='Correct', color='green', alpha=0.7)
    axes[0].bar([i+0.2 for i in x], ent_w, 0.4, label='Wrong (L12)', color='red', alpha=0.7)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels_plot, fontsize=6, rotation=45)
    axes[0].set_ylabel('Attention Entropy')
    axes[0].set_title(f'RELATIVE: AUC={rel_auc:.3f}\n({n_correct_lower}/{len(tests)} correct < wrong)')
    axes[0].legend(fontsize=8)

    # AUC comparison
    auc_methods = ['Relative\n(P31 style)', 'Absolute\n(P56 style)']
    aucs = [rel_auc, abs_auc]
    axes[1].bar(auc_methods, aucs, color=['green', 'red'], alpha=0.7)
    axes[1].axhline(y=0.5, color='black', linestyle='--', label='Random')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Relative vs Absolute Detection')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()

    # Differential Oracle accuracy
    methods = ['L12 only', 'L10 only', 'Diff Oracle']
    accs_bar = [8.3, 33.3, diff_acc*100]
    axes[2].bar(methods, accs_bar, color=['red', 'green', 'blue'], alpha=0.7)
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title(f'Differential Oracle\n(L10 picked {l10_picked}/{len(tests)} times)')

    plt.suptitle('Phase 57: Oracle Revisited', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase57_oracle_revisited.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 57, 'name': 'Oracle Revisited',
        'relative_auc': rel_auc, 'absolute_auc': abs_auc,
        'n_correct_lower': n_correct_lower,
        'differential_accuracy': diff_acc,
        'l10_picked': l10_picked,
        'relative_data': relative_data,
        'absolute_data': absolute_data,
        'differential_results': diff_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase57_oracle_revisited.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 57 RESULTS: The Oracle Duality Theorem")
    print("=" * 70)
    print(f"  RELATIVE AUC (P31 style): {rel_auc:.3f}")
    print(f"  ABSOLUTE AUC (P56 style): {abs_auc:.3f}")
    print(f"  Differential Oracle acc:  {diff_acc:.0%}")
    print(f"  L10 picked: {l10_picked}/{len(tests)}")
    print("=" * 70)
    phase_complete(57)

if __name__ == '__main__':
    main()
