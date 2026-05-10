# -*- coding: utf-8 -*-
"""
Phase 36: Entropy-Guided Self-Correction
The TRUE holy grail: reject high-entropy tokens and try alternatives.
NO external knowledge. NO fact IDs. Just attention entropy as a compass.

Algorithm:
1. Generate next token candidate
2. Measure attention entropy
3. If entropy > threshold: REJECT, try 2nd-best token
4. Repeat until entropy drops or max retries
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

ENTROPY_THRESHOLD = 1.0  # From P31

def load_model():
    print("[P36] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def measure_entropy(model, input_ids):
    """Measure mean attention entropy for current sequence."""
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, return_dict=True)
    entropies = []
    for attn in out.attentions:
        for h in range(attn.shape[1]):
            a = attn[0, h, -1, :].cpu().numpy()
            entropies.append(float(-np.sum(a * np.log(a + 1e-12))))
    return float(np.mean(entropies)), out.logits[:, -1, :].squeeze(0)

def generate_baseline(model, tok, prompt, max_tokens=15):
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:]).encode('ascii','replace').decode()

def generate_self_correcting(model, tok, prompt, max_tokens=15, max_retries=5):
    """Generate with entropy-guided self-correction."""
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    gen = ids.clone()
    tokens = []
    corrections = 0
    entropy_trace = []

    for step in range(max_tokens):
        ent, logits = measure_entropy(model, gen)
        entropy_trace.append(ent)

        # Get ranked candidates
        sorted_indices = logits.argsort(descending=True)

        # Try candidates until entropy is acceptable
        chosen = None
        for retry in range(min(max_retries, len(sorted_indices))):
            candidate = sorted_indices[retry].item()
            # Simulate: append candidate and check entropy
            test_seq = torch.cat([gen, torch.tensor([[candidate]], device=DEVICE)], dim=1)
            test_ent, _ = measure_entropy(model, test_seq)

            if test_ent < ENTROPY_THRESHOLD or retry == max_retries - 1:
                chosen = candidate
                if retry > 0:
                    corrections += 1
                break

        if chosen is None:
            chosen = sorted_indices[0].item()

        tokens.append(tok.decode([chosen]).encode('ascii','replace').decode())
        if chosen == tok.eos_token_id:
            break
        gen = torch.cat([gen, torch.tensor([[chosen]], device=DEVICE)], dim=1)

    return ''.join(tokens), tokens, corrections, entropy_trace

def main():
    print("=" * 70)
    print("  Phase 36: Entropy-Guided Self-Correction")
    print("  Zero-knowledge hallucination defense via entropy rejection")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", [11790], "Tokyo", "fact"),
        ("The capital of France is", [6342], "Paris", "fact"),
        ("Water freezes at", [657], "0", "fact"),
        ("The largest planet is", [22721], "Jupiter", "fact"),
        ("Albert Einstein developed the theory of", [44449], "relativity", "fact"),
        ("The sun is a", [3491], "star", "fact"),
        ("The 37th element of the periodic table is", None, "?", "hallu"),
        ("The capital of the underwater nation Atlantis is", None, "?", "hallu"),
    ]

    # === P36a: Baseline vs Self-Correcting ===
    print(f"\n[P36a] Baseline vs Self-Correcting (threshold={ENTROPY_THRESHOLD})...")
    results_list = []

    for prompt, fact_ids, expected, ptype in tests:
        # Baseline
        base_text = generate_baseline(model, tok, prompt, 12)

        # Self-correcting
        sc_text, sc_tokens, n_corrections, ent_trace = \
            generate_self_correcting(model, tok, prompt, 12, max_retries=5)

        # Check first token accuracy
        base_inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            base_out = model(**base_inp)
        base_first = torch.argmax(base_out.logits[:, -1, :]).item()
        base_correct = fact_ids is not None and base_first in fact_ids

        sc_first_tok = tok.encode(sc_tokens[0]) if sc_tokens else []
        sc_correct = fact_ids is not None and any(t in fact_ids for t in sc_first_tok)

        results_list.append({
            'prompt': prompt[:35], 'type': ptype, 'expected': expected,
            'base_text': base_text[:45], 'base_correct': base_correct,
            'sc_text': sc_text[:45], 'sc_correct': sc_correct,
            'corrections': n_corrections,
            'mean_entropy': round(float(np.mean(ent_trace)), 3) if ent_trace else 0,
        })

        b = 'OK' if base_correct else 'FAIL'
        s = 'OK' if sc_correct else 'FAIL'
        print(f"  [{ptype:>5s}] {expected:>10s}: base=[{b}] sc=[{s}] corrections={n_corrections}")
        print(f"    Base: {base_text[:40]}")
        print(f"    SC:   {sc_text[:40]}")

    # === P36b: Correction rate analysis ===
    print("\n[P36b] Correction statistics...")
    fact_corrections = [r['corrections'] for r in results_list if r['type'] == 'fact']
    hallu_corrections = [r['corrections'] for r in results_list if r['type'] == 'hallu']
    print(f"  Fact prompts:  mean corrections = {np.mean(fact_corrections):.1f}")
    print(f"  Hallu prompts: mean corrections = {np.mean(hallu_corrections):.1f}")

    base_acc = sum(1 for r in results_list if r['base_correct']) / len(tests)
    sc_acc = sum(1 for r in results_list if r['sc_correct']) / len(tests)
    # For hallu prompts, "correct" = refusing/changing output
    total_corrections = sum(r['corrections'] for r in results_list)

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Accuracy comparison
    labels_short = [r['expected'][:6] for r in results_list]
    base_vals = [int(r['base_correct']) for r in results_list]
    sc_vals = [int(r['sc_correct']) for r in results_list]
    x = range(len(labels_short))
    axes[0].bar([i-0.2 for i in x], base_vals, 0.4, label='Baseline', color='red', alpha=0.7)
    axes[0].bar([i+0.2 for i in x], sc_vals, 0.4, label='Self-Correct', color='green', alpha=0.7)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels_short, fontsize=7, rotation=45)
    axes[0].set_ylabel('Correct (1=yes)')
    axes[0].set_title('Baseline vs Self-Correcting')
    axes[0].legend(fontsize=8)

    # Plot 2: Corrections per prompt
    corr_vals = [r['corrections'] for r in results_list]
    corr_colors = ['green' if r['type']=='fact' else 'red' for r in results_list]
    axes[1].bar(labels_short, corr_vals, color=corr_colors, alpha=0.7)
    axes[1].set_ylabel('# Corrections')
    axes[1].set_title('Entropy Rejections per Prompt')
    axes[1].tick_params(axis='x', rotation=45, labelsize=7)

    # Plot 3: Entropy traces
    for r in results_list[:4]:  # First 4 fact prompts
        axes[2].axhline(y=ENTROPY_THRESHOLD, color='black', linestyle='--', alpha=0.3)
    axes[2].bar(['Fact\ncorrections', 'Hallu\ncorrections'],
               [np.mean(fact_corrections), np.mean(hallu_corrections)],
               color=['green', 'red'], alpha=0.7)
    axes[2].set_ylabel('Mean Corrections')
    axes[2].set_title('Fact vs Hallu Correction Rate')

    plt.suptitle('Phase 36: Entropy-Guided Self-Correction\n(Zero-Knowledge Defense)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase36_self_correction.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 36, 'name': 'Entropy-Guided Self-Correction',
        'baseline_accuracy': base_acc,
        'self_correcting_accuracy': sc_acc,
        'total_corrections': total_corrections,
        'fact_mean_corrections': float(np.mean(fact_corrections)),
        'hallu_mean_corrections': float(np.mean(hallu_corrections)),
        'per_case': results_list,
    }
    with open(os.path.join(RESULTS_DIR, 'phase36_self_correction.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 36 RESULTS: Entropy-Guided Self-Correction")
    print("=" * 70)
    print(f"  Baseline accuracy:       {base_acc:.0%}")
    print(f"  Self-correcting accuracy: {sc_acc:.0%}")
    print(f"  Total corrections: {total_corrections}")
    print("=" * 70)
    phase_complete(36)
    return results

if __name__ == '__main__':
    main()
