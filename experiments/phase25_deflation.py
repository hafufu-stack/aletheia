# -*- coding: utf-8 -*-
"""
Phase 25: Anti-Skill Deflation
Suppress hedge words that Skill Heads use to construct fluent lies.
logit_bias=-100 on "perhaps", "typically", "suggests" etc.
Forces model to output raw facts or silence.
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

# Hedge words: grammatical glue that enables fluent lies
HEDGE_WORDS = [
    " perhaps", " typically", " suggests", " likely", " probably",
    " generally", " often", " usually", " sometimes", " maybe",
    " might", " could", " would", " should", " appears",
    " seems", " believed", " considered", " thought", " known",
    " actually", " really", " basically", " certainly", " obviously",
]

def load_model():
    print("[P25] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def gen_with_suppression(model, tok, prompt, suppress_ids, max_tokens=20):
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    gen = ids.clone()
    tokens = []
    for step in range(max_tokens):
        with torch.no_grad():
            out = model(gen)
        logits = out.logits[:, -1, :].squeeze(0)
        for sid in suppress_ids:
            if sid < logits.shape[0]:
                logits[sid] = -100.0
        next_tok = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        tokens.append(tok.decode([next_tok.item()]).encode('ascii', 'replace').decode())
        if next_tok.item() == tok.eos_token_id:
            break
        gen = torch.cat([gen, next_tok], dim=1)
    return ''.join(tokens), tokens

def eval_accuracy(model, tok, prompt, fact_ids, suppress_ids):
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[:, -1, :].squeeze(0)
    for sid in suppress_ids:
        if sid < logits.shape[0]:
            logits[sid] = -100.0
    return torch.argmax(logits).item() in fact_ids

def main():
    print("=" * 70)
    print("  Phase 25: Anti-Skill Deflation")
    print("  Suppress hedge words -> force raw facts or silence")
    print("=" * 70)

    model, tok = load_model()

    # Build suppress set
    suppress_ids = set()
    for word in HEDGE_WORDS:
        ids = tok.encode(word)
        suppress_ids.update(ids)
    suppress_ids = list(suppress_ids)
    print(f"  Suppressing {len(suppress_ids)} token IDs from {len(HEDGE_WORDS)} hedge words")

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("DNA stands for", [390], "de"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The speed of light is approximately", [22626], "299"),
    ]

    # === P25a: Compare baseline vs deflated ===
    print("\n[P25a] Baseline vs Anti-Skill Deflation...")
    results_comparison = []
    baseline_correct = 0
    deflated_correct = 0

    for prompt, fact_ids, expected in tests:
        # Baseline
        text_base, _ = gen_with_suppression(model, tok, prompt, [], 15)
        base_ok = eval_accuracy(model, tok, prompt, fact_ids, [])
        baseline_correct += int(base_ok)

        # Deflated
        text_defl, _ = gen_with_suppression(model, tok, prompt, suppress_ids, 15)
        defl_ok = eval_accuracy(model, tok, prompt, fact_ids, suppress_ids)
        deflated_correct += int(defl_ok)

        results_comparison.append({
            'prompt': prompt[:35], 'expected': expected,
            'baseline_text': text_base[:50], 'baseline_ok': base_ok,
            'deflated_text': text_defl[:50], 'deflated_ok': defl_ok,
        })
        b_status = 'OK' if base_ok else 'FAIL'
        d_status = 'OK' if defl_ok else 'FAIL'
        print(f"  {expected:>8s}:")
        print(f"    Baseline:  [{b_status}] {text_base[:45]}")
        print(f"    Deflated:  [{d_status}] {text_defl[:45]}")

    # === P25b: Gradual suppression (how many hedge words needed?) ===
    print("\n[P25b] Gradual suppression sweep...")
    sweep_results = {}
    for n_words in [0, 5, 10, 15, 20, 25]:
        subset = HEDGE_WORDS[:n_words]
        sub_ids = set()
        for w in subset:
            sub_ids.update(tok.encode(w))
        sub_ids = list(sub_ids)

        correct = 0
        for prompt, fact_ids, _ in tests:
            if eval_accuracy(model, tok, prompt, fact_ids, sub_ids):
                correct += 1
        sweep_results[n_words] = correct / len(tests)
        print(f"  n_hedge={n_words:>3d}: {correct}/{len(tests)} = {correct/len(tests):.0%}")

    # === P25c: Combined spike + deflation ===
    print("\n[P25c] Spike + Deflation combo...")
    combo_results = {}
    for spike_mag in [0, 3, 5, 7, 10]:
        correct = 0
        for prompt, fact_ids, _ in tests:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[:, -1, :].squeeze(0)
            for sid in suppress_ids:
                if sid < logits.shape[0]:
                    logits[sid] = -100.0
            for fid in fact_ids:
                if fid < logits.shape[0]:
                    logits[fid] += spike_mag
            if torch.argmax(logits).item() in fact_ids:
                correct += 1
        combo_results[spike_mag] = correct / len(tests)
        print(f"  spike={spike_mag:>3d} + deflation: {correct}/{len(tests)} = {correct/len(tests):.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(['Baseline', 'Deflated'],
                [baseline_correct/len(tests)*100, deflated_correct/len(tests)*100],
                color=['red', 'teal'], alpha=0.7)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Hedge Word Suppression Effect')
    axes[0].set_ylim(0, 110)

    ns = sorted(sweep_results.keys())
    axes[1].plot(ns, [sweep_results[n]*100 for n in ns], 'b.-', linewidth=2)
    axes[1].set_xlabel('# Hedge Words Suppressed')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Gradual Suppression Sweep')
    axes[1].grid(True, alpha=0.3)

    spikes = sorted(combo_results.keys())
    axes[2].plot(spikes, [combo_results[s]*100 for s in spikes], 'g.-', linewidth=2)
    axes[2].set_xlabel('Spike Magnitude (+ deflation)')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Spike + Deflation Combo')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 25: Anti-Skill Deflation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase25_deflation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 25, 'name': 'Anti-Skill Deflation',
        'baseline_acc': baseline_correct / len(tests),
        'deflated_acc': deflated_correct / len(tests),
        'sweep': {str(k): v for k, v in sweep_results.items()},
        'combo': {str(k): v for k, v in combo_results.items()},
        'comparison': results_comparison,
    }
    with open(os.path.join(RESULTS_DIR, 'phase25_deflation.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 25 RESULTS: Anti-Skill Deflation")
    print("=" * 70)
    print(f"  Baseline: {baseline_correct}/{len(tests)}")
    print(f"  Deflated: {deflated_correct}/{len(tests)}")
    print("=" * 70)

    phase_complete(25)
    return results

if __name__ == '__main__':
    main()
