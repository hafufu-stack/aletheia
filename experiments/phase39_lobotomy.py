# -*- coding: utf-8 -*-
"""
Phase 39: Dynamic Skill Lobotomy
When entropy is high, dynamically mute Skill heads and boost Fact heads.
The model's "ability to lie fluently" is surgically disabled at the moment
it tries to hallucinate.
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

# From P14: heads whose ablation hurts fact accuracy (fact heads)
# and heads whose ablation helps fact accuracy (skill/hallu heads)
FACT_HEADS = [(2,1),(0,3),(1,5),(3,7),(4,2)]  # Top fact heads
SKILL_HEADS = [(9,6),(7,4),(5,9),(11,0),(8,7)]  # Top skill heads

ENTROPY_THRESHOLD = 1.0

def load_model():
    print("[P39] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_entropy(model, input_ids):
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, return_dict=True)
    ents = []
    for attn in out.attentions:
        for h in range(attn.shape[1]):
            a = attn[0, h, -1, :].cpu().numpy()
            ents.append(float(-np.sum(a * np.log(a + 1e-12))))
    return float(np.mean(ents)), out

def lobotomy_forward(model, tok, prompt, mute_factor=0.0, boost_factor=2.0):
    """Forward pass with Skill heads muted and Fact heads boosted."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    # Install hooks to modify head outputs
    modifications = {}
    handles = []

    for li in range(12):
        def make_hook(layer_idx):
            def hook_fn(module, args, output):
                # output[0] = hidden state (batch, seq, d_model)
                hs = output[0].clone()
                for (fl, fh) in FACT_HEADS:
                    if fl == layer_idx:
                        start = fh * d_head
                        end = (fh + 1) * d_head
                        hs[:, :, start:end] *= boost_factor
                for (sl, sh) in SKILL_HEADS:
                    if sl == layer_idx:
                        start = sh * d_head
                        end = (sh + 1) * d_head
                        hs[:, :, start:end] *= mute_factor
                return (hs,) + output[1:]
            return hook_fn
        h = model.transformer.h[li].register_forward_hook(make_hook(li))
        handles.append(h)

    with torch.no_grad():
        out = model(**inp, output_attentions=True, return_dict=True)

    for h in handles:
        h.remove()

    logits = out.logits[:, -1, :].squeeze(0)
    return logits

def main():
    print("=" * 70)
    print("  Phase 39: Dynamic Skill Lobotomy")
    print("  Mute skill heads + boost fact heads when entropy high")
    print("=" * 70)

    model, tok = load_model()

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

    # === P39a: Mute factor sweep ===
    print(f"\n[P39a] Skill mute factor sweep (boost=2.0)...")
    mute_sweep = {}
    for mute in [1.0, 0.5, 0.1, 0.0]:
        correct = 0
        for prompt, fact_ids, _ in tests:
            logits = lobotomy_forward(model, tok, prompt, mute_factor=mute, boost_factor=2.0)
            if torch.argmax(logits).item() in fact_ids:
                correct += 1
        mute_sweep[mute] = correct / len(tests)
        print(f"  mute={mute:.1f}: {correct}/{len(tests)} = {correct/len(tests):.0%}")

    # === P39b: Boost factor sweep ===
    print(f"\n[P39b] Fact boost factor sweep (mute=0.0)...")
    boost_sweep = {}
    for boost in [1.0, 1.5, 2.0, 3.0, 5.0]:
        correct = 0
        for prompt, fact_ids, _ in tests:
            logits = lobotomy_forward(model, tok, prompt, mute_factor=0.0, boost_factor=boost)
            if torch.argmax(logits).item() in fact_ids:
                correct += 1
        boost_sweep[boost] = correct / len(tests)
        print(f"  boost={boost:.1f}: {correct}/{len(tests)} = {correct/len(tests):.0%}")

    # === P39c: Conditional lobotomy (only when entropy > threshold) ===
    print(f"\n[P39c] Conditional lobotomy (only when H > {ENTROPY_THRESHOLD})...")
    cond_results = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        ent, base_out = get_entropy(model, inp['input_ids'])
        base_logits = base_out.logits[:, -1, :].squeeze(0)

        if ent > ENTROPY_THRESHOLD:
            logits = lobotomy_forward(model, tok, prompt, mute_factor=0.0, boost_factor=2.0)
            action = 'LOBOTOMY'
        else:
            logits = base_logits
            action = 'PASS'

        winner = torch.argmax(logits).item()
        correct = winner in fact_ids
        base_correct = torch.argmax(base_logits).item() in fact_ids

        # Rank comparison
        base_rank = int((base_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
        lob_rank = int((logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1

        cond_results.append({
            'expected': expected, 'entropy': round(ent, 3),
            'action': action, 'correct': correct, 'base_correct': base_correct,
            'base_rank': base_rank, 'lobotomy_rank': lob_rank,
        })
        b = 'OK' if base_correct else 'FAIL'
        l = 'OK' if correct else 'FAIL'
        print(f"  {expected:>8s}: H={ent:.3f} [{action:>9s}] "
              f"base=[{b}]r={base_rank:>4d} lob=[{l}]r={lob_rank:>4d}")

    cond_acc = sum(1 for r in cond_results if r['correct']) / len(tests)
    base_acc = sum(1 for r in cond_results if r['base_correct']) / len(tests)

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    mutes = sorted(mute_sweep.keys(), reverse=True)
    axes[0].plot([str(m) for m in mutes], [mute_sweep[m]*100 for m in mutes],
                'r.-', linewidth=2, markersize=10)
    axes[0].set_xlabel('Skill Mute Factor')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Skill Head Muting')
    axes[0].grid(True, alpha=0.3)

    boosts = sorted(boost_sweep.keys())
    axes[1].plot([str(b) for b in boosts], [boost_sweep[b]*100 for b in boosts],
                'g.-', linewidth=2, markersize=10)
    axes[1].set_xlabel('Fact Boost Factor')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Fact Head Boosting')
    axes[1].grid(True, alpha=0.3)

    labels = [r['expected'] for r in cond_results]
    base_ranks = [r['base_rank'] for r in cond_results]
    lob_ranks = [r['lobotomy_rank'] for r in cond_results]
    x = range(len(labels))
    axes[2].bar([i-0.2 for i in x], base_ranks, 0.4, label='Baseline', color='red', alpha=0.7)
    axes[2].bar([i+0.2 for i in x], lob_ranks, 0.4, label='Lobotomy', color='green', alpha=0.7)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[2].set_ylabel('Fact Token Rank')
    axes[2].set_title(f'Conditional Lobotomy (base={base_acc:.0%} lob={cond_acc:.0%})')
    axes[2].legend(fontsize=8)

    plt.suptitle('Phase 39: Dynamic Skill Lobotomy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase39_lobotomy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 39, 'name': 'Dynamic Skill Lobotomy',
        'mute_sweep': {str(k): v for k, v in mute_sweep.items()},
        'boost_sweep': {str(k): v for k, v in boost_sweep.items()},
        'baseline_accuracy': base_acc,
        'conditional_accuracy': cond_acc,
        'conditional_results': cond_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase39_lobotomy.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 39 RESULTS: Dynamic Skill Lobotomy")
    print("=" * 70)
    print(f"  Baseline: {base_acc:.0%}")
    print(f"  Conditional lobotomy: {cond_acc:.0%}")
    print("=" * 70)
    phase_complete(39)

if __name__ == '__main__':
    main()
