# -*- coding: utf-8 -*-
"""
Phase 45: Dynamic Skill Lobotomy v2
P39 failed because skill heads were muted ALL the time -> grammar collapse.
This time: mute ONLY for the single token where Oracle detects hallucination.
Pinpoint surgical strike, not sustained lobotomy.
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

ENTROPY_THRESHOLD = 1.0
SKILL_HEADS = [(9,6),(7,4),(5,9),(11,0),(8,7),(10,3)]
FACT_HEADS = [(2,1),(0,3),(1,5),(3,7),(4,2),(6,8)]

def load_model():
    print("[P45] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_entropy(attentions):
    ents = []
    for attn in attentions:
        for h in range(attn.shape[1]):
            a = attn[0, h, -1, :].cpu().numpy()
            ents.append(float(-np.sum(a * np.log(a + 1e-12))))
    return float(np.mean(ents))

def lobotomy_forward_v2(model, tok, prompt, mute_skill=True, boost_fact=True,
                         mute_factor=0.0, boost_factor=2.0):
    """Single-token lobotomy: mute skill + boost fact for first generated token only."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    handles = []
    for li in range(12):
        def make_hook(layer_idx):
            def hook_fn(module, args, output):
                hs = output[0].clone()
                if mute_skill:
                    for (sl, sh) in SKILL_HEADS:
                        if sl == layer_idx:
                            start = sh * d_head
                            end = (sh + 1) * d_head
                            hs[:, -1, start:end] *= mute_factor  # Only last token pos!
                if boost_fact:
                    for (fl, fh) in FACT_HEADS:
                        if fl == layer_idx:
                            start = fh * d_head
                            end = (fh + 1) * d_head
                            hs[:, -1, start:end] *= boost_factor  # Only last token pos!
                return (hs,) + output[1:]
            return hook_fn
        h = model.transformer.h[li].register_forward_hook(make_hook(li))
        handles.append(h)

    with torch.no_grad():
        out = model(**inp, output_attentions=True, return_dict=True)

    for h in handles:
        h.remove()

    return out.logits[:, -1, :].squeeze(0), out.attentions

def main():
    print("=" * 70)
    print("  Phase 45: Dynamic Skill Lobotomy v2")
    print("  Pinpoint muting: only at the hallucination moment")
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
        ("The Earth orbits the", [4252], "Sun"),
        ("The boiling point of water is", [1802], "100"),
    ]

    # === P45a: Always-on lobotomy (like P39 but v2 last-token-only) ===
    print("\n[P45a] Always-on lobotomy (last-token-only muting)...")
    configs = [
        ('baseline', False, False, 1.0, 1.0),
        ('mute_skill', True, False, 0.0, 1.0),
        ('boost_fact', False, True, 1.0, 2.0),
        ('mute+boost', True, True, 0.0, 2.0),
        ('mute+boost3x', True, True, 0.0, 3.0),
        ('mute+boost5x', True, True, 0.0, 5.0),
    ]

    config_results = {}
    for name, ms, bf, mf, bfac in configs:
        correct = 0
        ranks = []
        for prompt, fact_ids, _ in tests:
            logits, _ = lobotomy_forward_v2(model, tok, prompt, ms, bf, mf, bfac)
            rank = int((logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            ranks.append(rank)
            if torch.argmax(logits).item() in fact_ids:
                correct += 1
        config_results[name] = {
            'accuracy': correct / len(tests),
            'median_rank': float(np.median(ranks)),
            'mean_rank': float(np.mean(ranks)),
        }
        print(f"  {name:>15s}: {correct}/{len(tests)} = {correct/len(tests):.0%} "
              f"median_rank={np.median(ranks):.0f}")

    # === P45b: Oracle-conditional lobotomy ===
    print(f"\n[P45b] Oracle-conditional: lobotomy only when H > {ENTROPY_THRESHOLD}...")
    cond_results = []
    for prompt, fact_ids, expected in tests:
        # Baseline forward for entropy check
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            base_out = model(**inp, output_attentions=True, return_dict=True)
        base_logits = base_out.logits[:, -1, :].squeeze(0)
        ent = get_entropy(base_out.attentions)

        if ent > ENTROPY_THRESHOLD:
            logits, _ = lobotomy_forward_v2(model, tok, prompt, True, True, 0.0, 3.0)
            action = 'LOBOTOMY'
        else:
            logits = base_logits
            action = 'PASS'

        base_rank = int((base_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
        lob_rank = int((logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
        correct = torch.argmax(logits).item() in fact_ids
        base_correct = torch.argmax(base_logits).item() in fact_ids

        cond_results.append({
            'expected': expected, 'entropy': round(ent, 3),
            'action': action, 'correct': correct, 'base_correct': base_correct,
            'base_rank': base_rank, 'lob_rank': lob_rank,
        })
        b = 'OK' if base_correct else 'FAIL'
        l = 'OK' if correct else 'FAIL'
        print(f"  {expected:>8s}: H={ent:.3f} [{action:>9s}] "
              f"base=[{b}]r={base_rank:>4d} -> [{l}]r={lob_rank:>4d}")

    cond_acc = sum(1 for r in cond_results if r['correct']) / len(tests)
    base_acc = sum(1 for r in cond_results if r['base_correct']) / len(tests)

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cnames = list(config_results.keys())
    caccs = [config_results[n]['accuracy']*100 for n in cnames]
    axes[0].bar(cnames, caccs, color='teal', alpha=0.7)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Lobotomy Configs')
    axes[0].tick_params(axis='x', rotation=30, labelsize=7)

    labels = [r['expected'] for r in cond_results]
    x = range(len(labels))
    br = [r['base_rank'] for r in cond_results]
    lr = [r['lob_rank'] for r in cond_results]
    axes[1].bar([i-0.2 for i in x], br, 0.4, label='Base', color='red', alpha=0.7)
    axes[1].bar([i+0.2 for i in x], lr, 0.4, label='Lobotomy', color='green', alpha=0.7)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[1].set_ylabel('Fact Rank')
    axes[1].set_title(f'Conditional (base={base_acc:.0%} lob={cond_acc:.0%})')
    axes[1].legend(fontsize=8)

    # Median rank by config
    med_ranks = [config_results[n]['median_rank'] for n in cnames]
    axes[2].bar(cnames, med_ranks, color='purple', alpha=0.7)
    axes[2].set_ylabel('Median Fact Rank')
    axes[2].set_title('Rank Improvement')
    axes[2].tick_params(axis='x', rotation=30, labelsize=7)

    plt.suptitle('Phase 45: Dynamic Skill Lobotomy v2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase45_lobotomy_v2.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 45, 'name': 'Dynamic Skill Lobotomy v2',
        'config_results': config_results,
        'conditional_accuracy': cond_acc,
        'baseline_accuracy': base_acc,
        'conditional_results': cond_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase45_lobotomy_v2.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 45 RESULTS")
    print("=" * 70)
    print(f"  Baseline: {base_acc:.0%}")
    print(f"  Conditional lobotomy: {cond_acc:.0%}")
    for n in cnames:
        print(f"  {n}: {config_results[n]['accuracy']:.0%} (med_rank={config_results[n]['median_rank']:.0f})")
    print("=" * 70)
    phase_complete(45)

if __name__ == '__main__':
    main()
