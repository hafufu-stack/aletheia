# -*- coding: utf-8 -*-
"""
Phase 55: Rank Trajectory Fingerprinting
P53 proved facts are SUPPRESSED (rank improves L0->L10, drops L10->L12).
This "rise-then-fall" trajectory is a FINGERPRINT of factual tokens.
Grammar tokens have "fall-then-rise" (weak early, strong late).
Can we use this trajectory shape to classify fact vs grammar tokens?
"""
import os, json, sys
import numpy as np
import torch
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
    print("[P55] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_token_trajectories(model, tok, prompt, token_ids):
    """Get rank trajectory of specific tokens across all 12 layers."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    layer_hs = {}
    handles = []
    for li in range(12):
        def make_hook(idx):
            def hook_fn(module, args, output):
                layer_hs[idx] = output[0][0, -1, :].detach()
            return hook_fn
        h = model.transformer.h[li].register_forward_hook(make_hook(li))
        handles.append(h)
    with torch.no_grad():
        out = model(**inp)
    for h in handles:
        h.remove()

    trajectories = {}
    for tid in token_ids:
        ranks = []
        for li in range(12):
            normed = model.transformer.ln_f(layer_hs[li].unsqueeze(0))
            logits = model.lm_head(normed).squeeze(0)
            rank = int((logits.argsort(descending=True) == tid).nonzero().item()) + 1
            ranks.append(rank)
        # Final layer rank
        final_rank = int((out.logits[:, -1, :].squeeze(0).argsort(descending=True) == tid).nonzero().item()) + 1
        ranks.append(final_rank)
        trajectories[tid] = ranks
    return trajectories

def trajectory_shape(ranks):
    """Classify trajectory: rise-then-fall (fact), or fall-then-rise (grammar)."""
    min_idx = np.argmin(ranks)
    peak_rank = ranks[min_idx]  # Best (lowest) rank
    final_rank = ranks[-1]
    early_rank = ranks[0]

    # Suppression = how much rank WORSENED from peak to final
    suppression = final_rank - peak_rank

    # Rise = how much rank improved from early to peak
    improvement = early_rank - peak_rank

    return {
        'peak_layer': min_idx,
        'peak_rank': peak_rank,
        'early_rank': early_rank,
        'final_rank': final_rank,
        'suppression': suppression,
        'improvement': improvement,
        'is_suppressed': suppression > 0 and min_idx < 12,
    }

def main():
    print("=" * 70)
    print("  Phase 55: Rank Trajectory Fingerprinting")
    print("  Fact tokens: rise-then-fall. Grammar: fall-then-rise.")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", 11790, "Tokyo"),
        ("The capital of France is", 6342, "Paris"),
        ("Water freezes at", 657, "0"),
        ("The largest planet is", 22721, "Jupiter"),
        ("The chemical symbol for gold is", 7591, "Au"),
        ("Shakespeare wrote", 13483, "Hamlet"),
        ("The Earth orbits the", 4252, "Sun"),
        ("The boiling point of water is", 1802, "100"),
        ("Albert Einstein developed the theory of", 44449, "relativity"),
        ("Oxygen has the atomic number", 807, "8"),
    ]

    # === P55a: Fact token trajectories ===
    print("\n[P55a] Fact token rank trajectories...")
    fact_trajectories = []
    for prompt, fact_id, expected in tests:
        trajs = get_token_trajectories(model, tok, prompt, [fact_id])
        ranks = trajs[fact_id]
        shape = trajectory_shape(ranks)
        fact_trajectories.append({
            'expected': expected, 'ranks': ranks, **shape,
        })
        sup_tag = 'SUPPRESSED' if shape['is_suppressed'] else 'not_suppressed'
        print(f"  {expected:>12s}: peak=L{shape['peak_layer']}(r{shape['peak_rank']:>5d}) "
              f"final=r{shape['final_rank']:>5d} supp={shape['suppression']:>+6d} [{sup_tag}]")

    n_suppressed = sum(1 for f in fact_trajectories if f['is_suppressed'])
    print(f"  Suppressed: {n_suppressed}/{len(tests)}")

    # === P55b: Compare fact vs L12-top-1 (grammar winner) trajectories ===
    print("\n[P55b] Fact vs Grammar winner trajectories...")
    comparison = []
    for prompt, fact_id, expected in tests:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        grammar_winner = torch.argmax(out.logits[:, -1, :].squeeze(0)).item()

        trajs = get_token_trajectories(model, tok, prompt, [fact_id, grammar_winner])
        fact_ranks = trajs[fact_id]
        grammar_ranks = trajs[grammar_winner]

        fact_shape = trajectory_shape(fact_ranks)
        grammar_shape = trajectory_shape(grammar_ranks)

        gname = tok.decode([grammar_winner]).encode('ascii','replace').decode().strip()

        comparison.append({
            'expected': expected, 'grammar_winner': gname,
            'fact_trajectory': fact_ranks, 'grammar_trajectory': grammar_ranks,
            'fact_shape': fact_shape, 'grammar_shape': grammar_shape,
        })
        print(f"  {expected:>12s}: fact peak=L{fact_shape['peak_layer']} "
              f"grammar('{gname}') peak=L{grammar_shape['peak_layer']}")

    # === P55c: Trajectory-based token selector ===
    print("\n[P55c] Trajectory-based selection: pick token with earliest peak...")
    traj_results = []
    for prompt, fact_id, expected in tests:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)

        # Get L10 top-10 candidates
        layer_hs = {}
        def hook(module, args, output):
            layer_hs[10] = output[0][0, -1, :].detach()
        handle = model.transformer.h[10].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        normed = model.transformer.ln_f(layer_hs[10].unsqueeze(0))
        l10_logits = model.lm_head(normed).squeeze(0)
        candidates = torch.topk(l10_logits, 10).indices.tolist()

        # Get trajectories for all candidates
        trajs = get_token_trajectories(model, tok, prompt, candidates)

        # Score: prefer candidates with earliest peak and highest suppression
        # (suppressed = was strong in middle layers but weakened at end)
        best_cand = None
        best_score = -float('inf')
        for cand in candidates:
            shape = trajectory_shape(trajs[cand])
            # Penalize late peaks (grammar tokens peak at L11-L12)
            # Reward suppression (facts get suppressed)
            score = shape['suppression'] - shape['peak_rank'] * 0.1
            if shape['peak_layer'] <= 10:
                score += 10  # Bonus for early peak
            if score > best_score:
                best_score = score
                best_cand = cand

        is_correct = best_cand in [fact_id]
        cname = tok.decode([best_cand]).encode('ascii','replace').decode().strip()
        traj_results.append({
            'expected': expected, 'selected': cname, 'correct': is_correct,
        })
        tag = 'OK' if is_correct else 'FAIL'
        print(f"  {expected:>12s}: selected='{cname}' [{tag}]")

    traj_acc = sum(1 for t in traj_results if t['correct']) / len(tests)

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Fact trajectories (log scale)
    for ft in fact_trajectories:
        c = 'green' if ft['is_suppressed'] else 'gray'
        axes[0].plot(range(13), ft['ranks'], '.-', alpha=0.6, color=c)
    axes[0].set_xlabel('Layer (0-11 + Final)')
    axes[0].set_ylabel('Fact Rank')
    axes[0].set_yscale('log')
    axes[0].set_title('Fact Token Rank Trajectories')
    axes[0].grid(True, alpha=0.3)

    # Fact vs grammar trajectories (one example)
    if comparison:
        ex = comparison[0]
        axes[1].plot(range(13), ex['fact_trajectory'], 'g.-',
                    linewidth=2, label=f"Fact: {ex['expected']}")
        axes[1].plot(range(13), ex['grammar_trajectory'], 'r.-',
                    linewidth=2, label=f"Grammar: {ex['grammar_winner']}")
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Rank')
        axes[1].set_yscale('log')
        axes[1].set_title('Fact vs Grammar Trajectory')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    # Accuracy comparison
    methods = ['L12', 'L10', 'Trajectory']
    accs = [8.3, 33.3, traj_acc*100]
    axes[2].bar(methods, accs, color=['red', 'green', 'purple'], alpha=0.7)
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Selection Method Comparison')

    plt.suptitle('Phase 55: Rank Trajectory Fingerprinting', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase55_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 55, 'name': 'Rank Trajectory Fingerprinting',
        'n_suppressed': n_suppressed, 'total': len(tests),
        'trajectory_selection_accuracy': traj_acc,
        'fact_trajectories': fact_trajectories,
        'comparison': comparison,
        'trajectory_results': traj_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase55_trajectory.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 55 RESULTS")
    print("=" * 70)
    print(f"  Facts suppressed: {n_suppressed}/{len(tests)}")
    print(f"  Trajectory selection: {traj_acc:.0%}")
    print("=" * 70)
    phase_complete(55)

if __name__ == '__main__':
    main()
