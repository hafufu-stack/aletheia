# -*- coding: utf-8 -*-
"""
Phase 73: L10-Proposal Entropy Rejection
P36's revenge: entropy rejection failed because candidates came from L12.
Now: pull candidates from L10 (where facts live), evaluate each by
1-step-ahead entropy. The correct fact should COOL the model's entropy.
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
    print("[P73] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def compute_entropy(model, input_ids):
    """Compute mean attention entropy for input."""
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
    print("  Phase 73: L10-Proposal Entropy Rejection")
    print("  P36's revenge: candidates from L10, selection by entropy cooling")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The Earth orbits the", [4252], "Sun"),
        ("The boiling point of water is", [1802], "100"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("Oxygen has the atomic number", [807], "8"),
    ]

    results = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

        # Get L10 top-K candidates (the "truth pool")
        h10 = {}
        def hook(m, a, o):
            h10['h'] = o[0][0, -1, :].detach()
        handle = model.transformer.h[10].register_forward_hook(hook)
        with torch.no_grad():
            out = model(inp, output_attentions=True, return_dict=True)
        handle.remove()

        # Base entropy (before any token)
        base_entropy = compute_entropy(model, inp)

        # L10 top-K candidates
        normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
        l10_logits = model.lm_head(normed).squeeze(0)
        l10_top_k = 10
        l10_candidates = l10_logits.argsort(descending=True)[:l10_top_k].tolist()

        # L12 top-K candidates (for comparison)
        l12_logits = out.logits[0, -1, :]
        l12_candidates = l12_logits.argsort(descending=True)[:l10_top_k].tolist()

        # For each L10 candidate: append to prompt, measure 1-step-ahead entropy
        l10_candidate_entropies = []
        for cand_id in l10_candidates:
            extended = torch.cat([inp, torch.tensor([[cand_id]], device=DEVICE)], dim=1)
            ent = compute_entropy(model, extended)
            cand_tok = tok.decode([cand_id]).encode('ascii', 'replace').decode().strip()
            l10_candidate_entropies.append({
                'token_id': cand_id, 'token': cand_tok,
                'entropy': ent, 'is_fact': cand_id in fact_ids,
                'cooling': base_entropy - ent,
            })

        # Same for L12 candidates
        l12_candidate_entropies = []
        for cand_id in l12_candidates:
            extended = torch.cat([inp, torch.tensor([[cand_id]], device=DEVICE)], dim=1)
            ent = compute_entropy(model, extended)
            cand_tok = tok.decode([cand_id]).encode('ascii', 'replace').decode().strip()
            l12_candidate_entropies.append({
                'token_id': cand_id, 'token': cand_tok,
                'entropy': ent, 'is_fact': cand_id in fact_ids,
                'cooling': base_entropy - ent,
            })

        # Sort by cooling (highest cooling = best candidate)
        l10_sorted = sorted(l10_candidate_entropies, key=lambda x: x['cooling'], reverse=True)
        l12_sorted = sorted(l12_candidate_entropies, key=lambda x: x['cooling'], reverse=True)

        # Did the correct fact win by cooling?
        l10_winner = l10_sorted[0]
        l12_winner = l12_sorted[0]
        l10_fact_rank = next((i+1 for i, c in enumerate(l10_sorted) if c['is_fact']), -1)
        l12_fact_rank = next((i+1 for i, c in enumerate(l12_sorted) if c['is_fact']), -1)

        result = {
            'expected': expected, 'base_entropy': round(base_entropy, 4),
            'l10_winner': l10_winner['token'], 'l10_winner_correct': l10_winner['is_fact'],
            'l10_fact_cooling_rank': l10_fact_rank,
            'l12_winner': l12_winner['token'], 'l12_winner_correct': l12_winner['is_fact'],
            'l12_fact_cooling_rank': l12_fact_rank,
            'l10_candidates': l10_candidate_entropies[:5],
            'l12_candidates': l12_candidate_entropies[:5],
        }
        results.append(result)

        l10_tag = 'CORRECT' if l10_winner['is_fact'] else f'wrong(fact@{l10_fact_rank})'
        l12_tag = 'CORRECT' if l12_winner['is_fact'] else f'wrong(fact@{l12_fact_rank})'
        print(f"  {expected:>12s}: L10-winner='{l10_winner['token']}'[{l10_tag}] "
              f"L12-winner='{l12_winner['token']}'[{l12_tag}]")

    # Aggregate
    l10_correct = sum(1 for r in results if r['l10_winner_correct'])
    l12_correct = sum(1 for r in results if r['l12_winner_correct'])

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Method comparison
    methods = ['L12 Greedy\n(baseline)', 'L12 Entropy\nRejection', 'L10 Entropy\nRejection']
    # L12 greedy = standard model output
    l12_greedy = sum(1 for r in results if r['l12_candidates'][0]['is_fact'])
    accs = [l12_greedy/len(tests)*100, l12_correct/len(tests)*100, l10_correct/len(tests)*100]
    colors = ['red', 'orange', 'green']
    axes[0].bar(range(3), accs, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(methods, fontsize=8)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('L10 vs L12 Entropy Rejection')

    # 2. Cooling profile for one example (Tokyo)
    tokyo_r = results[0]
    l10_cands = tokyo_r['l10_candidates']
    labels = [c['token'][:8] for c in l10_cands]
    coolings = [c['cooling'] for c in l10_cands]
    colors2 = ['green' if c['is_fact'] else 'gray' for c in l10_cands]
    axes[1].bar(range(len(labels)), coolings, color=colors2, alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[1].set_ylabel('Entropy Cooling')
    axes[1].set_title(f"L10 Candidates for '{tokyo_r['expected']}'")

    # 3. Fact cooling rank distribution
    l10_ranks = [r['l10_fact_cooling_rank'] for r in results if r['l10_fact_cooling_rank'] > 0]
    l12_ranks = [r['l12_fact_cooling_rank'] for r in results if r['l12_fact_cooling_rank'] > 0]
    axes[2].hist(l10_ranks, bins=range(1, 12), alpha=0.5, color='green', label='L10', edgecolor='black')
    axes[2].hist(l12_ranks, bins=range(1, 12), alpha=0.5, color='red', label='L12', edgecolor='black')
    axes[2].set_xlabel('Fact Rank by Cooling')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Where Does the Fact Rank\nby Entropy Cooling?')
    axes[2].legend()

    plt.suptitle('Phase 73: L10-Proposal Entropy Rejection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase73_l10_entropy_reject.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 73, 'name': 'L10-Proposal Entropy Rejection',
        'l10_accuracy': l10_correct / len(tests),
        'l12_accuracy': l12_correct / len(tests),
        'l12_greedy_accuracy': l12_greedy / len(tests),
        'results': results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase73_l10_entropy_reject.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  L12 greedy:            {l12_greedy}/{len(tests)} = {l12_greedy/len(tests):.0%}")
    print(f"  L12 entropy rejection: {l12_correct}/{len(tests)} = {l12_correct/len(tests):.0%}")
    print(f"  L10 entropy rejection: {l10_correct}/{len(tests)} = {l10_correct/len(tests):.0%}")
    print("=" * 70)
    phase_complete(73)

if __name__ == '__main__':
    main()
