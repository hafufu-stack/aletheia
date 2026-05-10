# -*- coding: utf-8 -*-
"""
Phase 40: Entropy-Guided Monte Carlo Tree Search
Use future entropy as compass: simulate multiple paths,
pick the one where the model is most confident (lowest entropy).
"Truth is the path of least epistemic resistance."
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
    print("[P40] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def measure_entropy(model, input_ids):
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, return_dict=True)
    ents = []
    for attn in out.attentions:
        for h in range(attn.shape[1]):
            a = attn[0, h, -1, :].cpu().numpy()
            ents.append(float(-np.sum(a * np.log(a + 1e-12))))
    return float(np.mean(ents)), out.logits[:, -1, :].squeeze(0)

def entropy_mcts(model, tok, prompt, n_candidates=5, lookahead=3):
    """Select the first token that leads to lowest future entropy."""
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

    # Get top-k candidates for first token
    ent0, logits0 = measure_entropy(model, inp)
    topk = logits0.argsort(descending=True)[:n_candidates]

    best_candidate = None
    best_future_ent = float('inf')
    candidate_traces = []

    for ci, candidate in enumerate(topk):
        cid = candidate.item()
        # Simulate: append candidate, then greedily generate lookahead tokens
        sim_seq = torch.cat([inp, torch.tensor([[cid]], device=DEVICE)], dim=1)
        future_ents = []

        for step in range(lookahead):
            ent, logits = measure_entropy(model, sim_seq)
            future_ents.append(ent)
            next_tok = torch.argmax(logits).item()
            if next_tok == tok.eos_token_id:
                break
            sim_seq = torch.cat([sim_seq, torch.tensor([[next_tok]], device=DEVICE)], dim=1)

        mean_future_ent = float(np.mean(future_ents)) if future_ents else float('inf')
        candidate_traces.append({
            'token_id': cid,
            'token': tok.decode([cid]).encode('ascii','replace').decode().strip(),
            'future_ents': future_ents,
            'mean_future_ent': round(mean_future_ent, 4),
        })

        if mean_future_ent < best_future_ent:
            best_future_ent = mean_future_ent
            best_candidate = cid

    return best_candidate, candidate_traces, ent0

def main():
    print("=" * 70)
    print("  Phase 40: Entropy-Guided MCTS")
    print("  Pick the path of least epistemic resistance")
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

    # === P40a: MCTS with different lookahead depths ===
    print(f"\n[P40a] MCTS lookahead sweep...")
    depth_results = {}

    for lookahead in [1, 2, 3, 5]:
        correct = 0
        per_prompt = []
        for prompt, fact_ids, expected in tests:
            chosen, traces, base_ent = entropy_mcts(
                model, tok, prompt, n_candidates=5, lookahead=lookahead)

            is_correct = chosen in fact_ids
            correct += int(is_correct)

            # Baseline (greedy)
            _, logits0 = measure_entropy(model, tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE))
            greedy = torch.argmax(logits0).item()
            greedy_correct = greedy in fact_ids

            chosen_tok = tok.decode([chosen]).encode('ascii','replace').decode().strip()
            greedy_tok = tok.decode([greedy]).encode('ascii','replace').decode().strip()

            per_prompt.append({
                'expected': expected,
                'greedy_token': greedy_tok, 'greedy_correct': greedy_correct,
                'mcts_token': chosen_tok, 'mcts_correct': is_correct,
                'base_entropy': round(base_ent, 3),
                'traces': traces,
            })

            if lookahead == 3:
                g = 'OK' if greedy_correct else 'FAIL'
                m = 'OK' if is_correct else 'FAIL'
                print(f"  {expected:>8s}: greedy=[{g}]{greedy_tok:>10s} "
                      f"mcts=[{m}]{chosen_tok:>10s} (H0={base_ent:.3f})")

        depth_results[lookahead] = {
            'accuracy': correct / len(tests),
            'per_prompt': per_prompt,
        }

    print(f"\n  Accuracy by lookahead depth:")
    for d in sorted(depth_results.keys()):
        print(f"    depth={d}: {depth_results[d]['accuracy']:.0%}")

    # === P40b: Candidate analysis ===
    print(f"\n[P40b] Candidate entropy analysis (depth=3)...")
    main_results = depth_results[3]['per_prompt']
    for r in main_results:
        print(f"  {r['expected']:>8s}: candidates:")
        for t in r['traces']:
            marker = ' <--' if t['token'] == r['mcts_token'] else ''
            correct = ' (CORRECT)' if t['token_id'] in [fid for fid in [11790,6342,657,22721,390,7591,13483,22626] if True] else ''
            print(f"    {t['token']:>12s}: mean_H={t['mean_future_ent']:.3f} "
                  f"path={[round(e,2) for e in t['future_ents']]}{marker}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    depths = sorted(depth_results.keys())
    accs = [depth_results[d]['accuracy']*100 for d in depths]
    axes[0].plot(depths, accs, 'g.-', linewidth=2, markersize=10)
    # Add greedy baseline
    greedy_acc = sum(1 for r in main_results if r['greedy_correct']) / len(tests) * 100
    axes[0].axhline(y=greedy_acc, color='red', linestyle='--', label=f'Greedy ({greedy_acc:.0f}%)')
    axes[0].set_xlabel('Lookahead Depth')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('MCTS Accuracy vs Depth')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Entropy traces for best case
    for r in main_results[:4]:
        for t in r['traces']:
            color = 'green' if t['token'] == r['mcts_token'] else 'lightgray'
            alpha = 0.8 if t['token'] == r['mcts_token'] else 0.3
            axes[1].plot(t['future_ents'], color=color, alpha=alpha)
    axes[1].set_xlabel('Lookahead Step')
    axes[1].set_ylabel('Attention Entropy')
    axes[1].set_title('Candidate Path Entropies')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Greedy vs MCTS
    labels = [r['expected'] for r in main_results]
    x = range(len(labels))
    g_vals = [int(r['greedy_correct']) for r in main_results]
    m_vals = [int(r['mcts_correct']) for r in main_results]
    axes[2].bar([i-0.2 for i in x], g_vals, 0.4, label='Greedy', color='red', alpha=0.7)
    axes[2].bar([i+0.2 for i in x], m_vals, 0.4, label='MCTS', color='green', alpha=0.7)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[2].set_ylabel('Correct')
    axes[2].set_title('Greedy vs MCTS')
    axes[2].legend(fontsize=8)

    plt.suptitle('Phase 40: Entropy-Guided MCTS', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase40_mcts.png'), dpi=150, bbox_inches='tight')
    plt.close()

    greedy_acc_final = sum(1 for r in main_results if r['greedy_correct']) / len(tests)
    mcts_acc_final = depth_results[3]['accuracy']
    results = {
        'phase': 40, 'name': 'Entropy-Guided MCTS',
        'greedy_accuracy': greedy_acc_final,
        'mcts_accuracy_by_depth': {str(k): v['accuracy'] for k, v in depth_results.items()},
        'per_case': main_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase40_mcts.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 40 RESULTS: Entropy-Guided MCTS")
    print("=" * 70)
    print(f"  Greedy: {greedy_acc_final:.0%}")
    for d in depths:
        print(f"  MCTS depth={d}: {depth_results[d]['accuracy']:.0%}")
    print("=" * 70)
    phase_complete(40)

if __name__ == '__main__':
    main()
