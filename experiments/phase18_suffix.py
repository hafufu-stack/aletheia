# -*- coding: utf-8 -*-
"""
Phase 18: Aletheia Suffix - Decompiling the Truth Activator
- Find adversarial suffix that activates fact heads and silences skill heads
- Gradient-free optimization (random search + hill climbing)
- Test if suffix transfers across prompts
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_model():
    print("[P18] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def score_suffix(model, tok, prompt, fact_ids, suffix_ids):
    """Score how well a suffix boosts fact token probability."""
    full_ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    if suffix_ids:
        suffix_tensor = torch.tensor([suffix_ids], dtype=torch.long, device=DEVICE)
        full_ids = torch.cat([full_ids, suffix_tensor], dim=1)

    with torch.no_grad():
        out = model(full_ids)
    logits = out.logits[:, -1, :].squeeze(0)
    probs = F.softmax(logits, dim=-1)

    # Score = sum of probabilities of fact tokens
    score = sum(float(probs[tid]) for tid in fact_ids if tid < probs.shape[0])
    rank = 0
    sorted_ids = logits.argsort(descending=True).tolist()
    for r, tid in enumerate(sorted_ids):
        if tid in fact_ids:
            rank = r + 1
            break
    return score, rank


def main():
    print("=" * 70)
    print("  Phase 18: The Aletheia Suffix")
    print("  Decompiling a Truth-Activating Token Sequence")
    print("=" * 70)

    model, tok = load_model()
    vocab_size = tok.vocab_size

    # Training prompts (optimize suffix on these)
    train_pairs = [
        ("The capital of Japan is", [11790]),
        ("The capital of France is", [6342]),
        ("Water freezes at", [657]),
        ("The largest planet is", [22721]),
    ]

    # Test prompts (check transfer)
    test_pairs = [
        ("DNA stands for", [390]),
        ("The chemical symbol for gold is", [7591]),
        ("The speed of light is approximately", [22626]),
        ("The first president of the United States was", [4502]),
    ]

    suffix_len = 5  # Number of tokens in suffix
    n_iterations = 200
    n_candidates = 50

    # === Baseline ===
    print("\n[P18a] Baseline (no suffix)...")
    baseline_ranks = {}
    for prompt, fact_ids in train_pairs + test_pairs:
        _, rank = score_suffix(model, tok, prompt, fact_ids, [])
        baseline_ranks[prompt[:30]] = rank
        print(f"  rank={rank:>5d}: {prompt[:40]}")

    # === Random search + hill climbing ===
    print(f"\n[P18b] Optimizing suffix ({suffix_len} tokens, {n_iterations} iterations)...")
    best_suffix = list(np.random.randint(0, vocab_size, size=suffix_len))
    best_total_score = 0

    # Evaluate initial
    for prompt, fact_ids in train_pairs:
        s, _ = score_suffix(model, tok, prompt, fact_ids, best_suffix)
        best_total_score += s

    score_history = [best_total_score]

    for it in range(n_iterations):
        # Generate candidates by mutating 1-2 positions
        for _ in range(n_candidates):
            candidate = best_suffix.copy()
            n_mutations = np.random.randint(1, 3)
            for _ in range(n_mutations):
                pos = np.random.randint(0, suffix_len)
                candidate[pos] = np.random.randint(0, vocab_size)

            # Score
            total_score = 0
            for prompt, fact_ids in train_pairs:
                s, _ = score_suffix(model, tok, prompt, fact_ids, candidate)
                total_score += s

            if total_score > best_total_score:
                best_total_score = total_score
                best_suffix = candidate

        score_history.append(best_total_score)

        if it % 50 == 0:
            suffix_text = tok.decode(best_suffix).encode('ascii', 'replace').decode()
            print(f"  iter={it:>4d}: score={best_total_score:.4f} "
                  f"suffix='{suffix_text}'")

    suffix_text = tok.decode(best_suffix).encode('ascii', 'replace').decode()
    print(f"\n  Final suffix: '{suffix_text}'")
    print(f"  Final score: {best_total_score:.4f}")

    # === Evaluate on train + test ===
    print("\n[P18c] Evaluating suffix on train and test prompts...")
    train_results = []
    test_results = []

    for prompt, fact_ids in train_pairs:
        s_base, r_base = score_suffix(model, tok, prompt, fact_ids, [])
        s_suf, r_suf = score_suffix(model, tok, prompt, fact_ids, best_suffix)
        train_results.append({
            'prompt': prompt[:40], 'rank_before': r_base, 'rank_after': r_suf,
            'improvement': r_base - r_suf,
        })
        print(f"  [TRAIN] rank: {r_base:>5d} -> {r_suf:>5d} | {prompt[:35]}")

    for prompt, fact_ids in test_pairs:
        s_base, r_base = score_suffix(model, tok, prompt, fact_ids, [])
        s_suf, r_suf = score_suffix(model, tok, prompt, fact_ids, best_suffix)
        test_results.append({
            'prompt': prompt[:40], 'rank_before': r_base, 'rank_after': r_suf,
            'improvement': r_base - r_suf,
        })
        print(f"  [TEST]  rank: {r_base:>5d} -> {r_suf:>5d} | {prompt[:35]}")

    mean_train_imp = np.mean([r['improvement'] for r in train_results])
    mean_test_imp = np.mean([r['improvement'] for r in test_results])
    transfer_rate = sum(1 for r in test_results if r['improvement'] > 0) / len(test_results)

    print(f"\n  Mean improvement: train={mean_train_imp:.0f}, test={mean_test_imp:.0f}")
    print(f"  Transfer rate: {transfer_rate:.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(score_history, 'b-', linewidth=1, alpha=0.7)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Total Score')
    axes[0].set_title('Suffix Optimization')
    axes[0].grid(True, alpha=0.3)

    all_res = train_results + test_results
    labels = [r['prompt'][:15] for r in all_res]
    before = [r['rank_before'] for r in all_res]
    after = [r['rank_after'] for r in all_res]
    x = range(len(all_res))
    axes[1].scatter(x, before, c='red', label='Before', s=60, zorder=5)
    axes[1].scatter(x, after, c='green', label='After', s=60, zorder=5)
    for i in x:
        axes[1].plot([i, i], [before[i], after[i]], 'gray', alpha=0.5)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    axes[1].set_ylabel('Fact Token Rank')
    axes[1].set_title('Rank Before/After Suffix')
    axes[1].legend()
    axes[1].set_yscale('log')

    axes[2].bar(['Train', 'Test'], [mean_train_imp, mean_test_imp],
                color=['blue', 'orange'], alpha=0.7)
    axes[2].set_ylabel('Mean Rank Improvement')
    axes[2].set_title(f'Transfer: {transfer_rate:.0%}')

    plt.suptitle('Phase 18: The Aletheia Suffix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase18_suffix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 18, 'name': 'Aletheia Suffix',
        'suffix_ids': best_suffix,
        'suffix_text': suffix_text,
        'final_score': best_total_score,
        'train_results': train_results,
        'test_results': test_results,
        'transfer_rate': transfer_rate,
        'mean_train_improvement': float(mean_train_imp),
        'mean_test_improvement': float(mean_test_imp),
    }
    with open(os.path.join(RESULTS_DIR, 'phase18_suffix.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 18 RESULTS: Aletheia Suffix")
    print("=" * 70)
    print(f"  Suffix: '{suffix_text}'")
    print(f"  Transfer: {transfer_rate:.0%}")
    print(f"  Mean improvement: train={mean_train_imp:.0f}, test={mean_test_imp:.0f}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
