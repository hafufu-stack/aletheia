# -*- coding: utf-8 -*-
"""
Phase 13: Spike Universality Theorem
- Test spike=7 threshold on 50 diverse QA pairs
- Is the phase transition universal or fact-dependent?
- Measure variance of critical spike across questions
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
    print("[P13] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def main():
    print("=" * 70)
    print("  Phase 13: Spike Universality Theorem")
    print("  Is the phase transition threshold universal?")
    print("=" * 70)

    model, tok = load_model()

    # 50 diverse QA pairs with expected first token
    qa_pairs = [
        ("The capital of Japan is", " Tokyo"),
        ("The capital of France is", " Paris"),
        ("The capital of Germany is", " Berlin"),
        ("The capital of Italy is", " Rome"),
        ("The capital of Spain is", " Madrid"),
        ("The capital of China is", " Beijing"),
        ("The capital of Russia is", " Moscow"),
        ("The capital of Brazil is", " Bras"),
        ("The capital of Australia is", " Canberra"),
        ("The capital of Canada is", " Ottawa"),
        ("Water freezes at", " 0"),
        ("Water boils at", " 100"),
        ("The speed of light is", " 299"),
        ("Pi is approximately", " 3"),
        ("The square root of 4 is", " 2"),
        ("The largest planet is", " Jupiter"),
        ("The smallest planet is", " Mercury"),
        ("The hottest planet is", " Venus"),
        ("The red planet is", " Mars"),
        ("The sun is a", " star"),
        ("DNA stands for", " de"),
        ("RNA stands for", " rib"),
        ("The chemical symbol for gold is", " Au"),
        ("The chemical symbol for silver is", " Ag"),
        ("The chemical symbol for iron is", " Fe"),
        ("Einstein developed the theory of", " rel"),
        ("Newton discovered the law of", " grav"),
        ("Darwin proposed the theory of", " evol"),
        ("Shakespeare wrote", " Ham"),
        ("Beethoven composed", " the"),
        ("The Earth has how many moons", " one"),
        ("Humans have how many chromosomes", " 46"),
        ("A triangle has how many sides", " three"),
        ("A hexagon has how many sides", " six"),
        ("The freezing point of water in Fahrenheit is", " 32"),
        ("The boiling point of water in Fahrenheit is", " 212"),
        ("The tallest mountain is", " Mount"),
        ("The longest river is", " the"),
        ("The largest ocean is", " the"),
        ("The deepest ocean trench is", " the"),
        ("Photosynthesis converts sunlight into", " energy"),
        ("The mitochondria is the", " power"),
        ("The periodic table was created by", " D"),
        ("The first president of the United States was", " George"),
        ("The year the Titanic sank was", " 1912"),
        ("The year man first walked on the moon was", " 1969"),
        ("The speed of sound is approximately", " 343"),
        ("Absolute zero is", " -"),
        ("The atomic number of hydrogen is", " 1"),
        ("The atomic number of carbon is", " 6"),
    ]

    # Resolve token IDs
    resolved = []
    for prompt, answer in qa_pairs:
        first_token_id = tok.encode(answer)[0]
        resolved.append((prompt, [first_token_id], answer.strip()))

    magnitudes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

    # === Find critical spike for EACH question ===
    print(f"\n[P13a] Finding critical spike for {len(resolved)} questions...")
    critical_spikes = []
    per_question = []

    for qi, (prompt, fact_ids, answer) in enumerate(resolved):
        inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        base_logits = out.logits[:, -1, :].squeeze(0)

        # Find minimum spike for correct answer
        crit = None
        for mag in magnitudes:
            spiked = base_logits.clone()
            for tid in fact_ids:
                spiked[tid] += mag
            if torch.argmax(spiked).item() in fact_ids:
                crit = mag
                break

        critical_spikes.append(crit)
        per_question.append({
            'prompt': prompt[:40], 'answer': answer,
            'critical_spike': crit,
            'baseline_rank': int((base_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
                if fact_ids[0] < base_logits.shape[0] else -1,
        })

        if qi % 10 == 0:
            print(f"  [{qi+1}/{len(resolved)}] {prompt[:35]}... -> spike={crit}")

    # === Statistics ===
    valid_crits = [c for c in critical_spikes if c is not None]
    never_solved = sum(1 for c in critical_spikes if c is None)

    print(f"\n[P13b] Statistics:")
    print(f"  Questions tested: {len(resolved)}")
    print(f"  Solved: {len(valid_crits)}/{len(resolved)}")
    print(f"  Never solved (spike>20): {never_solved}")
    if valid_crits:
        print(f"  Critical spike: mean={np.mean(valid_crits):.1f}, "
              f"median={np.median(valid_crits):.0f}, "
              f"std={np.std(valid_crits):.1f}")
        print(f"  Range: [{min(valid_crits)}, {max(valid_crits)}]")

    # === Aggregate accuracy curve ===
    print("\n[P13c] Aggregate accuracy curve...")
    agg_accuracy = {}
    for mag in magnitudes:
        correct = 0
        for qi, (prompt, fact_ids, _) in enumerate(resolved):
            inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[:, -1, :].squeeze(0)
            for tid in fact_ids:
                logits[tid] += mag
            if torch.argmax(logits).item() in fact_ids:
                correct += 1
        agg_accuracy[mag] = correct / len(resolved)
        print(f"  spike={mag:>3d}: {correct}/{len(resolved)} = {correct/len(resolved):.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Critical spike histogram
    axes[0].hist(valid_crits, bins=range(0, max(valid_crits)+2), color='steelblue',
                 alpha=0.7, edgecolor='black')
    if valid_crits:
        axes[0].axvline(x=np.mean(valid_crits), color='r', linestyle='--',
                        label=f'Mean={np.mean(valid_crits):.1f}')
        axes[0].axvline(x=np.median(valid_crits), color='g', linestyle='--',
                        label=f'Median={np.median(valid_crits):.0f}')
    axes[0].set_xlabel('Critical Spike Magnitude')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Distribution of Critical Spikes (n={len(valid_crits)})')
    axes[0].legend()

    # Plot 2: Aggregate accuracy curve
    mags_plot = sorted(agg_accuracy.keys())
    accs_plot = [agg_accuracy[m]*100 for m in mags_plot]
    axes[1].plot(mags_plot, accs_plot, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Spike Magnitude')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Aggregate Accuracy (n={len(resolved)})')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-5, 105)

    # Plot 3: Baseline rank vs critical spike
    ranks = [q['baseline_rank'] for q in per_question if q['critical_spike'] is not None]
    crits = [q['critical_spike'] for q in per_question if q['critical_spike'] is not None]
    axes[2].scatter(ranks, crits, c='purple', alpha=0.6, s=40)
    axes[2].set_xlabel('Baseline Rank of Correct Token')
    axes[2].set_ylabel('Critical Spike')
    axes[2].set_title('Rank vs Critical Spike')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 13: Spike Universality Theorem', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase13_universality.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 13, 'name': 'Spike Universality Theorem',
        'n_questions': len(resolved),
        'n_solved': len(valid_crits),
        'mean_critical': float(np.mean(valid_crits)) if valid_crits else None,
        'median_critical': float(np.median(valid_crits)) if valid_crits else None,
        'std_critical': float(np.std(valid_crits)) if valid_crits else None,
        'aggregate_accuracy': {str(k): v for k, v in agg_accuracy.items()},
        'per_question': per_question,
    }
    with open(os.path.join(RESULTS_DIR, 'phase13_universality.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 13 RESULTS: Spike Universality")
    print("=" * 70)
    if valid_crits:
        print(f"  Critical spike: {np.mean(valid_crits):.1f} +/- {np.std(valid_crits):.1f}")
        print(f"  Range: [{min(valid_crits)}, {max(valid_crits)}]")
    print(f"  Solved: {len(valid_crits)}/{len(resolved)}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
