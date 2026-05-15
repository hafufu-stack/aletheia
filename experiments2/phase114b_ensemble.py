# -*- coding: utf-8 -*-
"""
Phase 114b: Cross-Template Contrastive Ensemble
Extend P114 by exploring weighted multi-template ensembles.

Method:
  Final = w1*Logits(Code#) + w2*Logits(Answer) - w3*Logits(QA)
  Grid search over (w1, w2, w3) to find optimal ensemble weights.

Also test: Logits(Answer) - alpha * Logits(Natural) [different negative template]

Model: Qwen2.5-1.5B (GPU)
"""
import torch, json, os, gc, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FACTS = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("The chemical symbol for gold is", " Au"),
    ("The speed of light is approximately", " 299"),
    ("Albert Einstein was born in", " Ul"),
    ("The tallest mountain in the world is", " Mount"),
    ("The currency of the United Kingdom is the", " pound"),
    ("The author of Romeo and Juliet is", " William"),
    ("The first president of the United States was", " George"),
    ("The chemical formula for water is", " H"),
    ("The boiling point of water is", " 100"),
    ("The atomic number of carbon is", " 6"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("The speed of sound is approximately", " 343"),
    ("Photosynthesis converts sunlight into", " chemical"),
]

TEMPLATES = {
    'code':    lambda p: f"# {p}",
    'qa':      lambda p: f"Q: {p}\nA:",
    'natural': lambda p: p,
    'answer':  lambda p: f"{p} The answer is:",
}


def get_logits(model, tok, prompt):
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1, :]


def evaluate_ensemble(model, tok, facts, w_code, w_answer, w_qa):
    """Evaluate weighted ensemble: w1*Code + w2*Answer - w3*QA."""
    correct = 0
    for prompt, answer in facts:
        logits_code = get_logits(model, tok, TEMPLATES['code'](prompt))
        logits_ans = get_logits(model, tok, TEMPLATES['answer'](prompt))
        logits_qa = get_logits(model, tok, TEMPLATES['qa'](prompt))

        final = w_code * logits_code + w_answer * logits_ans - w_qa * logits_qa

        fact_tokens = tok.encode(answer)
        fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0
        if final.argmax().item() == fact_id:
            correct += 1
    return correct / len(facts)


def main():
    print("[P114b] Cross-Template Contrastive Ensemble")
    print(f"  Device: {DEVICE}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B', local_files_only=True,
        torch_dtype=torch.float16
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', local_files_only=True)

    # 1. Grid search: w1*Code + w2*Answer - w3*QA
    print("\n  === Ensemble Grid Search ===")
    weights = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]
    grid_results = {}
    best_acc = 0
    best_combo = (1, 0, 0)

    for w1 in weights:
        for w2 in weights:
            for w3 in weights:
                if w1 == 0 and w2 == 0:
                    continue  # skip all-zero positive
                acc = evaluate_ensemble(model, tok, FACTS, w1, w2, w3)
                key = f"{w1:.1f}_{w2:.1f}_{w3:.1f}"
                grid_results[key] = {
                    'w_code': w1, 'w_answer': w2, 'w_qa': w3,
                    'accuracy': acc
                }
                if acc > best_acc:
                    best_acc = acc
                    best_combo = (w1, w2, w3)
                if acc >= 0.7:
                    print(f"    w=({w1:.1f}, {w2:.1f}, {w3:.1f}): {acc:.0%}")

    print(f"\n  Best ensemble: w=({best_combo[0]:.1f}, {best_combo[1]:.1f}, {best_combo[2]:.1f}) -> {best_acc:.0%}")

    # 2. Fine-grained sweep around best
    print("\n  === Fine-grained sweep around best ===")
    fine_results = {}
    w1_best, w2_best, w3_best = best_combo
    for dw1 in [-0.2, -0.1, 0, 0.1, 0.2]:
        for dw2 in [-0.2, -0.1, 0, 0.1, 0.2]:
            for dw3 in [-0.2, -0.1, 0, 0.1, 0.2]:
                w1 = max(0, w1_best + dw1)
                w2 = max(0, w2_best + dw2)
                w3 = max(0, w3_best + dw3)
                if w1 == 0 and w2 == 0:
                    continue
                acc = evaluate_ensemble(model, tok, FACTS, w1, w2, w3)
                key = f"{w1:.1f}_{w2:.1f}_{w3:.1f}"
                fine_results[key] = {
                    'w_code': w1, 'w_answer': w2, 'w_qa': w3,
                    'accuracy': acc
                }
                if acc > best_acc:
                    best_acc = acc
                    best_combo = (w1, w2, w3)

    print(f"  Refined best: w=({best_combo[0]:.1f}, {best_combo[1]:.1f}, {best_combo[2]:.1f}) -> {best_acc:.0%}")

    # 3. Alternative negative: Answer - alpha * Natural
    print("\n  === Answer - alpha * Natural ===")
    ans_vs_nat = {}
    for alpha in [round(a * 0.1, 2) for a in range(0, 21)]:
        correct = 0
        for prompt, answer in FACTS:
            logits_ans = get_logits(model, tok, TEMPLATES['answer'](prompt))
            logits_nat = get_logits(model, tok, TEMPLATES['natural'](prompt))
            final = logits_ans - alpha * logits_nat
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0
            if final.argmax().item() == fact_id:
                correct += 1
        acc = correct / len(FACTS)
        ans_vs_nat[alpha] = acc
        print(f"    alpha={alpha:.2f}: {acc:.0%}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save
    out = {
        'phase': '114b',
        'name': 'Cross-Template Contrastive Ensemble',
        'model': 'Qwen2.5-1.5B',
        'best_ensemble': {
            'w_code': best_combo[0], 'w_answer': best_combo[1], 'w_qa': best_combo[2],
            'accuracy': best_acc,
        },
        'grid_results': grid_results,
        'fine_results': fine_results,
        'answer_vs_natural': {str(k): v for k, v in ans_vs_nat.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase114b_ensemble.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Heat map of w_qa vs accuracy (fixing best w1, w2)
    all_entries = list(grid_results.values()) + list(fine_results.values())
    qa_vals = sorted(set(e['w_qa'] for e in all_entries))
    code_vals = sorted(set(e['w_code'] for e in all_entries))
    # Scatter: w_qa on x, accuracy on y, colored by w_code
    for entry in all_entries:
        c = plt.cm.viridis(entry['w_code'] / max(1, max(code_vals)))
        axes[0].scatter(entry['w_qa'], entry['accuracy'], c=[c], s=20, alpha=0.6)
    axes[0].set_xlabel('w_qa (suppression subtraction)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Ensemble Search Space')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Answer - alpha * Natural
    ans_alphas = sorted(ans_vs_nat.keys())
    ans_accs = [ans_vs_nat[a] for a in ans_alphas]
    axes[1].plot(ans_alphas, ans_accs, 'o-', color='#e74c3c', linewidth=2)
    axes[1].set_xlabel('alpha')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Answer - alpha * Natural')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Top 10 ensembles
    all_sorted = sorted(all_entries, key=lambda e: e['accuracy'], reverse=True)[:10]
    labels = [f"({e['w_code']:.1f},{e['w_answer']:.1f},{e['w_qa']:.1f})" for e in all_sorted]
    accs_top = [e['accuracy'] for e in all_sorted]
    bars = axes[2].barh(range(len(labels)), accs_top, color='#2ecc71', edgecolor='black')
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].set_xlabel('Accuracy')
    axes[2].set_title('Top 10 Ensembles (w_code, w_ans, w_qa)')
    axes[2].invert_yaxis()
    for bar, acc in zip(bars, accs_top):
        axes[2].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc:.0%}', va='center', fontsize=9)

    fig.suptitle('Phase 114b: Cross-Template Contrastive Ensemble (Qwen2.5-1.5B)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase114b_ensemble.png'), dpi=150)
    plt.close()
    print("[Phase 114b] Complete.")


if __name__ == '__main__':
    main()
