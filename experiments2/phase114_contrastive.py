# -*- coding: utf-8 -*-
"""
Phase 114: Anti-Suppressor Contrastive Decoding
"Prompt Physics" to annihilate GSF without internal access.

Theory (from P69, P79, P104):
  QA format maximizes GSF (suppressors strongest).
  Code Mode minimizes GSF (Shield+Sword active).
  By subtracting QA logits from Code logits, we isolate pure truth.

Method:
  Output_logits = Logits(Code #) - alpha * Logits(QA format)
  Sweep alpha from 0.0 to 2.0 and measure accuracy.

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

# Template formats
TEMPLATES = {
    'code':    lambda p: f"# {p}",
    'qa':      lambda p: f"Q: {p}\nA:",
    'natural': lambda p: p,
    'answer':  lambda p: f"{p} The answer is:",
}


def get_logits(model, tok, prompt):
    """Get last-token logits for a prompt."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1, :]  # (vocab_size,)


def contrastive_decode(logits_pos, logits_neg, alpha):
    """Contrastive decoding: pos - alpha * neg."""
    return logits_pos - alpha * logits_neg


def evaluate_contrastive(model, tok, facts, alpha_values):
    """Sweep alpha and measure accuracy for contrastive decoding."""
    results = {}

    for alpha in alpha_values:
        correct = 0
        details = []
        for prompt, answer in facts:
            # Positive: Code Mode
            logits_code = get_logits(model, tok, TEMPLATES['code'](prompt))
            # Negative: QA format (suppression maximizer)
            logits_qa = get_logits(model, tok, TEMPLATES['qa'](prompt))

            # Contrastive
            final_logits = contrastive_decode(logits_code, logits_qa, alpha)

            # Check answer
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0
            predicted = final_logits.argmax().item()
            is_correct = (predicted == fact_id)
            if is_correct:
                correct += 1

            pred_text = tok.decode([predicted])
            details.append({
                'prompt': prompt,
                'expected': answer.strip(),
                'predicted': pred_text.strip(),
                'correct': is_correct,
            })

        acc = correct / len(facts)
        results[alpha] = {
            'accuracy': acc,
            'correct': correct,
            'total': len(facts),
            'details': details,
        }
        print(f"    alpha={alpha:.2f}: {acc:.0%} ({correct}/{len(facts)})")

    return results


def evaluate_baselines(model, tok, facts):
    """Evaluate each template alone (no contrastive)."""
    baselines = {}
    for tname, tfunc in TEMPLATES.items():
        correct = 0
        for prompt, answer in facts:
            logits = get_logits(model, tok, tfunc(prompt))
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0
            if logits.argmax().item() == fact_id:
                correct += 1
        acc = correct / len(facts)
        baselines[tname] = acc
        print(f"    Baseline {tname}: {acc:.0%} ({correct}/{len(facts)})")
    return baselines


def main():
    print("[P114] Anti-Suppressor Contrastive Decoding")
    print(f"  Device: {DEVICE}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B', local_files_only=True,
        torch_dtype=torch.float16
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', local_files_only=True)

    # 1. Baselines
    print("\n  === Baselines ===")
    baselines = evaluate_baselines(model, tok, FACTS)

    # 2. Contrastive sweep: Code - alpha * QA
    print("\n  === Contrastive: Code - alpha * QA ===")
    alphas = [round(a * 0.1, 2) for a in range(0, 21)]  # 0.0 to 2.0
    contrastive_results = evaluate_contrastive(model, tok, FACTS, alphas)

    # 3. Also try: Answer - alpha * QA
    print("\n  === Contrastive: Answer - alpha * QA ===")
    answer_vs_qa = {}
    for alpha in alphas:
        correct = 0
        for prompt, answer in FACTS:
            logits_ans = get_logits(model, tok, TEMPLATES['answer'](prompt))
            logits_qa = get_logits(model, tok, TEMPLATES['qa'](prompt))
            final = contrastive_decode(logits_ans, logits_qa, alpha)
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0
            if final.argmax().item() == fact_id:
                correct += 1
        acc = correct / len(FACTS)
        answer_vs_qa[alpha] = acc
        print(f"    alpha={alpha:.2f}: {acc:.0%}")

    # 4. Natural - alpha * QA
    print("\n  === Contrastive: Natural - alpha * QA ===")
    nat_vs_qa = {}
    for alpha in alphas:
        correct = 0
        for prompt, answer in FACTS:
            logits_nat = get_logits(model, tok, TEMPLATES['natural'](prompt))
            logits_qa = get_logits(model, tok, TEMPLATES['qa'](prompt))
            final = contrastive_decode(logits_nat, logits_qa, alpha)
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0
            if final.argmax().item() == fact_id:
                correct += 1
        acc = correct / len(FACTS)
        nat_vs_qa[alpha] = acc
        print(f"    alpha={alpha:.2f}: {acc:.0%}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Find best
    best_alpha_code = max(contrastive_results, key=lambda a: contrastive_results[a]['accuracy'])
    best_alpha_ans = max(answer_vs_qa, key=lambda a: answer_vs_qa[a])
    best_alpha_nat = max(nat_vs_qa, key=lambda a: nat_vs_qa[a])

    print(f"\n  === RESULTS ===")
    print(f"  Best Code-QA: alpha={best_alpha_code:.2f} -> {contrastive_results[best_alpha_code]['accuracy']:.0%}")
    print(f"  Best Answer-QA: alpha={best_alpha_ans:.2f} -> {answer_vs_qa[best_alpha_ans]:.0%}")
    print(f"  Best Natural-QA: alpha={best_alpha_nat:.2f} -> {nat_vs_qa[best_alpha_nat]:.0%}")
    for k, v in baselines.items():
        print(f"  Baseline {k}: {v:.0%}")

    # Save
    out = {
        'phase': 114,
        'name': 'Anti-Suppressor Contrastive Decoding',
        'model': 'Qwen2.5-1.5B',
        'baselines': baselines,
        'code_vs_qa': {str(k): v for k, v in contrastive_results.items()},
        'answer_vs_qa': {str(k): v for k, v in answer_vs_qa.items()},
        'natural_vs_qa': {str(k): v for k, v in nat_vs_qa.items()},
        'best_code_alpha': best_alpha_code,
        'best_answer_alpha': best_alpha_ans,
        'best_natural_alpha': best_alpha_nat,
    }
    with open(os.path.join(RESULTS_DIR, 'phase114_contrastive.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Alpha sweep curves
    code_accs = [contrastive_results[a]['accuracy'] for a in alphas]
    ans_accs = [answer_vs_qa[a] for a in alphas]
    nat_accs = [nat_vs_qa[a] for a in alphas]
    axes[0].plot(alphas, code_accs, 'o-', color='#2ecc71', label='Code# - a*QA', linewidth=2)
    axes[0].plot(alphas, ans_accs, 's-', color='#e74c3c', label='Answer - a*QA', linewidth=2)
    axes[0].plot(alphas, nat_accs, '^-', color='#3498db', label='Natural - a*QA', linewidth=2)
    axes[0].axhline(y=baselines['code'], color='#2ecc71', linestyle='--', alpha=0.5, label=f"Code baseline ({baselines['code']:.0%})")
    axes[0].axhline(y=baselines['answer'], color='#e74c3c', linestyle='--', alpha=0.5, label=f"Answer baseline ({baselines['answer']:.0%})")
    axes[0].axhline(y=baselines['natural'], color='#3498db', linestyle='--', alpha=0.5, label=f"Natural baseline ({baselines['natural']:.0%})")
    axes[0].set_xlabel('alpha (subtraction strength)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Contrastive Decoding: X - alpha * QA')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Best alpha detail (Code-QA)
    best_detail = contrastive_results[best_alpha_code]['details']
    colors_bar = ['#2ecc71' if d['correct'] else '#e74c3c' for d in best_detail]
    axes[1].barh(range(len(best_detail)), [1]*len(best_detail), color=colors_bar)
    for i, d in enumerate(best_detail):
        short_prompt = d['prompt'][:30] + '...' if len(d['prompt']) > 30 else d['prompt']
        axes[1].text(0.02, i, f"{short_prompt} -> {d['predicted']}", fontsize=6, va='center')
    axes[1].set_yticks([])
    axes[1].set_title(f'Best Code-QA (alpha={best_alpha_code:.2f}, acc={contrastive_results[best_alpha_code]["accuracy"]:.0%})')

    # Panel 3: Baseline comparison bar chart
    methods = ['Natural', 'Code#', 'Answer', f'Code-QA\na={best_alpha_code}', f'Ans-QA\na={best_alpha_ans}']
    accs_bar = [baselines['natural'], baselines['code'], baselines['answer'],
                contrastive_results[best_alpha_code]['accuracy'], answer_vs_qa[best_alpha_ans]]
    bar_colors = ['#3498db', '#2ecc71', '#e74c3c', '#27ae60', '#c0392b']
    bars = axes[2].bar(methods, accs_bar, color=bar_colors, edgecolor='black')
    for bar, acc in zip(bars, accs_bar):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.0%}', ha='center', fontweight='bold', fontsize=10)
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Contrastive vs Baselines')
    axes[2].set_ylim(0, 1.15)

    fig.suptitle('Phase 114: Anti-Suppressor Contrastive Decoding (Qwen2.5-1.5B)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase114_contrastive.png'), dpi=150)
    plt.close()
    print("[Phase 114] Complete.")


if __name__ == '__main__':
    main()
