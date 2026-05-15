# -*- coding: utf-8 -*-
"""
Phase 116b: Calibrated Refusal Curve
Sweep entropy thresholds and plot the Pareto frontier of
accuracy vs refusal rate across multiple models.

Extends P116 with:
  - GPT-2 family (Small, XL) for cross-architecture comparison
  - Pareto frontier visualization
  - ROC-like analysis (true refusal vs false refusal)

Models: GPT-2 Small, GPT-2 XL, Qwen2.5-0.5B, Qwen2.5-1.5B (GPU)
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

KNOWN_FACTS = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("The chemical symbol for gold is", " Au"),
    ("The author of Romeo and Juliet is", " William"),
    ("The first president of the United States was", " George"),
    ("The chemical formula for water is", " H"),
    ("The boiling point of water is", " 100"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("The atomic number of carbon is", " 6"),
    ("The currency of the United Kingdom is the", " pound"),
]

UNKNOWN_FACTS = [
    ("The 47th prime number is", " 211"),
    ("The population of Funabashi in 2025 was approximately", " 640"),
    ("The melting point of hafnium in Kelvin is", " 2506"),
    ("The shortest-serving US president by days is", " William"),
    ("The capital of Nauru is", " Yar"),
    ("The deepest point of Lake Baikal in meters is", " 1642"),
    ("The atomic mass of Lutetium is approximately", " 175"),
    ("The ISO country code for Bhutan is", " BT"),
    ("The year the Treaty of Westphalia was signed is", " 1648"),
    ("The founder of the Mughal Empire was", " Bab"),
    ("The chemical formula for aluminum oxide is", " Al"),
    ("The speed of sound in steel in m/s is approximately", " 5960"),
    ("The orbital period of Neptune in Earth years is", " 165"),
    ("The wavelength of green light in nanometers is approximately", " 550"),
    ("The melting point of tungsten in Celsius is", " 3422"),
]

MODEL_CONFIGS = [
    ('gpt2',               'GPT2-Small',    12, 'gpt2'),
    ('gpt2-xl',            'GPT2-XL',       48, 'gpt2'),
    ('Qwen/Qwen2.5-0.5B', 'Qwen2.5-0.5B',  24, 'qwen'),
    ('Qwen/Qwen2.5-1.5B', 'Qwen2.5-1.5B',  28, 'qwen'),
]


def get_entropy_at_layer(model, tok, prompt, layer_idx, model_type):
    """Get entropy and correctness at a specific layer."""
    hidden = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden['h'] = output[0][:, -1, :].detach()
        else:
            hidden['h'] = output[:, -1, :].detach()

    if model_type == 'gpt2':
        handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
        ln_f = model.transformer.ln_f
        lm_head = model.lm_head
    else:
        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        ln_f = model.model.norm
        lm_head = model.lm_head

    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model(**inp)

    h = ln_f(hidden['h'].float())  # float32 for stable entropy
    logits = lm_head(h.to(next(lm_head.parameters()).dtype)).squeeze()
    handle.remove()

    # Cast to float32 to avoid fp16 NaN
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    return entropy, logits


def main():
    print("[P116b] Calibrated Refusal Curve")
    print(f"  Device: {DEVICE}")

    all_results = {}

    for model_id, label, n_layers, model_type in MODEL_CONFIGS:
        print(f"\n  === {label} ===")
        try:
            if model_type == 'gpt2':
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                model = GPT2LMHeadModel.from_pretrained(
                    model_id, local_files_only=True).eval().to(DEVICE)
                tok = GPT2Tokenizer.from_pretrained(model_id, local_files_only=True)
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, local_files_only=True,
                    torch_dtype=torch.float16
                ).eval().to(DEVICE)
                tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        best_layer = int(n_layers * 0.94)
        print(f"    L_best (0.94): L{best_layer}")

        # Measure entropies
        known_data = []
        for prompt, answer in KNOWN_FACTS:
            ent, logits = get_entropy_at_layer(
                model, tok, prompt, best_layer, model_type)
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if model_type != 'gpt2' else fact_tokens[0]
            correct = (logits.argmax().item() == fact_id)
            known_data.append({'entropy': ent, 'correct': correct, 'type': 'known'})

        unknown_data = []
        for prompt, answer in UNKNOWN_FACTS:
            ent, logits = get_entropy_at_layer(
                model, tok, prompt, best_layer, model_type)
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if model_type != 'gpt2' else fact_tokens[0]
            correct = (logits.argmax().item() == fact_id)
            unknown_data.append({'entropy': ent, 'correct': correct, 'type': 'unknown'})

        # Threshold sweep for ROC curve
        all_ents = [d['entropy'] for d in known_data + unknown_data]
        thresholds = np.linspace(min(all_ents) - 1, max(all_ents) + 1, 50)

        roc_points = []
        for thr in thresholds:
            # True Positive: unknown fact correctly refused (entropy > thr)
            tp = sum(1 for d in unknown_data if d['entropy'] > thr)
            # False Positive: known fact incorrectly refused
            fp = sum(1 for d in known_data if d['entropy'] > thr)
            # True Negative: known fact correctly answered
            tn = sum(1 for d in known_data if d['entropy'] <= thr)
            # False Negative: unknown fact incorrectly answered
            fn = sum(1 for d in unknown_data if d['entropy'] <= thr)

            tpr = tp / max(1, tp + fn)  # Sensitivity
            fpr = fp / max(1, fp + tn)  # 1 - Specificity

            # Accuracy considering refusal
            known_answered = [d for d in known_data if d['entropy'] <= thr]
            known_correct = sum(1 for d in known_answered if d['correct'])
            unknown_refused = sum(1 for d in unknown_data if d['entropy'] > thr)

            n_total = len(KNOWN_FACTS) + len(UNKNOWN_FACTS)
            overall_score = (known_correct + unknown_refused) / n_total

            roc_points.append({
                'threshold': float(thr),
                'tpr': float(tpr), 'fpr': float(fpr),
                'overall_score': float(overall_score),
                'known_refuse_rate': float(fp / max(1, len(KNOWN_FACTS))),
                'unknown_refuse_rate': float(tp / max(1, len(UNKNOWN_FACTS))),
            })

        # AUC approximation
        fpr_vals = [p['fpr'] for p in sorted(roc_points, key=lambda p: p['fpr'])]
        tpr_vals = [p['tpr'] for p in sorted(roc_points, key=lambda p: p['fpr'])]
        auc = np.trapz(tpr_vals, fpr_vals) if len(fpr_vals) > 1 else 0

        best_point = max(roc_points, key=lambda p: p['overall_score'])

        print(f"    Known entropy: mean={np.mean([d['entropy'] for d in known_data]):.2f}")
        print(f"    Unknown entropy: mean={np.mean([d['entropy'] for d in unknown_data]):.2f}")
        print(f"    AUC: {auc:.3f}")
        print(f"    Best threshold: {best_point['threshold']:.2f} (score={best_point['overall_score']:.0%})")

        all_results[label] = {
            'model': model_id,
            'n_layers': n_layers,
            'best_layer': best_layer,
            'known_entropies': [d['entropy'] for d in known_data],
            'unknown_entropies': [d['entropy'] for d in unknown_data],
            'known_baseline_acc': np.mean([d['correct'] for d in known_data]),
            'roc_points': roc_points,
            'auc': float(auc),
            'best_threshold': best_point,
        }

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    out = {'phase': '116b', 'name': 'Calibrated Refusal Curve', 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase116b_refusal.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    n_models = len(all_results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: ROC curves
    ax = axes[0, 0]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    for i, (label, data) in enumerate(all_results.items()):
        roc = data['roc_points']
        fpr = [p['fpr'] for p in sorted(roc, key=lambda p: p['fpr'])]
        tpr = [p['tpr'] for p in sorted(roc, key=lambda p: p['fpr'])]
        ax.plot(fpr, tpr, '-', color=colors[i % len(colors)], linewidth=2,
                label=f"{label} (AUC={data['auc']:.2f})")
    ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    ax.set_xlabel('False Positive Rate (Known wrongly refused)')
    ax.set_ylabel('True Positive Rate (Unknown correctly refused)')
    ax.set_title('ROC: Epistemic Entropy as Uncertainty Detector')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Entropy distributions (all models)
    ax = axes[0, 1]
    for i, (label, data) in enumerate(all_results.items()):
        ax.scatter([data['known_entropies']], [[i]*len(data['known_entropies'])],
                  c=colors[i % len(colors)], alpha=0.5, s=30, marker='o')
        ax.scatter([data['unknown_entropies']], [[i+0.3]*len(data['unknown_entropies'])],
                  c=colors[i % len(colors)], alpha=0.5, s=30, marker='x')
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(list(all_results.keys()), fontsize=8)
    ax.set_xlabel('L_best Entropy')
    ax.set_title('Known (o) vs Unknown (x) Entropy')
    ax.grid(True, alpha=0.3)

    # Panel 3: Score vs threshold
    ax = axes[1, 0]
    for i, (label, data) in enumerate(all_results.items()):
        roc = sorted(data['roc_points'], key=lambda p: p['threshold'])
        thrs = [p['threshold'] for p in roc]
        scores = [p['overall_score'] for p in roc]
        ax.plot(thrs, scores, '-', color=colors[i % len(colors)], linewidth=2, label=label)
        best = data['best_threshold']
        ax.scatter([best['threshold']], [best['overall_score']],
                  c=colors[i % len(colors)], s=100, zorder=5, edgecolors='black')
    ax.set_xlabel('Entropy Threshold')
    ax.set_ylabel('Overall Score')
    ax.set_title('Score vs Threshold (Pareto Frontier)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary bar chart
    ax = axes[1, 1]
    labels_bar = list(all_results.keys())
    baseline_accs = [d['known_baseline_acc'] for d in all_results.values()]
    best_scores = [d['best_threshold']['overall_score'] for d in all_results.values()]
    aucs = [d['auc'] for d in all_results.values()]
    x = np.arange(len(labels_bar))
    w = 0.25
    ax.bar(x - w, baseline_accs, w, label='Known Baseline', color='#3498db', alpha=0.8)
    ax.bar(x, best_scores, w, label='Best w/ Refusal', color='#2ecc71', alpha=0.8)
    ax.bar(x + w, aucs, w, label='AUC', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('Baseline vs Refusal-Augmented')
    ax.legend(fontsize=8)

    fig.suptitle('Phase 116b: Calibrated Refusal Curve (Cross-Architecture)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase116b_refusal.png'), dpi=150)
    plt.close()
    print("[Phase 116b] Complete.")


if __name__ == '__main__':
    main()
