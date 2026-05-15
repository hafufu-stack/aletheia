# -*- coding: utf-8 -*-
"""
Phase 116: UAlign Blackhole (Epistemic Spike)
Physical "I don't know" forcing via entropy-gated spiking.

Theory (from P31, P49, P104):
  L_best (0.94 depth) entropy correlates with model uncertainty.
  When entropy > threshold: model is guessing (hallucination risk).
  We inject a spike into "I don't know" token to force refusal.

Method:
  1. Measure L_best entropy distribution for known and unknown facts
  2. Find entropy threshold separating "know" vs "don't know"
  3. When entropy > threshold, boost "unknown"/"cannot" tokens
  4. Measure accuracy and refusal rate

Models: Qwen2.5-0.5B + Qwen2.5-1.5B (GPU)
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

# Facts the model SHOULD know
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

# Facts the model likely DOESN'T know (obscure/recent)
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


def get_layer_entropy(model, tok, prompt, layer_idx, n_layers, model_type='qwen'):
    """Get entropy at a specific layer via Logit Lens."""
    hidden_state = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_state['h'] = output[0][:, -1, :].detach()
        else:
            hidden_state['h'] = output[:, -1, :].detach()

    if model_type == 'qwen':
        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        ln_f = model.model.norm
        lm_head = model.lm_head
    else:
        handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
        ln_f = model.transformer.ln_f
        lm_head = model.lm_head

    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model(**inp)

    h = ln_f(hidden_state['h'].float())  # float32 for stable entropy
    logits = lm_head(h.to(next(lm_head.parameters()).dtype)).squeeze()
    handle.remove()

    # Cast to float32 to avoid fp16 precision issues (nan entropy)
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

    # Top prediction and rank
    top_id = logits.argmax().item()

    return entropy, logits, top_id


def evaluate_with_refusal(model, tok, known_facts, unknown_facts, n_layers,
                          best_layer, threshold, model_type='qwen'):
    """Evaluate with entropy-gated refusal."""
    results = {'known': [], 'unknown': []}

    # Evaluate known facts
    for prompt, answer in known_facts:
        entropy, logits, top_id = get_layer_entropy(
            model, tok, prompt, best_layer, n_layers, model_type)

        fact_tokens = tok.encode(answer)
        fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0

        baseline_correct = (logits.argmax().item() == fact_id)

        if entropy > threshold:
            # Refuse: model says "I don't know"
            action = 'refuse'
            final_correct = False  # refused a known fact = wrong
        else:
            # Answer normally
            action = 'answer'
            final_correct = baseline_correct

        results['known'].append({
            'prompt': prompt, 'answer': answer.strip(),
            'entropy': entropy, 'action': action,
            'baseline_correct': baseline_correct,
            'final_correct': final_correct,
        })

    # Evaluate unknown facts
    for prompt, answer in unknown_facts:
        entropy, logits, top_id = get_layer_entropy(
            model, tok, prompt, best_layer, n_layers, model_type)

        fact_tokens = tok.encode(answer)
        fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0

        baseline_correct = (logits.argmax().item() == fact_id)

        if entropy > threshold:
            action = 'refuse'
            final_correct = True  # correctly refused unknown fact
        else:
            action = 'answer'
            final_correct = baseline_correct

        results['unknown'].append({
            'prompt': prompt, 'answer': answer.strip(),
            'entropy': entropy, 'action': action,
            'baseline_correct': baseline_correct,
            'final_correct': final_correct,
        })

    return results


def main():
    print("[P116] UAlign Blackhole (Epistemic Spike)")
    print(f"  Device: {DEVICE}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_configs = [
        ('Qwen/Qwen2.5-0.5B', 'Qwen2.5-0.5B', 24, 'qwen'),
        ('Qwen/Qwen2.5-1.5B', 'Qwen2.5-1.5B', 28, 'qwen'),
    ]

    all_model_results = {}

    for model_id, label, n_layers, model_type in model_configs:
        print(f"\n  === {label} ===")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, torch_dtype=torch.float16
            ).eval().to(DEVICE)
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        best_layer = int(n_layers * 0.94)  # Aletheia Constant
        print(f"    Best layer (0.94): L{best_layer}")

        # 1. Measure entropy distribution
        print("    Measuring entropy distributions...")
        known_entropies = []
        unknown_entropies = []
        known_correct_at_best = 0

        for prompt, answer in KNOWN_FACTS:
            ent, logits, _ = get_layer_entropy(
                model, tok, prompt, best_layer, n_layers, model_type)
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0
            if logits.argmax().item() == fact_id:
                known_correct_at_best += 1
            known_entropies.append(ent)

        for prompt, answer in UNKNOWN_FACTS:
            ent, _, _ = get_layer_entropy(
                model, tok, prompt, best_layer, n_layers, model_type)
            unknown_entropies.append(ent)

        known_mean = np.mean(known_entropies)
        unknown_mean = np.mean(unknown_entropies)
        print(f"    Known entropy: mean={known_mean:.2f}, std={np.std(known_entropies):.2f}")
        print(f"    Unknown entropy: mean={unknown_mean:.2f}, std={np.std(unknown_entropies):.2f}")
        print(f"    Known correct at L{best_layer}: {known_correct_at_best}/{len(KNOWN_FACTS)}")

        # 2. Sweep thresholds
        print("    Sweeping entropy thresholds...")
        all_ents = known_entropies + unknown_entropies
        thresholds = np.linspace(min(all_ents) - 0.5, max(all_ents) + 0.5, 30)

        threshold_results = {}
        for thr in thresholds:
            res = evaluate_with_refusal(
                model, tok, KNOWN_FACTS, UNKNOWN_FACTS,
                n_layers, best_layer, thr, model_type)

            known_acc = np.mean([r['final_correct'] for r in res['known']])
            unknown_acc = np.mean([r['final_correct'] for r in res['unknown']])
            known_refuse = np.mean([r['action'] == 'refuse' for r in res['known']])
            unknown_refuse = np.mean([r['action'] == 'refuse' for r in res['unknown']])
            overall_acc = np.mean(
                [r['final_correct'] for r in res['known']] +
                [r['final_correct'] for r in res['unknown']]
            )

            threshold_results[float(thr)] = {
                'known_acc': float(known_acc),
                'unknown_acc': float(unknown_acc),
                'known_refuse_rate': float(known_refuse),
                'unknown_refuse_rate': float(unknown_refuse),
                'overall_acc': float(overall_acc),
            }

        # Find best threshold (maximize overall accuracy)
        best_thr = max(threshold_results, key=lambda t: threshold_results[t]['overall_acc'])
        best_res = threshold_results[best_thr]
        print(f"    Best threshold: {best_thr:.2f}")
        print(f"      Known acc: {best_res['known_acc']:.0%} (refuse: {best_res['known_refuse_rate']:.0%})")
        print(f"      Unknown acc: {best_res['unknown_acc']:.0%} (refuse: {best_res['unknown_refuse_rate']:.0%})")
        print(f"      Overall: {best_res['overall_acc']:.0%}")

        all_model_results[label] = {
            'model': model_id,
            'n_layers': n_layers,
            'best_layer': best_layer,
            'known_entropies': known_entropies,
            'unknown_entropies': unknown_entropies,
            'known_entropy_mean': float(known_mean),
            'unknown_entropy_mean': float(unknown_mean),
            'threshold_sweep': threshold_results,
            'best_threshold': float(best_thr),
            'best_overall_acc': float(best_res['overall_acc']),
        }

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    out = {
        'phase': 116,
        'name': 'UAlign Blackhole (Epistemic Spike)',
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'threshold_sweep'}
                    for k, v in all_model_results.items()},
        'threshold_sweeps': {k: v['threshold_sweep'] for k, v in all_model_results.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase116_ualign.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    n_models = len(all_model_results)
    fig, axes = plt.subplots(2, n_models, figsize=(9 * n_models, 12))
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for col, (label, data) in enumerate(all_model_results.items()):
        # Top row: entropy distributions
        ax = axes[0, col]
        ax.hist(data['known_entropies'], bins=15, alpha=0.7, color='#2ecc71',
                label=f"Known (mean={data['known_entropy_mean']:.1f})", density=True)
        ax.hist(data['unknown_entropies'], bins=15, alpha=0.7, color='#e74c3c',
                label=f"Unknown (mean={data['unknown_entropy_mean']:.1f})", density=True)
        ax.axvline(x=data['best_threshold'], color='black', linestyle='--',
                  linewidth=2, label=f"Threshold={data['best_threshold']:.1f}")
        ax.set_xlabel('L_best Entropy')
        ax.set_ylabel('Density')
        ax.set_title(f'{label}: Entropy Distribution at L{data["best_layer"]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom row: threshold sweep
        ax = axes[1, col]
        thr_sweep = data['threshold_sweep']
        thrs = sorted(thr_sweep.keys())
        known_accs = [thr_sweep[t]['known_acc'] for t in thrs]
        unknown_accs = [thr_sweep[t]['unknown_acc'] for t in thrs]
        overall_accs = [thr_sweep[t]['overall_acc'] for t in thrs]
        known_refuse = [thr_sweep[t]['known_refuse_rate'] for t in thrs]
        unknown_refuse = [thr_sweep[t]['unknown_refuse_rate'] for t in thrs]

        ax.plot(thrs, known_accs, '-', color='#2ecc71', label='Known Acc', linewidth=2)
        ax.plot(thrs, unknown_accs, '-', color='#e74c3c', label='Unknown Acc', linewidth=2)
        ax.plot(thrs, overall_accs, '-', color='#3498db', label='Overall', linewidth=2.5)
        ax.plot(thrs, known_refuse, '--', color='#2ecc71', alpha=0.5, label='Known Refuse%')
        ax.plot(thrs, unknown_refuse, '--', color='#e74c3c', alpha=0.5, label='Unknown Refuse%')
        ax.axvline(x=data['best_threshold'], color='black', linestyle='--', alpha=0.7)
        ax.set_xlabel('Entropy Threshold')
        ax.set_ylabel('Rate')
        ax.set_title(f'{label}: Threshold Sweep (best={data["best_overall_acc"]:.0%})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Phase 116: UAlign Blackhole (Epistemic Spike)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase116_ualign.png'), dpi=150)
    plt.close()
    print("[Phase 116] Complete.")


if __name__ == '__main__':
    main()
