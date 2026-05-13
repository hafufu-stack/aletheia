# -*- coding: utf-8 -*-
"""
Phase 105b: Alignment Tax Autopsy (NaN fix)
Fixed: float32 + safe entropy computation.
"""
import torch, json, os, gc, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FACTS = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("The chemical symbol for gold is", " Au"),
    ("The tallest mountain in the world is", " Mount"),
    ("The first president of the United States was", " George"),
    ("The chemical formula for water is", " H"),
]

def safe_entropy(attn_weights):
    w = attn_weights.float().clamp(min=1e-10).cpu().numpy()
    return float(-(w * np.log(w)).sum())

def get_layer_entropies(model, tok, facts, n_layers):
    layer_entropies = np.zeros(n_layers)
    layer_counts = np.zeros(n_layers)
    for prompt, answer in facts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        for layer_idx in range(n_layers):
            attn = out.attentions[layer_idx][0]
            for h in range(attn.shape[0]):
                ent = safe_entropy(attn[h, -1, :])
                if not np.isnan(ent):
                    layer_entropies[layer_idx] += ent
                    layer_counts[layer_idx] += 1
    return layer_entropies / np.maximum(layer_counts, 1)

def main():
    print(f"[P105b] Alignment Tax Autopsy - NaN fix (device={DEVICE})")

    models_to_test = [
        ('Qwen/Qwen2.5-0.5B', 'Base'),
        ('Qwen/Qwen2.5-0.5B-Instruct', 'Instruct'),
    ]
    all_entropies = {}

    for model_id, label in models_to_test:
        print(f"\n  === {label} ({model_id}) ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation='eager'
        ).eval().to(DEVICE)
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        n_layers = model.config.num_hidden_layers

        entropies = get_layer_entropies(model, tok, FACTS, n_layers)
        all_entropies[label] = entropies
        print(f"    Mean: {entropies.mean():.4f}, Front: {entropies[:n_layers//2].mean():.4f}, Back: {entropies[n_layers//2:].mean():.4f}")

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    base_ent = all_entropies['Base']
    inst_ent = all_entropies['Instruct']
    delta = inst_ent - base_ent
    n_layers = len(base_ent)
    mid = n_layers // 2

    front_delta = delta[:mid].mean()
    back_delta = delta[mid:].mean()

    print(f"\n  === ALIGNMENT TAX ===")
    print(f"  Front (semantic): {front_delta:+.4f}")
    print(f"  Back (syntactic): {back_delta:+.4f}")
    print(f"  Overall: {delta.mean():+.4f}")

    # Hypertrophy = back layers MORE affected than front
    hypertrophy = abs(back_delta) > abs(front_delta)
    print(f"  Suppressor Hypertrophy: {'CONFIRMED' if hypertrophy else 'REJECTED'}")

    most_changed = np.argsort(np.abs(delta))[::-1]
    print(f"  Most affected: {[f'L{l}({delta[l]:+.3f})' for l in most_changed[:5]]}")

    results = {
        'phase': 105, 'name': 'Alignment Tax Autopsy',
        'base_entropies': base_ent.tolist(),
        'instruct_entropies': inst_ent.tolist(),
        'delta': delta.tolist(),
        'front_half_delta': float(front_delta),
        'back_half_delta': float(back_delta),
        'hypertrophy_confirmed': bool(hypertrophy),
        'most_affected_layers': [int(l) for l in most_changed[:10]],
    }
    with open(os.path.join(RESULTS_DIR, 'phase105_alignment_autopsy.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    layers = range(n_layers)
    axes[0].plot(layers, base_ent, 'o-', color='#2ecc71', label='Base', markersize=4)
    axes[0].plot(layers, inst_ent, 's-', color='#e74c3c', label='Instruct', markersize=4)
    axes[0].set_xlabel('Layer'); axes[0].set_ylabel('Mean Entropy')
    axes[0].set_title('Base vs Instruct Entropy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    colors_bar = ['#2ecc71' if d >= 0 else '#e74c3c' for d in delta]
    axes[1].bar(layers, delta, color=colors_bar)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].axvline(x=mid, color='red', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Layer'); axes[1].set_ylabel('Delta (Inst-Base)')
    axes[1].set_title('Alignment Tax per Layer')

    cats = ['Front (Fact)', 'Back (Suppressor)', 'Overall']
    vals = [front_delta, back_delta, delta.mean()]
    axes[2].bar(cats, vals, color=['#2ecc71' if v>=0 else '#e74c3c' for v in vals])
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].set_ylabel('Mean Delta'); axes[2].set_title('Tax by Region')

    fig.suptitle('Phase 105: Alignment Tax Autopsy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase105_alignment_autopsy.png'), dpi=150)
    plt.close()
    print("[Phase 105] Complete.")

if __name__ == '__main__':
    main()
