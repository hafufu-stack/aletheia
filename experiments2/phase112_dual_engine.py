# -*- coding: utf-8 -*-
"""
Phase 112: Dual-Engine Universality (from DT1)
P97 proved Shield+Sword on GPT-2 XL. Does the same pattern exist in Qwen?
Tests Qwen-1.5B for Dual-Engine head-level analysis.
GPU, float32.
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

def main():
    print(f"[P112] Dual-Engine Universality (device={DEVICE})")

    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B', local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager'
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', local_files_only=True)

    n_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    print(f"  Qwen-1.5B: {n_layers} layers, {n_kv_heads} KV heads")

    # Collect per-head entropy for natural vs code mode
    head_entropies = {'natural': {}, 'code': {}}

    for mode, prefix in [('natural', ''), ('code', '# ')]:
        for prompt, answer in FACTS:
            full = prefix + prompt
            inp = tok(full, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            for l in range(n_layers):
                attn = out.attentions[l][0]  # (n_heads, seq, seq)
                for h in range(attn.shape[0]):
                    key = (l, h)
                    ent = safe_entropy(attn[h, -1, :])
                    if not np.isnan(ent):
                        head_entropies[mode].setdefault(key, []).append(ent)

    # Average and compute delta
    head_delta = {}
    for key in head_entropies['natural']:
        nat_mean = np.mean(head_entropies['natural'].get(key, [0]))
        code_mean = np.mean(head_entropies['code'].get(key, [0]))
        head_delta[key] = code_mean - nat_mean

    # Find top amplifiers (negative delta = more focused in code)
    # and top suppressors (positive delta = more diffuse in code)
    sorted_heads = sorted(head_delta.items(), key=lambda x: x[1])
    amplifiers = sorted_heads[:10]  # most negative = strengthened by code
    suppressors = sorted_heads[-10:][::-1]  # most positive = weakened by code

    print(f"\n  Top 5 AMPLIFIERS (strengthened by Code Mode, front expected):")
    for (l, h), d in amplifiers[:5]:
        print(f"    L{l}H{h}: delta={d:+.4f}")

    print(f"\n  Top 5 SUPPRESSORS (weakened by Code Mode, back expected):")
    for (l, h), d in suppressors[:5]:
        print(f"    L{l}H{h}: delta={d:+.4f}")

    # Analyze: are amplifiers in front half, suppressors in back half?
    amp_layers = [l for (l, h), d in amplifiers[:5]]
    sup_layers = [l for (l, h), d in suppressors[:5]]
    mid = n_layers // 2
    amp_front = sum(1 for l in amp_layers if l < mid)
    sup_back = sum(1 for l in sup_layers if l >= mid)
    dual_engine = amp_front >= 3 and sup_back >= 3

    print(f"\n  Amplifiers in front half: {amp_front}/5")
    print(f"  Suppressors in back half: {sup_back}/5")
    print(f"  Dual-Engine Universal: {'CONFIRMED' if dual_engine else 'PARTIAL'}")

    results = {
        'phase': 112, 'name': 'Dual-Engine Universality',
        'model': 'Qwen/Qwen2.5-1.5B', 'n_layers': n_layers,
        'amplifiers': [{'layer': l, 'head': h, 'delta': d} for (l, h), d in amplifiers[:10]],
        'suppressors': [{'layer': l, 'head': h, 'delta': d} for (l, h), d in suppressors[:10]],
        'dual_engine_confirmed': bool(dual_engine),
        'amp_front_count': amp_front,
        'sup_back_count': sup_back,
    }
    with open(os.path.join(RESULTS_DIR, 'phase112_dual_engine.json'), 'w') as f:
        json.dump(results, f, indent=2)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Delta heatmap
    delta_matrix = np.zeros((n_layers, n_kv_heads))
    for (l, h), d in head_delta.items():
        if h < n_kv_heads:
            delta_matrix[l, h] = d
    im = axes[0].imshow(delta_matrix.T, aspect='auto', cmap='RdYlGn_r',
                        vmin=-0.3, vmax=0.3)
    axes[0].set_xlabel('Layer'); axes[0].set_ylabel('Head')
    axes[0].set_title('Code-Natural Entropy Delta\n(Red=Suppressor, Green=Amplifier)')
    plt.colorbar(im, ax=axes[0], label='Delta')

    # 2. Per-layer mean delta
    layer_deltas = np.zeros(n_layers)
    layer_counts = np.zeros(n_layers)
    for (l, h), d in head_delta.items():
        layer_deltas[l] += d
        layer_counts[l] += 1
    layer_mean = layer_deltas / np.maximum(layer_counts, 1)
    colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in layer_mean]
    axes[1].bar(range(n_layers), layer_mean, color=colors)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].axvline(x=mid, color='gold', linestyle='--', label='Midpoint')
    axes[1].set_xlabel('Layer'); axes[1].set_ylabel('Mean Delta')
    axes[1].set_title('Per-Layer: Suppressor (+) vs Amplifier (-)')
    axes[1].legend()

    fig.suptitle('Phase 112: Dual-Engine on Qwen-1.5B (vs GPT-2 XL P97)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase112_dual_engine.png'), dpi=150)
    plt.close()

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("[Phase 112] Complete.")

if __name__ == '__main__':
    main()
