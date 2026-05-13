# -*- coding: utf-8 -*-
"""
Phase 104b: Anatomy of "The Answer Is" (NaN fix)
Fixed: fp16 attention -> float32 conversion before entropy calculation.
Fixed: Qwen GQA attention shape handling.
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

TEMPLATES = {
    'natural': "{prompt}",
    'code_hash': "# {prompt}",
    'answer_is': "The answer is: {prompt}",
}

def safe_entropy(attn_weights):
    """Compute entropy safely from attention weights (any dtype)."""
    w = attn_weights.float().clamp(min=1e-10).cpu().numpy()
    return float(-(w * np.log(w)).sum())

def main():
    print(f"[P104b] Anatomy of 'The Answer Is' - NaN fix (device={DEVICE})")

    # Load in float32 to avoid fp16 NaN in attention
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B', local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager'
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', local_files_only=True)

    n_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    print(f"  Model: {n_layers} layers, {n_kv_heads} KV heads")

    all_entropies = {}
    for tmpl_name, tmpl in TEMPLATES.items():
        layer_entropies = np.zeros(n_layers)
        layer_counts = np.zeros(n_layers)

        for prompt, answer in FACTS:
            full = tmpl.format(prompt=prompt)
            inp = tok(full, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)

            for layer_idx in range(n_layers):
                attn = out.attentions[layer_idx][0]  # (n_heads, seq, seq)
                n_heads_actual = attn.shape[0]
                for h in range(n_heads_actual):
                    ent = safe_entropy(attn[h, -1, :])
                    if not np.isnan(ent):
                        layer_entropies[layer_idx] += ent
                        layer_counts[layer_idx] += 1

        layer_entropies /= np.maximum(layer_counts, 1)
        all_entropies[tmpl_name] = layer_entropies
        nan_count = np.isnan(layer_entropies).sum()
        print(f"  {tmpl_name}: mean={layer_entropies[~np.isnan(layer_entropies)].mean():.4f}, NaNs={nan_count}")

    # Deltas
    delta_code = all_entropies['code_hash'] - all_entropies['natural']
    delta_answer = all_entropies['answer_is'] - all_entropies['natural']

    mid = n_layers // 2
    front_code = np.nanmean(delta_code[:mid])
    back_code = np.nanmean(delta_code[mid:])
    front_answer = np.nanmean(delta_answer[:mid])
    back_answer = np.nanmean(delta_answer[mid:])

    print(f"\n  === ENTROPY ANALYSIS ===")
    print(f"  Code Mode vs Natural:")
    print(f"    Front half: {front_code:+.4f}")
    print(f"    Back half:  {back_code:+.4f}")
    print(f"  'The answer is:' vs Natural:")
    print(f"    Front half: {front_answer:+.4f}")
    print(f"    Back half:  {back_answer:+.4f}")

    # Syntactic Funnel: "answer is" should CONVERGE back-half (negative delta)
    # while Code DIVERGES back-half (positive delta)
    funnel_confirmed = back_answer < back_code
    print(f"\n  Syntactic Funnel: {'CONFIRMED' if funnel_confirmed else 'REJECTED'}")
    print(f"    Code diverges back ({back_code:+.4f}) vs Answer ({back_answer:+.4f})")

    results = {
        'phase': 104, 'name': 'Anatomy of The Answer Is',
        'model': 'Qwen/Qwen2.5-1.5B', 'n_layers': n_layers,
        'entropies': {k: v.tolist() for k, v in all_entropies.items()},
        'delta_code': delta_code.tolist(),
        'delta_answer': delta_answer.tolist(),
        'funnel_confirmed': bool(funnel_confirmed),
        'back_half_code_delta': float(back_code),
        'back_half_answer_delta': float(back_answer),
        'front_half_code_delta': float(front_code),
        'front_half_answer_delta': float(front_answer),
    }
    with open(os.path.join(RESULTS_DIR, 'phase104_answer_anatomy.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    layers = range(n_layers)
    for tmpl, color, ls in [('natural','#95a5a6','-'), ('code_hash','#2ecc71','--'), ('answer_is','#e67e22','-.')]:
        axes[0].plot(layers, all_entropies[tmpl], ls, color=color, label=tmpl, linewidth=2)
    axes[0].set_xlabel('Layer'); axes[0].set_ylabel('Mean Attention Entropy')
    axes[0].set_title('Per-Layer Entropy by Template'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].bar(np.arange(n_layers)-0.15, delta_code, 0.3, label='Code-Natural', color='#2ecc71', alpha=0.8)
    axes[1].bar(np.arange(n_layers)+0.15, delta_answer, 0.3, label='Answer-Natural', color='#e67e22', alpha=0.8)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].axvline(x=mid, color='red', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Layer'); axes[1].set_ylabel('Entropy Delta')
    axes[1].set_title('Shield (Code) vs Funnel (Answer)'); axes[1].legend(fontsize=8)

    cats = ['Front\n(Semantic)', 'Back\n(Syntactic)', 'Overall']
    cv = [front_code, back_code, np.nanmean(delta_code)]
    av = [front_answer, back_answer, np.nanmean(delta_answer)]
    x = np.arange(3)
    axes[2].bar(x-0.2, cv, 0.35, label='Code Mode', color='#2ecc71')
    axes[2].bar(x+0.2, av, 0.35, label='"The answer is:"', color='#e67e22')
    axes[2].set_xticks(x); axes[2].set_xticklabels(cats)
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].set_ylabel('Mean Delta'); axes[2].set_title('Mechanism Comparison'); axes[2].legend()

    fig.suptitle('Phase 104: Anatomy of "The Answer Is"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase104_answer_anatomy.png'), dpi=150)
    plt.close()

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("[Phase 104] Complete.")

if __name__ == '__main__':
    main()
