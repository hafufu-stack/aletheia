# -*- coding: utf-8 -*-
"""
Phase 160: The Semantic Category Map (Opus original)
Does the LM_head phase transition affect ALL semantic categories?

If "number clustering" in lm_head is universal (countries, colors,
animals all cluster at 14B), then Dual Surgery is needed for
ALL categories of factual knowledge, not just numbers.

If it's NUMBER-SPECIFIC, then numbers have a unique property.

Model: Qwen2.5-0.5B and 14B comparison
"""
import torch, json, os, gc, numpy as np, time
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CATEGORIES = {
    'digits': [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"],
    'countries': [" Japan", " France", " Germany", " Italy", " Spain",
                  " China", " India", " Brazil", " Canada", " Russia"],
    'capitals': [" Tokyo", " Paris", " Berlin", " Rome", " Madrid",
                 " Beijing", " Delhi", " Brasilia", " Ottawa", " Moscow"],
    'colors': [" red", " blue", " green", " yellow", " black",
               " white", " orange", " purple", " pink", " brown"],
    'animals': [" cat", " dog", " bird", " fish", " horse",
                " lion", " tiger", " bear", " wolf", " fox"],
    'elements': [" hydrogen", " helium", " carbon", " nitrogen", " oxygen",
                 " sodium", " iron", " gold", " silver", " copper"],
    'planets': [" Mercury", " Venus", " Earth", " Mars", " Jupiter",
                " Saturn", " Uranus", " Neptune"],
    'months': [" January", " February", " March", " April", " May",
               " June", " July", " August", " September", " October"],
}


def analyze_model(model_id, name, tok=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token

    try:
        if '14B' in name or '7B' in name or '32B' in name:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, quantization_config=bnb,
                device_map="auto", torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, torch_dtype=torch.float32)
    except Exception as e:
        print(f"    SKIP: {e}")
        return None

    lm = model.lm_head.weight.data.float()
    embed = model.model.embed_tokens.weight.data.float()
    d = model.config.hidden_size

    result = {'name': name, 'd': d}

    for cat_name, tokens in CATEGORIES.items():
        ids = [tok.encode(t)[-1] for t in tokens]
        n = len(ids)

        # LM head
        vecs = lm[ids]
        cos_mat = F.cosine_similarity(vecs.unsqueeze(0), vecs.unsqueeze(1), dim=-1)
        mask = ~torch.eye(n, dtype=bool)
        avg_cos_lm = cos_mat[mask].mean().item()

        # Embed
        evecs = embed[ids]
        ecos_mat = F.cosine_similarity(evecs.unsqueeze(0), evecs.unsqueeze(1), dim=-1)
        avg_cos_embed = ecos_mat[mask].mean().item()

        # L2
        l2_mat = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0)).squeeze(0)
        avg_l2 = l2_mat[mask].mean().item()

        result[cat_name] = {
            'lm_cos': avg_cos_lm, 'embed_cos': avg_cos_embed, 'l2': avg_l2
        }

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def main():
    print("[P160] The Semantic Category Map")
    start_time = time.time()

    models = [
        ('Qwen/Qwen2.5-0.5B', '0.5B'),
        ('Qwen/Qwen2.5-1.5B', '1.5B'),
        ('Qwen/Qwen2.5-14B', '14B'),
    ]

    results = []
    for model_id, name in models:
        print(f"\n  === {name} ===")
        r = analyze_model(model_id, name)
        if r:
            results.append(r)
            for cat in CATEGORIES:
                v = r[cat]
                print(f"    {cat:12s}: lm_cos={v['lm_cos']:.4f}, embed_cos={v['embed_cos']:.4f}")

    with open(os.path.join(RESULTS_DIR, 'phase160_categories.json'), 'w') as f:
        json.dump({'phase': '160', 'name': 'Semantic Category Map',
                   'results': results}, f, indent=2, default=str)

    if len(results) < 2:
        print("  Not enough models"); return

    # Plot: heatmap of lm_cos per category per model
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 8))
    if len(results) == 1: axes = [axes]

    cats = list(CATEGORIES.keys())
    for i, r in enumerate(results):
        ax = axes[i]
        lm_vals = [r[c]['lm_cos'] for c in cats]
        colors = ['#e74c3c' if v > 0.5 else '#f39c12' if v > 0.2 else '#2ecc71' for v in lm_vals]
        bars = ax.barh(cats, lm_vals, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
        for j, v in enumerate(lm_vals):
            ax.text(v + 0.02, j, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
        ax.set_xlim(-0.1, 1.0)
        ax.set_title(f'{r["name"]} (d={r["d"]})', fontsize=13, fontweight='bold')
        ax.set_xlabel('LM_head Cosine Similarity')
        ax.axvline(x=0.5, color='red', ls='--', alpha=0.3)
        ax.grid(True, alpha=0.2, axis='x')

    plt.suptitle('Phase 160: Semantic Category Map\n'
                'Which categories cluster in the output space?\n'
                '(Red > 0.5 = severely clustered)',
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase160_categories.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    if len(results) >= 2:
        small = results[0]
        big = results[-1]
        print(f"  Comparing {small['name']} vs {big['name']}:")
        for cat in cats:
            s_cos = small[cat]['lm_cos']
            b_cos = big[cat]['lm_cos']
            change = "CLUSTERED" if b_cos > s_cos + 0.3 else "stable"
            print(f"    {cat:12s}: {s_cos:.3f} -> {b_cos:.3f} [{change}]")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 160] Complete.")

if __name__ == '__main__':
    main()
