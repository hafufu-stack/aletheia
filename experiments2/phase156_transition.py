# -*- coding: utf-8 -*-
"""
Phase 156: LM_head Phase Transition
At what model scale does lm_head clustering "phase transition"?

P154: 0.5B/1.5B lm_head cos=0.01, 14B cos=0.74.
Where exactly does the transition happen?

Model: All available Qwen2.5 sizes (diagnostic, inference only)
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


def analyze_lm_head(model_id, name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except:
        print(f"    SKIP {name}: tokenizer not available")
        return None
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    try:
        params_B = 0
        if any(x in name for x in ['7B', '14B', '32B', '72B']):
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, quantization_config=bnb,
                device_map="auto", torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, torch_dtype=torch.float32)
        params_B = sum(p.numel() for p in model.parameters()) / 1e9
    except Exception as e:
        print(f"    SKIP {name}: {e}")
        return None

    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    # Categories to measure
    categories = {
        'digits': [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"],
        'countries': [" Japan", " France", " Germany", " Italy", " Spain",
                     " China", " India", " Brazil", " Canada", " Russia"],
        'colors': [" red", " blue", " green", " yellow", " black",
                  " white", " orange", " purple", " pink", " brown"],
    }

    lm = model.lm_head.weight.data.float()
    embed = model.model.embed_tokens.weight.data.float()

    result = {
        'name': name, 'd': d, 'n_layers': n_layers, 'params_B': params_B,
    }

    for cat_name, tokens in categories.items():
        # LM_head
        ids = [tok.encode(t)[-1] for t in tokens]
        vecs = lm[ids]
        cos = F.cosine_similarity(vecs.unsqueeze(0), vecs.unsqueeze(1), dim=-1)
        mask = ~torch.eye(len(ids), dtype=bool)
        avg_cos_lm = cos[mask].mean().item()
        # Embed
        evecs = embed[ids]
        ecos = F.cosine_similarity(evecs.unsqueeze(0), evecs.unsqueeze(1), dim=-1)
        avg_cos_embed = ecos[mask].mean().item()
        # Norms
        avg_norm_lm = vecs.norm(dim=-1).mean().item()
        avg_norm_embed = evecs.norm(dim=-1).mean().item()

        result[f'{cat_name}_lm_cos'] = avg_cos_lm
        result[f'{cat_name}_embed_cos'] = avg_cos_embed
        result[f'{cat_name}_lm_norm'] = avg_norm_lm
        result[f'{cat_name}_embed_norm'] = avg_norm_embed

    print(f"    {name} (d={d}, L={n_layers}, {params_B:.2f}B params):")
    for cat_name in categories:
        lc = result[f'{cat_name}_lm_cos']
        ec = result[f'{cat_name}_embed_cos']
        print(f"      {cat_name:10s}: embed_cos={ec:.4f}, lm_cos={lc:.4f}")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def main():
    print("[P156] LM_head Phase Transition")
    start_time = time.time()

    models = [
        ('Qwen/Qwen2.5-0.5B', '0.5B'),
        ('Qwen/Qwen2.5-1.5B', '1.5B'),
        ('Qwen/Qwen2.5-3B', '3B'),
        ('Qwen/Qwen2.5-7B', '7B'),
        ('Qwen/Qwen2.5-14B', '14B'),
        ('Qwen/Qwen2.5-32B', '32B'),
    ]

    results = []
    for model_id, name in models:
        print(f"\n  === {name} ===")
        r = analyze_lm_head(model_id, name)
        if r: results.append(r)

    with open(os.path.join(RESULTS_DIR, 'phase156_transition.json'), 'w') as f:
        json.dump({'phase': '156', 'name': 'LM_head Phase Transition',
                   'results': results}, f, indent=2, default=str)

    if len(results) < 2:
        print("  Not enough models for plot")
        return

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = [r['name'] for r in results]
    params = [r['params_B'] for r in results]

    # Left: LM_head cos by category
    ax = axes[0]
    for cat, color, marker in [('digits', '#e74c3c', 'o'),
                                ('countries', '#3498db', 's'),
                                ('colors', '#2ecc71', '^')]:
        vals = [r.get(f'{cat}_lm_cos', 0) for r in results]
        ax.plot(params, vals, f'-{marker}', color=color, lw=2.5, markersize=10, label=cat)
        for i, v in enumerate(vals):
            ax.annotate(f'{v:.3f}', (params[i], v), fontsize=8, ha='left',
                       xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Model Parameters (Billions)', fontsize=12)
    ax.set_ylabel('LM_head Avg Cosine Similarity', fontsize=12)
    ax.set_title('Output Space Clustering\n(Phase Transition)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.3)

    # Right: Embed cos by category
    ax = axes[1]
    for cat, color, marker in [('digits', '#e74c3c', 'o'),
                                ('countries', '#3498db', 's'),
                                ('colors', '#2ecc71', '^')]:
        vals = [r.get(f'{cat}_embed_cos', 0) for r in results]
        ax.plot(params, vals, f'-{marker}', color=color, lw=2.5, markersize=10, label=cat)
    ax.set_xlabel('Model Parameters (Billions)', fontsize=12)
    ax.set_ylabel('Embed Avg Cosine Similarity', fontsize=12)
    ax.set_title('Input Space Clustering\n(Stable?)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.suptitle('Phase 156: The LM_head Phase Transition\nDoes output clustering increase with model scale?',
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase156_transition.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    for r in results:
        print(f"    {r['name']:5s}: digits_lm={r['digits_lm_cos']:.4f}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 156] Complete.")

if __name__ == '__main__':
    main()
