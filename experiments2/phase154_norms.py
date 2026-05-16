# -*- coding: utf-8 -*-
"""
Phase 154: Embedding Norm Ratio Analysis
Why might surgery work at small scales but fail at large?

Hypothesis: Surgery strength s=2 creates a displacement relative
to the embedding NORM. If larger models have much larger norms,
the "effective" surgery strength is diluted.

Measure: embedding norms, L2 distances, and relative displacement
across all available model scales.

Model: All available Qwen2.5 models (GPU/CPU, diagnostic only)
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


def analyze_model(model_id, name, strength=2.0):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as e:
        print(f"    SKIP {name}: {e}")
        return None

    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Load model (use 4-bit for large models)
    try:
        if '14B' in name or '7B' in name:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, quantization_config=bnb,
                device_map="auto", torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, torch_dtype=torch.float32)
    except Exception as e:
        print(f"    SKIP {name}: {e}")
        return None

    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    n_params = sum(p.numel() for p in model.parameters())

    embed = model.model.embed_tokens.weight.data.float()
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]
    num_ids = [tok.encode(t)[-1] for t in num_tokens]

    # Pre-surgery metrics
    num_vecs = embed[num_ids]
    pre_norms = num_vecs.norm(dim=-1)
    pre_avg_norm = pre_norms.mean().item()
    pre_cos = F.cosine_similarity(num_vecs.unsqueeze(0), num_vecs.unsqueeze(1), dim=-1)
    pre_avg_cos = pre_cos[~torch.eye(10, dtype=bool)].mean().item()
    pre_l2 = torch.cdist(num_vecs.unsqueeze(0), num_vecs.unsqueeze(0)).squeeze(0)
    pre_avg_l2 = pre_l2[~torch.eye(10, dtype=bool)].mean().item()

    # All embeddings stats
    all_norms = embed.norm(dim=-1)
    global_avg_norm = all_norms.mean().item()

    # Apply surgery
    center = num_vecs.mean(dim=0)
    for i, idx in enumerate(num_ids):
        diff = num_vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * num_vecs[i].norm())

    # Post-surgery metrics
    post_vecs = embed[num_ids]
    post_norms = post_vecs.norm(dim=-1)
    post_avg_norm = post_norms.mean().item()
    post_cos = F.cosine_similarity(post_vecs.unsqueeze(0), post_vecs.unsqueeze(1), dim=-1)
    post_avg_cos = post_cos[~torch.eye(10, dtype=bool)].mean().item()
    post_l2 = torch.cdist(post_vecs.unsqueeze(0), post_vecs.unsqueeze(0)).squeeze(0)
    post_avg_l2 = post_l2[~torch.eye(10, dtype=bool)].mean().item()

    # Key ratios
    l2_increase = post_avg_l2 - pre_avg_l2
    l2_ratio = post_avg_l2 / (pre_avg_l2 + 1e-10)
    displacement_to_norm = l2_increase / (global_avg_norm + 1e-10)

    # FGA direction analysis: how separated are the lm_head vectors?
    lm_head = model.lm_head.weight.data.float()
    lm_num = lm_head[num_ids]
    lm_cos = F.cosine_similarity(lm_num.unsqueeze(0), lm_num.unsqueeze(1), dim=-1)
    lm_avg_cos = lm_cos[~torch.eye(10, dtype=bool)].mean().item()

    result = {
        'name': name, 'd': d, 'n_layers': n_layers,
        'n_params_M': n_params / 1e6,
        'pre_avg_norm': pre_avg_norm, 'post_avg_norm': post_avg_norm,
        'global_avg_norm': global_avg_norm,
        'pre_avg_cos': pre_avg_cos, 'post_avg_cos': post_avg_cos,
        'pre_avg_l2': pre_avg_l2, 'post_avg_l2': post_avg_l2,
        'l2_increase': l2_increase, 'l2_ratio': l2_ratio,
        'displacement_to_norm': displacement_to_norm,
        'lm_head_avg_cos': lm_avg_cos,
    }

    print(f"    {name}: d={d}, layers={n_layers}")
    print(f"      Embed norm: {pre_avg_norm:.3f} -> {post_avg_norm:.3f}")
    print(f"      Avg L2: {pre_avg_l2:.3f} -> {post_avg_l2:.3f} (ratio={l2_ratio:.2f}x)")
    print(f"      Avg Cos: {pre_avg_cos:.4f} -> {post_avg_cos:.4f}")
    print(f"      Displacement/GlobalNorm: {displacement_to_norm:.4f}")
    print(f"      LM_head num cos: {lm_avg_cos:.4f}")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def main():
    print("[P154] Embedding Norm Ratio Analysis")
    start_time = time.time()

    models = [
        ('Qwen/Qwen2.5-0.5B', '0.5B'),
        ('Qwen/Qwen2.5-1.5B', '1.5B'),
        ('Qwen/Qwen2.5-14B', '14B'),
    ]

    results = []
    for model_id, name in models:
        print(f"\n  === Analyzing {name} ===")
        r = analyze_model(model_id, name, strength=2.0)
        if r: results.append(r)

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase154_norms.json'), 'w') as f:
        json.dump({'phase': '154', 'name': 'Embedding Norm Ratio',
                   'results': results}, f, indent=2, default=str)

    if len(results) < 2:
        print("  Not enough models for comparison.")
        return

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    names = [r['name'] for r in results]
    x = np.arange(len(names))

    # Top-left: L2 distance before/after
    ax = axes[0, 0]
    pre = [r['pre_avg_l2'] for r in results]
    post = [r['post_avg_l2'] for r in results]
    w = 0.35
    ax.bar(x-w/2, pre, w, label='Pre-surgery', color='#95a5a6', alpha=0.8)
    ax.bar(x+w/2, post, w, label='Post-surgery', color='#e74c3c', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w/2, pre[i]+0.1, f'{pre[i]:.2f}', ha='center', fontsize=9)
        ax.text(x[i]+w/2, post[i]+0.1, f'{post[i]:.2f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.legend(fontsize=10); ax.set_ylabel('Avg L2 Distance')
    ax.set_title('Number Token L2 Distances', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Top-right: Displacement / Global norm
    ax = axes[0, 1]
    disp = [r['displacement_to_norm'] for r in results]
    colors = ['#2ecc71' if d > 0.1 else '#e74c3c' for d in disp]
    ax.bar(names, disp, color=colors, alpha=0.8, edgecolor='black', lw=1)
    for i, v in enumerate(disp):
        ax.text(i, v+0.005, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Displacement / Global Norm')
    ax.set_title('Relative Surgery Strength\n(Higher = stronger effect)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-left: Cosine similarity
    ax = axes[1, 0]
    pre_cos = [r['pre_avg_cos'] for r in results]
    post_cos = [r['post_avg_cos'] for r in results]
    ax.bar(x-w/2, pre_cos, w, label='Pre-surgery', color='#95a5a6', alpha=0.8)
    ax.bar(x+w/2, post_cos, w, label='Post-surgery', color='#3498db', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w/2, pre_cos[i]+0.01, f'{pre_cos[i]:.3f}', ha='center', fontsize=9)
        ax.text(x[i]+w/2, post_cos[i]+0.01, f'{post_cos[i]:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.legend(fontsize=10); ax.set_ylabel('Avg Cosine Similarity')
    ax.set_title('Number Token Cosine Similarity', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-right: LM_head cosine
    ax = axes[1, 1]
    lm_cos = [r['lm_head_avg_cos'] for r in results]
    ax.bar(names, lm_cos, color='#9b59b6', alpha=0.8, edgecolor='black', lw=1)
    for i, v in enumerate(lm_cos):
        ax.text(i, v+0.01, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Avg Cosine (LM Head)')
    ax.set_title('LM Head Number Token Clustering\n(Output projection similarity)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Phase 154: Why Does Surgery Fail at Scale?\nEmbedding Geometry Across Model Sizes',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase154_norms.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    for r in results:
        print(f"    {r['name']:5s}: norm={r['pre_avg_norm']:.2f}, "
              f"disp/norm={r['displacement_to_norm']:.4f}, "
              f"lm_cos={r['lm_head_avg_cos']:.4f}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 154] Complete.")

if __name__ == '__main__':
    main()
