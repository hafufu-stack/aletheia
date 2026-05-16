# -*- coding: utf-8 -*-
"""
Phase 143: GPT-2 L2 Universality Test (CPU experiment)
Is L2* ~ 1.2 universal across architectures, or Qwen-specific?

Test: Apply GS + norm scaling to GPT-2 Small (124M) and measure
the L2 critical threshold. If L2* ~ 1.2 holds for GPT-2 too,
it's a universal property of transformer embeddings.

GPT-2 uses different architecture (LayerNorm ordering, no SwiGLU),
different tokenizer, and much smaller hidden dim (d=768).

Model: GPT-2 Small (CPU, float32) - can run in parallel with GPU experiments
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

DEVICE = 'cpu'  # Explicitly CPU for parallel execution

# GPT-2 uses different token format
NUM_TOKENS_GPT2 = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]

# DPO pairs adapted for GPT-2's knowledge
DPO_PAIRS = [
    ("Water freezes at", " 0", " 100"),
    ("The boiling point of water is", " 100", " 200"),
    ("The speed of light is", " 299", " 186"),
]

# Additional fact pairs for evaluation
EVAL_PAIRS = [
    ("Water freezes at", " 0"),
    ("The boiling point of water is", " 100"),
    ("The speed of light is", " 299"),
]


def dpo_loss_gpt2(model, ref_model, tok, prompt, chosen, rejected, beta=0.1):
    text_c = prompt + chosen
    text_r = prompt + rejected
    inp_c = tok(text_c, return_tensors='pt')
    inp_r = tok(text_r, return_tensors='pt')
    plen = tok(prompt, return_tensors='pt')['input_ids'].shape[1]
    lc = model(**inp_c).logits[0, plen-1:-1, :].float().clamp(-100, 100)
    lr_ = model(**inp_r).logits[0, plen-1:-1, :].float().clamp(-100, 100)
    lp_c = F.log_softmax(lc, dim=-1).gather(1, inp_c['input_ids'][0, plen:].unsqueeze(1)).squeeze()
    lp_r = F.log_softmax(lr_, dim=-1).gather(1, inp_r['input_ids'][0, plen:].unsqueeze(1)).squeeze()
    if lp_c.dim() == 0: lp_c = lp_c.unsqueeze(0)
    if lp_r.dim() == 0: lp_r = lp_r.unsqueeze(0)
    with torch.no_grad():
        rlc = ref_model(**inp_c).logits[0, plen-1:-1, :].float().clamp(-100, 100)
        rlr = ref_model(**inp_r).logits[0, plen-1:-1, :].float().clamp(-100, 100)
        rlp_c = F.log_softmax(rlc, dim=-1).gather(1, inp_c['input_ids'][0, plen:].unsqueeze(1)).squeeze()
        rlp_r = F.log_softmax(rlr, dim=-1).gather(1, inp_r['input_ids'][0, plen:].unsqueeze(1)).squeeze()
        if rlp_c.dim() == 0: rlp_c = rlp_c.unsqueeze(0)
        if rlp_r.dim() == 0: rlp_r = rlp_r.unsqueeze(0)
    diff = beta * ((lp_c.sum() - rlp_c.sum()) - (lp_r.sum() - rlp_r.sum()))
    return -F.logsigmoid(diff)


def gram_schmidt_gpt2(model, tok):
    """Gram-Schmidt on GPT-2's wte embeddings."""
    embed = model.transformer.wte.weight.data
    ids = [tok.encode(t)[0] for t in NUM_TOKENS_GPT2]
    vecs = embed[ids].clone().float()
    norms = vecs.norm(dim=-1, keepdim=True)
    ortho = torch.zeros_like(vecs)
    for i in range(len(vecs)):
        v = vecs[i].clone()
        for j in range(i):
            proj = torch.dot(v, ortho[j]) / (torch.dot(ortho[j], ortho[j]) + 1e-8)
            v = v - proj * ortho[j]
        ortho[i] = v / (v.norm() + 1e-8) * norms[i]
    for i, idx in enumerate(ids):
        embed[idx] = ortho[i].to(embed.dtype)


def scale_norms_gpt2(model, tok, factor):
    embed = model.transformer.wte.weight.data
    ids = [tok.encode(t)[0] for t in NUM_TOKENS_GPT2]
    for idx in ids:
        embed[idx] *= factor


def measure_geometry_gpt2(model, tok):
    embed = model.transformer.wte.weight.data
    ids = [tok.encode(t)[0] for t in NUM_TOKENS_GPT2]
    vecs = embed[ids].float()
    norms = vecs.norm(dim=-1, keepdim=True)
    cos_mat = (vecs @ vecs.T) / (norms @ norms.T + 1e-8)
    mask = ~torch.eye(len(ids), dtype=bool)
    cos_val = cos_mat[mask].mean().item()
    dists = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0)).squeeze(0)
    l2 = dists[mask].mean().item()
    return l2, cos_val, norms.mean().item()


def train_eval_gpt2(model, ref, tok, n_layers=12):
    boundary = int(n_layers * 0.83)  # Last 2 layers for GPT-2 small
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"h.{i}." in name and "mlp" in name: p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return 0.0
    opt = torch.optim.AdamW(trainable, lr=5e-5)
    for epoch in range(10):
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss_gpt2(model, ref, tok, prompt, chosen, rejected)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
    model.eval()
    correct = 0
    for prompt, expected in EVAL_PAIRS:
        inp = tok(prompt, return_tensors='pt')
        exp_id = tok.encode(expected)[0]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if logits.argmax().item() == exp_id: correct += 1
    return correct / len(EVAL_PAIRS)


def main():
    print("[P143] GPT-2 L2 Universality Test (CPU)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    model_id = 'gpt2'
    tok = GPT2Tokenizer.from_pretrained(model_id, local_files_only=True)

    # Measure baseline geometry
    model_tmp = GPT2LMHeadModel.from_pretrained(model_id, local_files_only=True).eval()
    l2_base, cos_base, _ = measure_geometry_gpt2(model_tmp, tok)
    d = model_tmp.config.n_embd
    print(f"  GPT-2 baseline: d={d}, L2={l2_base:.3f}, cos={cos_base:.3f}")
    del model_tmp; gc.collect()

    # Scale sweep
    scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = []

    for sf in scale_factors:
        print(f"\n  === scale = {sf:.1f}x (GPT-2, d={d}) ===")
        model = GPT2LMHeadModel.from_pretrained(model_id, local_files_only=True)
        ref = GPT2LMHeadModel.from_pretrained(model_id, local_files_only=True).eval()
        gram_schmidt_gpt2(model, tok)
        gram_schmidt_gpt2(ref, tok)
        if sf != 1.0:
            scale_norms_gpt2(model, tok, sf)
            scale_norms_gpt2(ref, tok, sf)
        l2, cos, norm = measure_geometry_gpt2(model, tok)
        print(f"    L2={l2:.3f}, cos={cos:.4f}")
        acc = train_eval_gpt2(model, ref, tok)
        print(f"    DPO num acc = {acc:.0%}")
        results.append({'scale': sf, 'l2': l2, 'cos': cos, 'norm': norm, 'acc': acc, 'd': d})
        del model, ref; gc.collect()

    # Find L2*
    l2_star_gpt2 = None
    for r in results:
        if r['acc'] > 0:
            l2_star_gpt2 = r['l2']
            break

    # Compare with Qwen predictions
    l2_star_qwen05 = 1.25  # From P138c (d=896)
    predicted_gpt2 = l2_star_qwen05 * (d / 896) ** 0.5  # sqrt(d) prediction

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase143_gpt2_universality.json'), 'w') as f:
        json.dump({
            'phase': '143', 'name': 'GPT-2 L2 Universality',
            'd': d, 'baseline_cos': cos_base, 'baseline_l2': l2_base,
            'l2_star_gpt2': l2_star_gpt2,
            'predicted_from_qwen': predicted_gpt2,
            'results': results
        }, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    l2_vals = [r['l2'] for r in results]
    acc_vals = [r['acc'] for r in results]

    # Left: L2 threshold for GPT-2
    ax = axes[0]
    ax.plot(l2_vals, acc_vals, 'g-o', lw=2.5, markersize=10, label=f'GPT-2 (d={d})')
    ax.fill_between(l2_vals, acc_vals, alpha=0.1, color='green')
    if l2_star_gpt2:
        ax.axvline(x=l2_star_gpt2, color='red', ls='--', lw=2,
                  label=f'L2* GPT-2 = {l2_star_gpt2:.2f}')
    ax.axvline(x=1.25, color='orange', ls=':', lw=2,
              label='L2* Qwen-0.5B = 1.25')
    ax.axvline(x=predicted_gpt2, color='blue', ls=':', lw=2,
              label=f'sqrt(d) prediction = {predicted_gpt2:.2f}')
    for i, r in enumerate(results):
        ax.annotate(f'{r["acc"]:.0%}', (l2_vals[i], acc_vals[i]),
                   fontsize=9, ha='center', va='bottom')
    ax.set_xlabel('Mean Pairwise L2 Distance', fontsize=12)
    ax.set_ylabel('DPO Number Accuracy', fontsize=12)
    ax.set_title('GPT-2 Small L2 Threshold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # Right: Cross-architecture comparison
    ax = axes[1]
    models_d = [768, 896]  # GPT-2, Qwen-0.5B
    models_l2star = [l2_star_gpt2 if l2_star_gpt2 else 0, 1.25]
    models_names = ['GPT-2\n(124M)', 'Qwen-0.5B\n(494M)']
    colors = ['green', 'orange']
    for i, (dd, ls, name, c) in enumerate(zip(models_d, models_l2star, models_names, colors)):
        if ls > 0:
            ax.scatter([dd], [ls], s=200, c=c, marker='os'[i], zorder=5, label=name)
    d_range = np.linspace(500, 2000, 100)
    if l2_star_gpt2:
        l2_fit = l2_star_gpt2 * (d_range / d) ** 0.5
        ax.plot(d_range, l2_fit, 'g--', lw=2, alpha=0.5, label=r'$L2^* \propto \sqrt{d}$ (GPT-2)')
    l2_fit_qwen = 1.25 * (d_range / 896) ** 0.5
    ax.plot(d_range, l2_fit_qwen, 'orange', ls='--', lw=2, alpha=0.5, label=r'$L2^* \propto \sqrt{d}$ (Qwen)')
    ax.set_xlabel('Hidden Dimension (d)', fontsize=12)
    ax.set_ylabel('Critical L2 Distance (L2*)', fontsize=12)
    ax.set_title('Cross-Architecture L2* Scaling', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 143: GPT-2 L2 Universality Test',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase143_gpt2_universality.png'), dpi=150,
               bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    if l2_star_gpt2:
        ratio = l2_star_gpt2 / 1.25
        pred_ratio = (d / 896) ** 0.5
        print(f"  L2* GPT-2 = {l2_star_gpt2:.2f}, Qwen-0.5B = 1.25, ratio = {ratio:.2f}")
        print(f"  sqrt(d) prediction ratio = {pred_ratio:.2f}")
        if abs(ratio - pred_ratio) / pred_ratio < 0.3:
            print("  -> L2* SCALES WITH sqrt(d) - UNIVERSAL LAW CONFIRMED!")
        elif abs(l2_star_gpt2 - 1.25) / 1.25 < 0.2:
            print("  -> L2* is ARCHITECTURE-INDEPENDENT (~1.2 for both)")
        else:
            print("  -> L2* differs between architectures")
    else:
        print("  -> GPT-2 DPO never activated (all 0%)")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 143] Complete.")

if __name__ == '__main__':
    main()
