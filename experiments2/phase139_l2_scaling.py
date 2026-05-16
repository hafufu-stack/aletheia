# -*- coding: utf-8 -*-
"""
Phase 139: The High-Dimensional L2 Scaling Law
Does L2* scale with sqrt(d)?

P138c found L2* ~ 1.2 for Qwen-0.5B (d=896).
P137 showed 1.5B (d=1536) is immune even with surgery.
Hypothesis: L2* grows with sqrt(d), so 1.5B needs L2* ~ 1.2 * sqrt(1536/896) ~ 1.57

Experiment: Qwen-1.5B, GS + extreme norm scaling (1x to 8x)
to find L2* for the larger model.

Model: Qwen2.5-1.5B (GPU, 4-bit)
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_TOKENS = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]

DPO_PAIRS = [
    ("Water freezes at", " 0", " 100"),
    ("The boiling point of water is", " 100", " 212"),
    ("The atomic number of carbon is", " 6", " 12"),
    ("The speed of light is approximately", " 299", " 186"),
]


def dpo_loss(model, ref_model, tok, prompt, chosen, rejected, beta=0.05):
    text_c = prompt + chosen
    text_r = prompt + rejected
    inp_c = tok(text_c, return_tensors='pt').to(DEVICE)
    inp_r = tok(text_r, return_tensors='pt').to(DEVICE)
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


def gram_schmidt(model, tok):
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in NUM_TOKENS]
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


def scale_norms(model, tok, factor):
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in NUM_TOKENS]
    for idx in ids:
        embed[idx] *= factor


def measure_geometry(model, tok):
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in NUM_TOKENS]
    vecs = embed[ids].float()
    # L2 distance
    dists = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0)).squeeze(0)
    mask = ~torch.eye(len(ids), dtype=bool, device=dists.device)
    l2 = dists[mask].mean().item()
    # Cosine
    norms = vecs.norm(dim=-1, keepdim=True)
    cos_mat = (vecs @ vecs.T) / (norms @ norms.T + 1e-8)
    cos_val = cos_mat[mask].mean().item()
    # Mean norm
    mean_norm = norms.mean().item()
    return l2, cos_val, mean_norm


def train_eval(model, ref, tok, n_layers=28):
    """Train DPO and evaluate on 1.5B (28 layers)."""
    boundary = int(n_layers * 0.94)
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print("    WARNING: No trainable params found!")
        return 0.0
    opt = torch.optim.AdamW(trainable, lr=2e-5)  # 1.5B uses higher lr (P124)
    for epoch in range(5):
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss(model, ref, tok, prompt, chosen, rejected)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
    model.eval()
    correct = 0
    for prompt, chosen, rejected in DPO_PAIRS:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        exp_id = tok.encode(chosen)[-1]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if logits.argmax().item() == exp_id: correct += 1
    return correct / len(DPO_PAIRS)


def main():
    print("[P139] High-Dimensional L2 Scaling Law")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-1.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Scale factors: aggressive range for 1.5B
    scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    results = []

    # Also measure 0.5B baseline for comparison
    model_05b = 'Qwen/Qwen2.5-0.5B'
    m05 = AutoModelForCausalLM.from_pretrained(
        model_05b, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    tok05 = AutoTokenizer.from_pretrained(model_05b, local_files_only=True)
    l2_05, cos_05, norm_05 = measure_geometry(m05, tok05)
    d_05 = m05.config.hidden_size
    print(f"  0.5B baseline: d={d_05}, L2={l2_05:.3f}, cos={cos_05:.3f}")
    del m05, tok05; gc.collect(); torch.cuda.empty_cache()

    for sf in scale_factors:
        print(f"\n  === scale = {sf:.1f}x (1.5B) ===")
        # Use float16 (not 4-bit) so DPO can set requires_grad
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float16).to(DEVICE)
        ref = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float16).eval().to(DEVICE)
        ref.eval()
        gram_schmidt(model, tok)
        gram_schmidt(ref, tok)
        if sf != 1.0:
            scale_norms(model, tok, sf)
            scale_norms(ref, tok, sf)
        l2, cos, norm = measure_geometry(model, tok)
        d_15 = model.config.hidden_size
        print(f"    d={d_15}, L2={l2:.3f}, cos={cos:.4f}, norm={norm:.4f}")
        acc = train_eval(model, ref, tok, n_layers=28)
        print(f"    DPO num acc = {acc:.0%}")
        results.append({
            'scale': sf, 'l2': l2, 'cos': cos, 'norm': norm,
            'acc': acc, 'd': d_15
        })
        del model, ref; gc.collect(); torch.cuda.empty_cache()

    # Compute predicted L2* from sqrt(d) scaling
    l2_star_05 = 1.25  # From P138c
    d_15 = results[0]['d']
    predicted_l2_star = l2_star_05 * (d_15 / d_05) ** 0.5
    print(f"\n  sqrt(d) prediction: L2*_1.5B = {l2_star_05} * sqrt({d_15}/{d_05}) = {predicted_l2_star:.2f}")

    # Find actual transition
    actual_l2_star = None
    for r in results:
        if r['acc'] > 0:
            actual_l2_star = r['l2']
            break

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase139_l2_scaling.json'), 'w') as f:
        json.dump({
            'phase': '139', 'name': 'High-Dimensional L2 Scaling Law',
            'd_05': d_05, 'l2_star_05': l2_star_05,
            'd_15': d_15, 'predicted_l2_star_15': predicted_l2_star,
            'actual_l2_star_15': actual_l2_star,
            'results': results
        }, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    l2_vals = [r['l2'] for r in results]
    acc_vals = [r['acc'] for r in results]

    # Left: L2 threshold curve for 1.5B
    ax = axes[0]
    ax.plot(l2_vals, acc_vals, 'b-o', lw=2.5, markersize=10, label='Qwen-1.5B')
    ax.fill_between(l2_vals, acc_vals, alpha=0.1, color='blue')
    if actual_l2_star:
        ax.axvline(x=actual_l2_star, color='red', ls='--', lw=2,
                  label=f'L2*_1.5B = {actual_l2_star:.2f}')
    ax.axvline(x=predicted_l2_star, color='green', ls=':', lw=2,
              label=f'sqrt(d) prediction = {predicted_l2_star:.2f}')
    ax.axvline(x=1.25, color='orange', ls=':', lw=2, alpha=0.7,
              label=f'L2*_0.5B = 1.25 (P138c)')
    for i, r in enumerate(results):
        ax.annotate(f'{r["acc"]:.0%}', (l2_vals[i], acc_vals[i]),
                   fontsize=9, ha='center', va='bottom')
    ax.set_xlabel('Mean Pairwise L2 Distance', fontsize=12)
    ax.set_ylabel('DPO Number Accuracy', fontsize=12)
    ax.set_title('1.5B L2 Critical Threshold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # Right: sqrt(d) scaling plot
    ax = axes[1]
    ax.scatter([d_05], [1.25], s=200, c='orange', marker='s', zorder=5, label='0.5B (P138c)')
    if actual_l2_star:
        ax.scatter([d_15], [actual_l2_star], s=200, c='blue', marker='o', zorder=5, label='1.5B (this)')
    # Fit line
    d_range = np.linspace(500, 2000, 100)
    l2_fit = 1.25 * (d_range / d_05) ** 0.5
    ax.plot(d_range, l2_fit, 'g--', lw=2, alpha=0.7, label=r'$L2^* \propto \sqrt{d}$')
    ax.set_xlabel('Hidden Dimension (d)', fontsize=12)
    ax.set_ylabel('Critical L2 Distance (L2*)', fontsize=12)
    ax.set_title(r'L2* Scaling: $L2^* \propto \sqrt{d}$ ?', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 139: High-Dimensional L2 Scaling Law',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase139_l2_scaling.png'), dpi=150,
               bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 139] Complete.")

if __name__ == '__main__':
    main()
