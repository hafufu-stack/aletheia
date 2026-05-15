# -*- coding: utf-8 -*-
"""
Phase 138b: Distance vs Angle - Why Does Gram-Schmidt Alone Fail?

P138 showed:
  - Dispersion (cos=0.07): num=25%
  - Gram-Schmidt (cos=0.00): num=0%  <-- WORSE despite better cos!
  - GS+Dispersion (cos=-0.07): num=50%  <-- BEST

Hypothesis: DPO needs BOTH angular separation AND absolute L2 distance.
Gram-Schmidt makes vectors perpendicular but preserves their norms,
so they stay in the same "neighborhood". Dispersion pushes them far apart.

This experiment measures L2 distances and tests the Distance Hypothesis.

Model: Qwen2.5-0.5B (GPU, float32)
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

DPO_PAIRS = [
    ("Water freezes at", " 0", " 100"),
    ("The boiling point of water is", " 100", " 212"),
    ("The atomic number of carbon is", " 6", " 12"),
    ("The speed of light is approximately", " 299", " 186"),
]

NUM_TOKENS = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]


def measure_geometry(model, tok):
    """Measure both cosine similarity AND L2 distance of number embeddings."""
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in NUM_TOKENS]
    vecs = embed[ids].float()

    # Cosine similarity
    vecs_norm = F.normalize(vecs, dim=-1)
    cos_matrix = vecs_norm @ vecs_norm.T
    mask = ~torch.eye(len(ids), dtype=bool, device=cos_matrix.device)
    mean_cos = cos_matrix[mask].mean().item()

    # L2 distance
    dists = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0)).squeeze(0)
    mean_l2 = dists[mask].mean().item()

    # Norms
    norms = vecs.norm(dim=-1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()

    return mean_cos, mean_l2, mean_norm, std_norm


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


def disperse(model, tok, strength=1.0):
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in NUM_TOKENS]
    vecs = embed[ids].clone()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()


def scale_norms(model, tok, factor=2.0):
    """Scale number token norms without changing direction."""
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in NUM_TOKENS]
    for idx in ids:
        embed[idx] *= factor


def train_eval(model, ref, tok, n_layers=24):
    boundary = int(n_layers * 0.94)
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-6)
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
    print("[P138b] Distance vs Angle: Why Gram-Schmidt Alone Fails")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    configs = [
        ('baseline', lambda m, t: None),
        ('disperse_s1', lambda m, t: disperse(m, t, 1.0)),
        ('gram_schmidt', lambda m, t: gram_schmidt(m, t)),
        ('gs_disperse', lambda m, t: (gram_schmidt(m, t), disperse(m, t, 0.5))),
        ('gs_scale2x', lambda m, t: (gram_schmidt(m, t), scale_norms(m, t, 2.0))),
        ('gs_scale3x', lambda m, t: (gram_schmidt(m, t), scale_norms(m, t, 3.0))),
    ]

    results = {}
    for name, transform_fn in configs:
        print(f"\n  === {name} ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
        ref = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

        transform_fn(model, tok)
        transform_fn(ref, tok)

        cos, l2, norm, norm_std = measure_geometry(model, tok)
        num_acc = train_eval(model, ref, tok)

        print(f"    cos={cos:.4f} l2={l2:.2f} norm={norm:.2f}+/-{norm_std:.2f} num_acc={num_acc:.0%}")
        results[name] = {'cos': cos, 'l2': l2, 'norm': norm, 'norm_std': norm_std, 'num_acc': num_acc}

        del model, ref; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase138b_distance.json'), 'w') as f:
        json.dump({'phase': '138b', 'name': 'Distance vs Angle', 'results': results},
                 f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    names = list(results.keys())
    cos_vals = [results[n]['cos'] for n in names]
    l2_vals = [results[n]['l2'] for n in names]
    acc_vals = [results[n]['num_acc'] for n in names]

    ax = axes[0]
    ax.bar(names, acc_vals, color=['#e74c3c' if a == 0 else '#2ecc71' for a in acc_vals], alpha=0.8)
    for i, v in enumerate(acc_vals):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontweight='bold')
    ax.set_ylabel('Num Accuracy'); ax.set_title('DPO Number Accuracy', fontweight='bold')
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(0, 1.1)

    ax = axes[1]
    colors = ['#e74c3c' if acc_vals[i] == 0 else '#2ecc71' for i in range(len(names))]
    ax.scatter(cos_vals, acc_vals, c=colors, s=150, zorder=5)
    for i, n in enumerate(names):
        ax.annotate(n, (cos_vals[i], acc_vals[i]), fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('Cosine Similarity'); ax.set_ylabel('Num Accuracy')
    ax.set_title('Accuracy vs Cosine (Angle)', fontweight='bold')
    ax.axvline(x=0.3, color='gray', ls='--', alpha=0.5)

    ax = axes[2]
    ax.scatter(l2_vals, acc_vals, c=colors, s=150, zorder=5)
    for i, n in enumerate(names):
        ax.annotate(n, (l2_vals[i], acc_vals[i]), fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('L2 Distance'); ax.set_ylabel('Num Accuracy')
    ax.set_title('Accuracy vs L2 Distance', fontweight='bold')

    fig.suptitle('Phase 138b: Distance vs Angle - Why Gram-Schmidt Alone Fails',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase138b_distance.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 138b] Complete.")

if __name__ == '__main__':
    main()
