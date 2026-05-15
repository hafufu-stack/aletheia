# -*- coding: utf-8 -*-
"""
Phase 138c: L2 Critical Threshold - Precision Measurement

P138b proved DPO needs L2 distance, not just cosine angle.
With Gram-Schmidt (cos=0 fixed):
  L2=0.71 (scale 1x): 0%
  L2=1.43 (scale 2x): 25%
  L2=2.14 (scale 3x): 50%

This phase does a fine-grained sweep to find the EXACT
critical L2 distance L2* where DPO transitions from 0% to >0%.

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


def measure_l2(model, tok):
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in NUM_TOKENS]
    vecs = embed[ids].float()
    dists = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0)).squeeze(0)
    mask = ~torch.eye(len(ids), dtype=bool, device=dists.device)
    return dists[mask].mean().item()


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
    print("[P138c] L2 Critical Threshold - Precision Sweep")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Fine-grained scale sweep: 1.0 to 5.0
    scale_factors = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    results = []

    for sf in scale_factors:
        print(f"\n  === scale = {sf:.2f}x ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
        ref = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
        gram_schmidt(model, tok)
        gram_schmidt(ref, tok)
        if sf != 1.0:
            scale_norms(model, tok, sf)
            scale_norms(ref, tok, sf)
        l2 = measure_l2(model, tok)
        acc = train_eval(model, ref, tok)
        print(f"    L2={l2:.3f} acc={acc:.0%}")
        results.append({'scale': sf, 'l2': l2, 'acc': acc})
        del model, ref; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase138c_l2_threshold.json'), 'w') as f:
        json.dump({'phase': '138c', 'name': 'L2 Critical Threshold', 'results': results},
                 f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    l2_vals = [r['l2'] for r in results]
    acc_vals = [r['acc'] for r in results]
    ax.plot(l2_vals, acc_vals, 'r-o', lw=2.5, markersize=10, zorder=5)
    ax.fill_between(l2_vals, acc_vals, alpha=0.15, color='red')

    # Find transition point
    for i in range(len(acc_vals)):
        if acc_vals[i] > 0:
            l2_star = l2_vals[i]
            ax.axvline(x=l2_star, color='green', ls='--', lw=2, alpha=0.7,
                      label=f'L2* = {l2_star:.2f} (transition)')
            break

    for i, r in enumerate(results):
        ax.annotate(f'{r["acc"]:.0%}\n(s={r["scale"]:.1f}x)',
                   (l2_vals[i], acc_vals[i]), fontsize=8, ha='center',
                   va='bottom' if acc_vals[i] < 0.4 else 'top')

    ax.set_xlabel('Mean Pairwise L2 Distance', fontsize=13)
    ax.set_ylabel('DPO Number Accuracy', fontsize=13)
    ax.set_title('Phase 138c: The L2 Critical Threshold\n'
                'Gram-Schmidt (cos=0 fixed) + Norm Scaling',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase138c_l2_threshold.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 138c] Complete.")

if __name__ == '__main__':
    main()
