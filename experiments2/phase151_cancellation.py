# -*- coding: utf-8 -*-
"""
Phase 151: DPO Gradient Cancellation
WHY does DPO fail at 1.5B? Measure gradient interference.

Hypothesis: In 1.5B, gradients from different DPO pairs
cancel each other out (high-dimensional curse).

Compare gradient vectors between 0.5B (DPO works) and
1.5B (DPO fails) to find the physical reason.

Model: Qwen2.5-0.5B and 1.5B (GPU)
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


def dpo_loss(model, ref_model, tok, prompt, chosen, rejected, beta=0.05):
    text_c = prompt + chosen
    text_r = prompt + rejected
    inp_c = tok(text_c, return_tensors='pt').to(DEVICE)
    inp_r = tok(text_r, return_tensors='pt').to(DEVICE)
    plen = tok(prompt, return_tensors='pt').to(DEVICE)['input_ids'].shape[1]
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


def disperse_embeddings(model, tok, strength=1.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365"]
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in num_tokens))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)


def measure_gradients(model_id, model_name, tok, dtype=torch.float32):
    """Compute per-pair gradients and measure their interference."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=dtype).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=dtype).eval().to(DEVICE)
    disperse_embeddings(model, tok, strength=1.0)
    disperse_embeddings(ref, tok, strength=1.0)

    n_layers = model.config.num_hidden_layers
    boundary = int(n_layers * 0.94)

    # Set trainable params (same as in actual DPO)
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"    Trainable params: {n_params:,}")

    # Compute per-pair gradients
    pair_grads = []
    pair_losses = []
    for prompt, chosen, rejected in DPO_PAIRS:
        model.zero_grad()
        loss = dpo_loss(model, ref, tok, prompt, chosen, rejected)
        loss.backward()
        # Flatten all trainable gradients into one vector
        grad_vec = torch.cat([p.grad.detach().float().view(-1) for p in trainable])
        pair_grads.append(grad_vec.clone())
        pair_losses.append(loss.item())

    # Compute pairwise cosine similarities between gradient vectors
    n_pairs = len(pair_grads)
    cos_matrix = torch.zeros(n_pairs, n_pairs)
    for i in range(n_pairs):
        for j in range(n_pairs):
            cos_matrix[i, j] = F.cosine_similarity(
                pair_grads[i].unsqueeze(0), pair_grads[j].unsqueeze(0)).item()

    # Gradient norms
    norms = [g.norm().item() for g in pair_grads]

    # Net gradient (sum of all pair grads)
    net_grad = sum(pair_grads)
    net_norm = net_grad.norm().item()
    sum_norms = sum(norms)
    cancellation_ratio = net_norm / (sum_norms + 1e-10)

    # Projection of each gradient onto the net direction
    net_dir = net_grad / (net_grad.norm() + 1e-8)
    projections = [torch.dot(g, net_dir).item() for g in pair_grads]

    del model, ref, pair_grads, net_grad
    gc.collect(); torch.cuda.empty_cache()

    return {
        'model': model_name,
        'n_params': n_params,
        'cos_matrix': cos_matrix.tolist(),
        'norms': norms,
        'losses': pair_losses,
        'net_norm': net_norm,
        'sum_norms': sum_norms,
        'cancellation_ratio': cancellation_ratio,
        'projections': projections,
    }


def main():
    print("[P151] DPO Gradient Cancellation")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoTokenizer

    results = {}

    # 0.5B (where DPO works)
    print("\n  === 0.5B (DPO works) ===")
    tok_05 = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', local_files_only=True)
    if tok_05.pad_token is None: tok_05.pad_token = tok_05.eos_token
    r05 = measure_gradients('Qwen/Qwen2.5-0.5B', '0.5B', tok_05, torch.float32)
    print(f"    Grad norms: {[f'{n:.6f}' for n in r05['norms']]}")
    print(f"    Net norm: {r05['net_norm']:.6f}, Sum norms: {r05['sum_norms']:.6f}")
    print(f"    Cancellation ratio: {r05['cancellation_ratio']:.4f} (1.0=no cancel, 0=total cancel)")
    results['0.5B'] = r05

    # 1.5B (where DPO fails)
    print("\n  === 1.5B (DPO fails) ===")
    tok_15 = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', local_files_only=True)
    if tok_15.pad_token is None: tok_15.pad_token = tok_15.eos_token
    r15 = measure_gradients('Qwen/Qwen2.5-1.5B', '1.5B', tok_15, torch.float16)
    print(f"    Grad norms: {[f'{n:.6f}' for n in r15['norms']]}")
    print(f"    Net norm: {r15['net_norm']:.6f}, Sum norms: {r15['sum_norms']:.6f}")
    print(f"    Cancellation ratio: {r15['cancellation_ratio']:.4f}")
    results['1.5B'] = r15

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase151_cancellation.json'), 'w') as f:
        json.dump({'phase': '151', 'name': 'DPO Gradient Cancellation',
                   'results': results}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: Cosine similarity matrices
    for col, (name, r) in enumerate(results.items()):
        ax = axes[0, col]
        cos = np.array(r['cos_matrix'])
        im = ax.imshow(cos, cmap='RdBu_r', vmin=-1, vmax=1)
        pair_labels = [f"P{i+1}" for i in range(len(DPO_PAIRS))]
        ax.set_xticks(range(len(pair_labels))); ax.set_xticklabels(pair_labels)
        ax.set_yticks(range(len(pair_labels))); ax.set_yticklabels(pair_labels)
        for i in range(len(pair_labels)):
            for j in range(len(pair_labels)):
                ax.text(j, i, f'{cos[i,j]:.2f}', ha='center', va='center', fontsize=10)
        ax.set_title(f'{name}: Gradient Cosine Similarity\n'
                    f'Cancel ratio={r["cancellation_ratio"]:.3f}',
                    fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Bottom-left: Gradient norms comparison
    ax = axes[1, 0]
    x = np.arange(len(DPO_PAIRS))
    w = 0.35
    ax.bar(x-w/2, results['0.5B']['norms'], w, label='0.5B', color='#2ecc71', alpha=0.8)
    ax.bar(x+w/2, results['1.5B']['norms'], w, label='1.5B', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"Pair {i+1}" for i in range(len(DPO_PAIRS))])
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Per-Pair Gradient Norms', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

    # Bottom-right: Cancellation summary
    ax = axes[1, 1]
    names = list(results.keys())
    cancel_vals = [results[n]['cancellation_ratio'] for n in names]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(names, cancel_vals, color=colors, alpha=0.8, edgecolor='black', lw=1.5)
    for i, v in enumerate(cancel_vals):
        ax.text(i, v+0.02, f'{v:.3f}', ha='center', fontsize=14, fontweight='bold')
    ax.set_ylabel('|Sum of grads| / Sum of |grads|', fontsize=11)
    ax.set_title('Gradient Cancellation Ratio\n(1.0 = aligned, 0 = total cancellation)',
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.axhline(y=1.0, color='green', ls='--', alpha=0.5)
    ax.axhline(y=0.0, color='red', ls='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Phase 151: DPO Gradient Cancellation\nWhy DPO works at 0.5B but dies at 1.5B',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase151_cancellation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    c05 = results['0.5B']['cancellation_ratio']
    c15 = results['1.5B']['cancellation_ratio']
    if c15 < c05 * 0.5:
        print(f"  -> CANCELLATION CONFIRMED! 0.5B={c05:.3f}, 1.5B={c15:.3f}")
        print(f"     1.5B gradients cancel {c05/c15:.1f}x more than 0.5B")
    else:
        print(f"  -> Cancellation similar: 0.5B={c05:.3f}, 1.5B={c15:.3f}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 151] Complete.")

if __name__ == '__main__':
    main()
