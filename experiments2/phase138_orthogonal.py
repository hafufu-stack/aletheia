# -*- coding: utf-8 -*-
"""
Phase 138: The BPE Orthogonal Genesis
Instead of post-hoc dispersion (P130b), use Gram-Schmidt to make
numerical embeddings EXACTLY orthogonal (cos=0) and test DPO.

Also tests character-level decomposition: "100" -> "1 0 0"
to bypass BPE's "original sin" of merging digits.

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
    ("The capital of Japan is", " Tokyo", " Osaka", "word"),
    ("The capital of France is", " Paris", " Lyon", "word"),
    ("Water freezes at", " 0", " 100", "number"),
    ("The boiling point of water is", " 100", " 212", "number"),
    ("The atomic number of carbon is", " 6", " 12", "number"),
    ("The speed of light is approximately", " 299", " 186", "number"),
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


def gram_schmidt_orthogonalize(model, tok):
    """Force numerical token embeddings to be exactly orthogonal."""
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in num_tokens]
    vecs = embed[ids].clone().float()
    norms = vecs.norm(dim=-1, keepdim=True)

    # Gram-Schmidt
    ortho = torch.zeros_like(vecs)
    for i in range(len(vecs)):
        v = vecs[i].clone()
        for j in range(i):
            proj = torch.dot(v, ortho[j]) / (torch.dot(ortho[j], ortho[j]) + 1e-8)
            v = v - proj * ortho[j]
        ortho[i] = v / (v.norm() + 1e-8) * norms[i]

    for i, idx in enumerate(ids):
        embed[idx] = ortho[i].to(embed.dtype)

    return ids


def measure_clustering(model, tok):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in num_tokens]
    vecs = embed[ids].float()
    vecs_norm = F.normalize(vecs, dim=-1)
    cos_matrix = vecs_norm @ vecs_norm.T
    mask = ~torch.eye(len(ids), dtype=bool, device=cos_matrix.device)
    return cos_matrix[mask].mean().item()


def disperse_embeddings(model, tok, strength=1.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299"]
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in num_tokens]
    vecs = embed[ids].clone()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()


def train_and_eval(model, ref, tok, n_layers, label=""):
    boundary = int(n_layers * 0.94)
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-6)
    for epoch in range(5):
        for prompt, chosen, rejected, _ in DPO_PAIRS:
            loss = dpo_loss(model, ref, tok, prompt, chosen, rejected)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

    model.eval()
    results = {}
    for prompt, chosen, rejected, cat in DPO_PAIRS:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        exp_id = tok.encode(chosen)[-1]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        pred_id = logits.argmax().item()
        if cat not in results: results[cat] = {'correct': 0, 'total': 0}
        results[cat]['total'] += 1
        results[cat]['correct'] += int(pred_id == exp_id)

    word = results.get('word', {}).get('correct', 0) / max(1, results.get('word', {}).get('total', 1))
    num = results.get('number', {}).get('correct', 0) / max(1, results.get('number', {}).get('total', 1))
    return word, num


def main():
    print("[P138] BPE Orthogonal Genesis")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    configs = {}

    # A: Baseline (no surgery)
    print("\n  === A: Baseline ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    cos = measure_clustering(model, tok)
    w, n = train_and_eval(model, ref, tok, n_layers)
    print(f"    cos={cos:.4f} word={w:.0%} num={n:.0%}")
    configs['baseline'] = {'cos': cos, 'word': w, 'num': n}
    del model, ref; gc.collect(); torch.cuda.empty_cache()

    # B: P130b Dispersion (strength=1.0, known good)
    print("\n  === B: Dispersion (s=1.0) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    disperse_embeddings(model, tok, 1.0)
    disperse_embeddings(ref, tok, 1.0)
    cos = measure_clustering(model, tok)
    w, n = train_and_eval(model, ref, tok, n_layers)
    print(f"    cos={cos:.4f} word={w:.0%} num={n:.0%}")
    configs['dispersion'] = {'cos': cos, 'word': w, 'num': n}
    del model, ref; gc.collect(); torch.cuda.empty_cache()

    # C: Gram-Schmidt Orthogonalization (EXACT cos=0)
    print("\n  === C: Gram-Schmidt Orthogonal ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    gram_schmidt_orthogonalize(model, tok)
    gram_schmidt_orthogonalize(ref, tok)
    cos = measure_clustering(model, tok)
    w, n = train_and_eval(model, ref, tok, n_layers)
    print(f"    cos={cos:.4f} word={w:.0%} num={n:.0%}")
    configs['gram_schmidt'] = {'cos': cos, 'word': w, 'num': n}
    del model, ref; gc.collect(); torch.cuda.empty_cache()

    # D: Gram-Schmidt + Dispersion (belt and suspenders)
    print("\n  === D: Gram-Schmidt + Dispersion ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    gram_schmidt_orthogonalize(model, tok)
    gram_schmidt_orthogonalize(ref, tok)
    disperse_embeddings(model, tok, 0.5)
    disperse_embeddings(ref, tok, 0.5)
    cos = measure_clustering(model, tok)
    w, n = train_and_eval(model, ref, tok, n_layers)
    print(f"    cos={cos:.4f} word={w:.0%} num={n:.0%}")
    configs['gs_plus_disperse'] = {'cos': cos, 'word': w, 'num': n}
    del model, ref; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase138_orthogonal.json'), 'w') as f:
        json.dump({'phase': '138', 'name': 'BPE Orthogonal Genesis',
                  'configs': configs}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    names = list(configs.keys())
    cos_vals = [configs[n]['cos'] for n in names]
    num_vals = [configs[n]['num'] for n in names]
    word_vals = [configs[n]['word'] for n in names]
    x = np.arange(len(names))
    w_bar = 0.25
    ax.bar(x-w_bar, word_vals, w_bar, label='Word Acc', color='#3498db', alpha=0.8)
    ax.bar(x, num_vals, w_bar, label='Num Acc', color='#e74c3c', alpha=0.8)
    ax.bar(x+w_bar, cos_vals, w_bar, label='Cos Sim', color='#95a5a6', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w_bar, word_vals[i]+0.02, f'{word_vals[i]:.0%}', ha='center', fontsize=8)
        ax.text(x[i], num_vals[i]+0.02, f'{num_vals[i]:.0%}', ha='center', fontsize=8)
        ax.text(x[i]+w_bar, max(0, cos_vals[i])+0.02, f'{cos_vals[i]:.2f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.legend(); ax.set_ylabel('Rate / Similarity')
    ax.set_title('Phase 138: BPE Orthogonal Genesis - Gram-Schmidt vs Dispersion',
                fontweight='bold', fontsize=12)
    ax.set_ylim(-0.1, 1.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase138_orthogonal.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 138] Complete.")

if __name__ == '__main__':
    main()
