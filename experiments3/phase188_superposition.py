# -*- coding: utf-8 -*-
"""
Phase 188: Superposition FGA
P185 failed because Oracle's single top-1 prediction was wrong,
and FGA amplified the error. P186 proved tokens are orthogonal.

Solution: Use probability-weighted sum of top-K token vectors as FGA direction.
When model is confident: direction ~= single correct vector (strong).
When uncertain: orthogonal vectors cancel out (zero energy = safety).

Model: Qwen2.5-0.5B (GPU)
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

MULTI_TOKEN = [
    ("# The year Columbus reached America is", " 1492"),
    ("# Mount Fuji is", " 3776"),
    ("# The year the Moon landing was", " 1969"),
    ("# The year WWII ended is", " 1945"),
    ("# A circle has", " 360"),
    ("# Light speed in km/s is", " 299"),
]

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"," 299"]

def apply_surgery(model, tok, strength=2.0):
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)

def superposition_chain(base, surg, tok, prompt, n_tokens, fga_gain=5,
                        top_k=5, fga_layer=None):
    n_layers = surg.config.num_hidden_layers
    if fga_layer is None:
        fga_layer = n_layers - max(1, n_layers // 4)
    base_ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    surg_ids = base_ids.clone()
    generated = []

    for step in range(n_tokens):
        # BASE: get probability distribution
        with torch.no_grad():
            base_logits = base(input_ids=base_ids).logits[0, -1, :].float()
        probs = F.softmax(base_logits, dim=-1)
        topk_probs, topk_ids = probs.topk(top_k)
        # Normalize top-k probs
        topk_probs = topk_probs / topk_probs.sum()
        # Superposition: probability-weighted sum of unembed vectors
        direction = torch.zeros(surg.lm_head.weight.shape[1], device=DEVICE)
        for p, tid in zip(topk_probs, topk_ids):
            vec = surg.lm_head.weight.data[tid.item()].float()
            direction += p.item() * vec
        dir_norm = direction.norm()
        if dir_norm > 1e-8:
            direction = direction / dir_norm
        # FGA with superposition direction
        def mk(d, g):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].float()
                    if h.dim() == 3: h[:, -1, :] += g * d.to(h.device)
                    return (h.to(output[0].dtype),) + output[1:]
                return output
            return fn
        handle = surg.model.layers[fga_layer].register_forward_hook(mk(direction, fga_gain))
        with torch.no_grad():
            surg_logits = surg(input_ids=surg_ids).logits[0, -1, :].float()
        handle.remove()
        pred_id = surg_logits.argmax().item()
        generated.append(pred_id)
        base_ids = torch.cat([base_ids, torch.tensor([[pred_id]], device=DEVICE)], dim=1)
        surg_ids = torch.cat([surg_ids, torch.tensor([[pred_id]], device=DEVICE)], dim=1)

    return generated

def main():
    print("[P188] Superposition FGA")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    surg = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(surg, tok, strength=2.0)

    results = {}
    for method, top_k, gain in [
        ('baseline', 1, 0), ('top1_g5', 1, 5), ('top5_g5', 5, 5),
        ('top10_g5', 10, 5), ('top5_g10', 5, 10), ('top10_g10', 10, 10)
    ]:
        print(f"\n  === {method} (K={top_k}, g={gain}) ===")
        per_tok = 0; per_tot = 0; full_ok = 0
        details = []
        for prompt, expected in MULTI_TOKEN:
            exp_tokens = tok.encode(expected)
            n_tok = len(exp_tokens)
            if gain == 0:
                ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
                gen = []
                for _ in range(n_tok):
                    with torch.no_grad():
                        logits = base(input_ids=ids).logits[0, -1, :].float()
                    pred = logits.argmax().item()
                    gen.append(pred)
                    ids = torch.cat([ids, torch.tensor([[pred]], device=DEVICE)], dim=1)
            else:
                gen = superposition_chain(base, surg, tok, prompt, n_tok, gain, top_k)
            gen_text = tok.decode(gen)
            tok_ok = sum(1 for g, e in zip(gen, exp_tokens) if g == e)
            per_tok += tok_ok; per_tot += n_tok
            is_full = gen_text.strip().startswith(expected.strip())
            if is_full: full_ok += 1
            print(f"    {expected.strip():>6s} -> {gen_text.strip()[:10]:10s} "
                  f"tok={tok_ok}/{n_tok} {'FULL' if is_full else 'MISS'}")
            details.append({'expected': expected, 'gen': gen_text[:15],
                            'tok_acc': tok_ok/n_tok, 'full': is_full})
        pt_acc = per_tok / max(1, per_tot)
        f_acc = full_ok / len(MULTI_TOKEN)
        results[method] = {'per_token': pt_acc, 'full': f_acc, 'top_k': top_k, 'gain': gain}
        print(f"  -> per_token={pt_acc:.0%} full={f_acc:.0%}")

    with open(os.path.join(RESULTS_DIR, 'phase188_superposition.json'), 'w') as f:
        json.dump({'phase': '188', 'name': 'Superposition FGA', 'results': results},
                  f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    methods = list(results.keys())
    pt_vals = [results[m]['per_token'] for m in methods]
    f_vals = [results[m]['full'] for m in methods]
    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x-w/2, pt_vals, w, label='Per-Token', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, f_vals, w, label='Full Match', color='#e74c3c', alpha=0.8)
    for i in range(len(methods)):
        ax.text(x[i]-w/2, pt_vals[i]+0.02, f'{pt_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i]+w/2, f_vals[i]+0.02, f'{f_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax.legend(fontsize=11); ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase 188: Superposition FGA\nProbability-weighted top-K as FGA direction',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase188_superposition.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    best_m = max(results, key=lambda m: results[m]['per_token'])
    print(f"\n  === VERDICT ===")
    print(f"  -> Best: {best_m} per_token={results[best_m]['per_token']:.0%}")
    print(f"  -> P185 top1 autopoietic: per_token=43%")
    if results[best_m]['per_token'] > 0.5:
        print("  -> SUPERPOSITION IMPROVES chaining!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 188] Complete.")
    del base, surg; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
