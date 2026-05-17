# -*- coding: utf-8 -*-
"""
Phase 191: Confidence-Gated FGA Chaining (Opus Addition)
P185: Autoregressive FGA made multi-token WORSE (error amplification).
P188: Superposition might help, but here's another approach:

Only apply FGA when BASE model is CONFIDENT (low entropy).
High entropy = uncertain = just use base prediction (no amplification).

This prevents amplifying mistakes while boosting correct predictions.

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

def compute_entropy(logits):
    probs = F.softmax(logits.float(), dim=-1)
    return -(probs * torch.log(probs + 1e-10)).sum().item()

def gated_chain(base, surg, tok, prompt, n_tokens, fga_gain=5,
                entropy_threshold=5.0, fga_layer=None):
    n_layers = surg.config.num_hidden_layers
    if fga_layer is None:
        fga_layer = n_layers - max(1, n_layers // 4)
    base_ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    surg_ids = base_ids.clone()
    generated = []
    fga_applied = []

    for step in range(n_tokens):
        with torch.no_grad():
            base_logits = base(input_ids=base_ids).logits[0, -1, :].float()
        entropy = compute_entropy(base_logits)
        base_pred_id = base_logits.argmax().item()

        if entropy < entropy_threshold:
            # CONFIDENT: apply FGA
            unembed = surg.lm_head.weight.data[base_pred_id].float()
            direction = unembed / (unembed.norm() + 1e-8)
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
            fga_applied.append(True)
        else:
            # UNCERTAIN: use base prediction only (no amplification)
            pred_id = base_pred_id
            fga_applied.append(False)

        generated.append(pred_id)
        base_ids = torch.cat([base_ids, torch.tensor([[pred_id]], device=DEVICE)], dim=1)
        surg_ids = torch.cat([surg_ids, torch.tensor([[pred_id]], device=DEVICE)], dim=1)

    return generated, fga_applied

def main():
    print("[P191] Confidence-Gated FGA Chaining")
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
    for thresh in [3.0, 5.0, 7.0, 10.0, 999.0]:  # 999 = always FGA
        label = f"t{thresh:.0f}" if thresh < 100 else "always_fga"
        print(f"\n  === Threshold={thresh} ({label}) ===")
        pt_ok = 0; pt_total = 0; full_ok = 0
        for prompt, expected in MULTI_TOKEN:
            exp_tokens = tok.encode(expected)
            n_tok = len(exp_tokens)
            gen, fga_flags = gated_chain(base, surg, tok, prompt, n_tok,
                                          fga_gain=5, entropy_threshold=thresh)
            gen_text = tok.decode(gen)
            tok_correct = sum(1 for g, e in zip(gen, exp_tokens) if g == e)
            pt_ok += tok_correct; pt_total += n_tok
            is_full = gen_text.strip().startswith(expected.strip())
            if is_full: full_ok += 1
            fga_count = sum(fga_flags)
            print(f"    {expected.strip():>6s} -> {gen_text.strip()[:10]:10s} "
                  f"tok={tok_correct}/{n_tok} fga={fga_count}/{n_tok} "
                  f"{'FULL' if is_full else 'MISS'}")
        pt_acc = pt_ok / max(1, pt_total)
        f_acc = full_ok / len(MULTI_TOKEN)
        results[label] = {'per_token': pt_acc, 'full': f_acc, 'threshold': thresh}
        print(f"  -> per_token={pt_acc:.0%} full={f_acc:.0%}")

    # Also run baseline (no FGA, no surgery, just base model)
    print(f"\n  === Baseline (base only) ===")
    pt_ok = 0; pt_total = 0; full_ok = 0
    for prompt, expected in MULTI_TOKEN:
        exp_tokens = tok.encode(expected)
        n_tok = len(exp_tokens)
        ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        gen = []
        for _ in range(n_tok):
            with torch.no_grad():
                logits = base(input_ids=ids).logits[0, -1, :].float()
            pred = logits.argmax().item()
            gen.append(pred)
            ids = torch.cat([ids, torch.tensor([[pred]], device=DEVICE)], dim=1)
        gen_text = tok.decode(gen)
        tok_correct = sum(1 for g, e in zip(gen, exp_tokens) if g == e)
        pt_ok += tok_correct; pt_total += n_tok
        is_full = gen_text.strip().startswith(expected.strip())
        if is_full: full_ok += 1
    results['baseline'] = {'per_token': pt_ok/max(1,pt_total), 'full': full_ok/len(MULTI_TOKEN)}
    print(f"  -> per_token={results['baseline']['per_token']:.0%} full={results['baseline']['full']:.0%}")

    with open(os.path.join(RESULTS_DIR, 'phase191_gated_chain.json'), 'w') as f:
        json.dump({'phase': '191', 'name': 'Confidence-Gated FGA Chaining',
                   'results': results}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    keys = ['baseline', 't3', 't5', 't7', 't10', 'always_fga']
    labels = ['Base\nOnly', 't=3\n(strict)', 't=5', 't=7', 't=10', 'Always\nFGA']
    pt = [results[k]['per_token'] for k in keys]
    fl = [results[k]['full'] for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x-w/2, pt, w, label='Per-Token', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, fl, w, label='Full Match', color='#e74c3c', alpha=0.8)
    for i in range(len(keys)):
        ax.text(x[i]-w/2, pt[i]+0.02, f'{pt[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i]+w/2, fl[i]+0.02, f'{fl[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11); ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase 191: Confidence-Gated FGA Chaining\nOnly amplify when confident',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase191_gated_chain.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    best_k = max(results, key=lambda k: results[k]['per_token'])
    print(f"\n  === VERDICT ===")
    print(f"  -> Best: {best_k} per_token={results[best_k]['per_token']:.0%}")
    print(f"  -> P185 always_fga: {results.get('always_fga', {}).get('per_token', 0):.0%}")
    print(f"  -> Baseline: {results['baseline']['per_token']:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 191] Complete.")
    del base, surg; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
