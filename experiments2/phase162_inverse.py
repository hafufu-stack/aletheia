# -*- coding: utf-8 -*-
"""
Phase 162: Inverse Surgery (Causal Proof)
If dispersing numbers = better facts, then COMPRESSING them
should make hallucinations WORSE.

This provides causal proof: clustering -> hallucination.

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

TEST_SET = [
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
    ("The capital of France is", " Paris", "word"),
    ("The capital of Japan is", " Tokyo", "word"),
]


def dual_surgery(model, tok, strength):
    """Positive strength = disperse. Negative = COMPRESS (inverse)."""
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365"]
    ids = list(set(tok.encode(t)[-1] for t in num_tokens))
    # Embed
    embed = model.model.embed_tokens.weight.data
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()
    # LM head
    lm = model.lm_head.weight.data
    vecs_lm = lm[ids].clone().float()
    center_lm = vecs_lm.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs_lm[i] - center_lm
        direction = diff / (diff.norm() + 1e-8)
        lm[idx] += strength * direction * lm[idx].norm()
    # Measure post-surgery cos
    post_embed = embed[ids].float()
    e_cos = F.cosine_similarity(post_embed.unsqueeze(0), post_embed.unsqueeze(1), dim=-1)
    post_lm = lm[ids].float()
    l_cos = F.cosine_similarity(post_lm.unsqueeze(0), post_lm.unsqueeze(1), dim=-1)
    mask = ~torch.eye(len(ids), dtype=bool)
    return e_cos[mask].mean().item(), l_cos[mask].mean().item()


class FGAHook:
    def __init__(self, model, target_token_id, gain):
        self.gain = gain
        unembed = model.lm_head.weight.data[target_token_id].float()
        self.direction = unembed / (unembed.norm() + 1e-8)
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        else:
            h = output.float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return h.to(output.dtype)

    def register(self, model, layer_idx):
        self.handle = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def evaluate(model, tok, code_mode=True, fga_gain=20):
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - 2
    w_correct, n_correct, w_total, n_total = 0, 0, 0, 0
    for prompt, expected, cat in TEST_SET:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[-1]
        hook = FGAHook(model, exp_id, fga_gain)
        hook.register(model, fga_layer)
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        hook.remove()
        if cat == 'word':
            w_total += 1; w_correct += int(logits.argmax().item() == exp_id)
        else:
            n_total += 1; n_correct += int(logits.argmax().item() == exp_id)
    return w_correct/max(1,w_total), n_correct/max(1,n_total)


def eval_baseline(model, tok):
    """No code mode, no FGA - pure baseline."""
    w_correct, n_correct, w_total, n_total = 0, 0, 0, 0
    for prompt, expected, cat in TEST_SET:
        exp_id = tok.encode(expected)[-1]
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if cat == 'word':
            w_total += 1; w_correct += int(logits.argmax().item() == exp_id)
        else:
            n_total += 1; n_correct += int(logits.argmax().item() == exp_id)
    return w_correct/max(1,w_total), n_correct/max(1,n_total)


def main():
    print("[P162] Inverse Surgery (Causal Proof)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Sweep: from strong compression (negative) to strong dispersion (positive)
    strengths = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
    results = []

    for s in strengths:
        print(f"\n  === Strength = {s:+.1f} ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

        if s != 0:
            e_cos, l_cos = dual_surgery(model, tok, strength=s)
        else:
            # Measure baseline cos
            embed = model.model.embed_tokens.weight.data.float()
            lm = model.lm_head.weight.data.float()
            num_ids = [tok.encode(f" {d}")[-1] for d in range(10)]
            e_vecs = embed[num_ids]
            l_vecs = lm[num_ids]
            mask = ~torch.eye(10, dtype=bool)
            e_cos = F.cosine_similarity(e_vecs.unsqueeze(0), e_vecs.unsqueeze(1), dim=-1)[mask].mean().item()
            l_cos = F.cosine_similarity(l_vecs.unsqueeze(0), l_vecs.unsqueeze(1), dim=-1)[mask].mean().item()

        # Eval baseline (no S&S)
        bw, bn = eval_baseline(model, tok)
        # Eval with S&S
        sw, sn = evaluate(model, tok, code_mode=True, fga_gain=20)

        print(f"    embed_cos={e_cos:.4f}, lm_cos={l_cos:.4f}")
        print(f"    Baseline: word={bw:.0%} num={bn:.0%}")
        print(f"    +S&S:     word={sw:.0%} num={sn:.0%}")

        results.append({
            'strength': s, 'embed_cos': e_cos, 'lm_cos': l_cos,
            'baseline_word': bw, 'baseline_num': bn,
            'ss_word': sw, 'ss_num': sn
        })
        del model; gc.collect(); torch.cuda.empty_cache()

    with open(os.path.join(RESULTS_DIR, 'phase162_inverse.json'), 'w') as f:
        json.dump({'phase': '162', 'name': 'Inverse Surgery',
                   'results': results}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ss = [r['strength'] for r in results]
    # Left: cosine similarity vs strength
    ax = axes[0]
    ax.plot(ss, [r['embed_cos'] for r in results], 'b-o', lw=2, label='Embed cos')
    ax.plot(ss, [r['lm_cos'] for r in results], 'r-s', lw=2, label='LM_head cos')
    ax.set_xlabel('Surgery Strength (- = compress, + = disperse)', fontsize=11)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Embedding Clustering vs Surgery', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='gray', ls='--', alpha=0.5)

    # Right: accuracy vs strength
    ax = axes[1]
    ax.plot(ss, [r['ss_num'] for r in results], 'r-o', lw=2.5, markersize=10, label='Number (S&S)')
    ax.plot(ss, [r['ss_word'] for r in results], 'b-s', lw=2.5, markersize=10, label='Word (S&S)')
    ax.plot(ss, [r['baseline_num'] for r in results], 'r--^', lw=1.5, alpha=0.5, label='Number (base)')
    ax.fill_between(ss, 0, [r['ss_num'] for r in results], alpha=0.1, color='red')
    ax.set_xlabel('Surgery Strength (- = compress, + = disperse)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Causal Proof: Clustering -> Hallucination', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='gray', ls='--', alpha=0.5)
    ax.set_ylim(-0.05, 1.1)

    plt.suptitle('Phase 162: Inverse Surgery\nDoes COMPRESSING numbers make hallucinations WORSE?',
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase162_inverse.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    baseline_acc = next(r['ss_num'] for r in results if r['strength'] == 0)
    compress_acc = next(r['ss_num'] for r in results if r['strength'] == -2.0)
    disperse_acc = next(r['ss_num'] for r in results if r['strength'] == 2.0)
    if compress_acc < baseline_acc and disperse_acc > baseline_acc:
        print(f"  -> CAUSAL PROOF! compress={compress_acc:.0%} < base={baseline_acc:.0%} < disperse={disperse_acc:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 162] Complete.")

if __name__ == '__main__':
    main()
