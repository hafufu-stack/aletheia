# -*- coding: utf-8 -*-
"""
Phase 185: Autopoietic FGA Chaining
P167: Teacher-forced chaining got 77% per-token.
P181: Final Oracle matches Teacher at 100% ratio.

Combine: Use BASE model's output at each step as FGA target
for SURGERY model, autoregressively. No ground truth needed.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time
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
    ("# Light speed in km/s is", " 299"),
    ("# A circle has", " 360"),
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

def autopoietic_chain(base, surg, tok, prompt, n_tokens, fga_gain=5, fga_layer=None):
    """Generate n_tokens using base->surg oracle at each step."""
    n_layers = surg.config.num_hidden_layers
    if fga_layer is None:
        fga_layer = n_layers - max(1, n_layers // 4)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    base_ids = inp['input_ids'].clone()
    surg_ids = inp['input_ids'].clone()
    generated = []
    base_preds = []

    for step in range(n_tokens):
        # Step 1: BASE model predicts next token
        with torch.no_grad():
            base_logits = base(input_ids=base_ids).logits[0, -1, :].float()
        base_pred_id = base_logits.argmax().item()
        base_preds.append(base_pred_id)

        # Step 2: SURGERY + FGA toward base prediction
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
        surg_pred_id = surg_logits.argmax().item()
        generated.append(surg_pred_id)

        # Extend context for BOTH models
        base_ids = torch.cat([base_ids, torch.tensor([[surg_pred_id]], device=DEVICE)], dim=1)
        surg_ids = torch.cat([surg_ids, torch.tensor([[surg_pred_id]], device=DEVICE)], dim=1)

    return generated, base_preds

def main():
    print("[P185] Autopoietic FGA Chaining")
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
    for fga_gain in [0, 3, 5, 10]:
        print(f"\n  === FGA Gain = {fga_gain} ===")
        per_token_correct = 0
        per_token_total = 0
        full_correct = 0
        details = []

        for prompt, expected in MULTI_TOKEN:
            exp_tokens = tok.encode(expected)
            n_tokens = len(exp_tokens)

            if fga_gain == 0:
                # Baseline: base model only
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                gen_ids = []
                ids = inp['input_ids'].clone()
                for _ in range(n_tokens):
                    with torch.no_grad():
                        logits = base(input_ids=ids).logits[0, -1, :].float()
                    pred_id = logits.argmax().item()
                    gen_ids.append(pred_id)
                    ids = torch.cat([ids, torch.tensor([[pred_id]], device=DEVICE)], dim=1)
                generated = gen_ids
                base_preds = gen_ids
            else:
                generated, base_preds = autopoietic_chain(
                    base, surg, tok, prompt, n_tokens, fga_gain=fga_gain)

            gen_text = tok.decode(generated)
            # Per-token accuracy
            tok_correct = sum(1 for g, e in zip(generated, exp_tokens) if g == e)
            per_token_correct += tok_correct
            per_token_total += n_tokens
            is_full = gen_text.strip().startswith(expected.strip())
            if is_full: full_correct += 1

            safe_exp = expected.strip()
            safe_gen = gen_text.strip()[:15]
            print(f"    {safe_exp:>8s} -> {safe_gen:15s} tok={tok_correct}/{n_tokens} "
                  f"{'FULL' if is_full else 'MISS'}")
            details.append({'expected': expected, 'generated': gen_text[:20],
                            'tok_acc': tok_correct/n_tokens, 'full': is_full})

        per_tok_acc = per_token_correct / max(1, per_token_total)
        full_acc = full_correct / len(MULTI_TOKEN)
        results[f'g{fga_gain}'] = {'gain': fga_gain, 'per_token': per_tok_acc,
                                    'full': full_acc, 'details': details}
        print(f"  Summary: per_token={per_tok_acc:.0%} full={full_acc:.0%}")

    with open(os.path.join(RESULTS_DIR, 'phase185_autopoietic_chain.json'), 'w') as f:
        json.dump({'phase': '185', 'name': 'Autopoietic FGA Chaining',
                   'results': results}, f, indent=2, default=str)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    gains = [0, 3, 5, 10]
    ax = axes[0]
    pt = [results[f'g{g}']['per_token'] for g in gains]
    ax.bar([str(g) for g in gains], pt, color='#3498db', alpha=0.8)
    for i, v in enumerate(pt):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=14, fontweight='bold')
    ax.set_xlabel('FGA Gain', fontsize=12); ax.set_ylabel('Per-Token Accuracy', fontsize=12)
    ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Per-Token (Autopoietic)', fontsize=13, fontweight='bold')

    ax = axes[1]
    fa = [results[f'g{g}']['full'] for g in gains]
    ax.bar([str(g) for g in gains], fa, color='#e74c3c', alpha=0.8)
    for i, v in enumerate(fa):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=14, fontweight='bold')
    ax.set_xlabel('FGA Gain', fontsize=12); ax.set_ylabel('Full Match Accuracy', fontsize=12)
    ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Full Match (Autopoietic)', fontsize=13, fontweight='bold')

    plt.suptitle('Phase 185: Autopoietic FGA Chaining\n'
                 'BASE predicts, SURGERY+FGA amplifies, no ground truth',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase185_autopoietic_chain.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    best_g = max(gains, key=lambda g: results[f'g{g}']['per_token'])
    print(f"\n  === VERDICT ===")
    print(f"  -> Best gain: g={best_g}")
    print(f"  -> Per-token: {results[f'g{best_g}']['per_token']:.0%}")
    print(f"  -> Full match: {results[f'g{best_g}']['full']:.0%}")
    print(f"  -> P167 teacher per-token: 79%")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 185] Complete.")
    del base, surg; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
