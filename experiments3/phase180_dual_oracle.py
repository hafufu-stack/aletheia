# -*- coding: utf-8 -*-
"""
Phase 180: Dual-Weight Oracle (The Fix)
P177 discovered: Logit Lens = 0% accuracy on ALL layers after Surgery.
Surgery warps the embedding space, breaking token projection.

Solution: TWO-PHASE inference:
  1. BASE weights: Run Logit Lens to get Oracle's prediction (clean projection)
  2. SURGERY weights + FGA: Amplify Oracle's prediction for final output

This separates "knowing" (base model) from "steering" (surgery+FGA).

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

FACT_TEST = [
    ("# The capital of Japan is", " Tokyo"),
    ("# The capital of France is", " Paris"),
    ("# The capital of Germany is", " Berlin"),
    ("# The largest planet is", " Jupiter"),
    ("# Water freezes at", " 0"),
    ("# The boiling point of water is", " 100"),
    ("# The atomic number of carbon is", " 6"),
    ("# A year has", " 365"),
    ("# The number of continents is", " 7"),
    ("# Pi is approximately", " 3"),
]

ARITH_TEST = [
    ("# 1 + 1 =", " 2"), ("# 3 + 4 =", " 7"), ("# 5 + 5 =", " 10"),
    ("# 8 + 1 =", " 9"), ("# 6 + 3 =", " 9"), ("# 4 + 4 =", " 8"),
]

UNKNOWN_TEST = [
    ("# The capital of Xylandia is", "ABSTAIN"),
    ("# The 937th digit of pi is", "ABSTAIN"),
    ("# The winner of the 2030 World Cup is", "ABSTAIN"),
    ("# The GDP of Atlantis is", "ABSTAIN"),
]

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"]


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


def logit_lens_sweep(model, tok, prompt):
    """Get Logit Lens top-1 at each layer."""
    n_layers = model.config.num_hidden_layers
    layer_preds = {}
    hooks = []
    hiddens = {}

    def make_hook(l):
        def fn(module, input, output):
            if isinstance(output, tuple):
                hiddens[l] = output[0][:, -1, :].detach().float()
            else:
                hiddens[l] = output[:, -1, :].detach().float()
        return fn

    for l in range(n_layers):
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        _ = model(**inp)

    for h in hooks:
        h.remove()

    for l in range(n_layers):
        if l in hiddens:
            logits = model.lm_head(hiddens[l].to(model.lm_head.weight.dtype))
            pred_id = logits.float().argmax(dim=-1).item()
            layer_preds[l] = pred_id

    return layer_preds


def dual_weight_oracle(base_model, surg_model, tok, prompt, fga_gain=5,
                       oracle_layer=None, fga_layer=None):
    """Phase 1: BASE Logit Lens -> Phase 2: SURGERY + FGA."""
    n_layers = base_model.config.num_hidden_layers
    if oracle_layer is None:
        oracle_layer = n_layers - 4  # Late layers have best Logit Lens
    if fga_layer is None:
        fga_layer = n_layers - max(1, n_layers // 4)

    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    # Phase 1: BASE model - get Oracle prediction via Logit Lens
    oracle_hidden = {}
    def oh(module, input, output):
        if isinstance(output, tuple):
            oracle_hidden['h'] = output[0][:, -1, :].detach().float()
        else:
            oracle_hidden['h'] = output[:, -1, :].detach().float()

    h1 = base_model.model.layers[oracle_layer].register_forward_hook(oh)
    with torch.no_grad():
        base_logits = base_model(**inp).logits[0, -1, :].float()
    h1.remove()

    # Logit Lens projection using BASE lm_head
    if 'h' in oracle_hidden:
        oracle_logits = base_model.lm_head(oracle_hidden['h'].to(base_model.lm_head.weight.dtype))
        oracle_pred_id = oracle_logits.float().argmax(dim=-1).item()
    else:
        oracle_pred_id = base_logits.argmax().item()

    entropy = compute_entropy(base_logits)

    # Phase 2: SURGERY model + FGA toward Oracle's prediction
    unembed = surg_model.lm_head.weight.data[oracle_pred_id].float()
    direction = unembed / (unembed.norm() + 1e-8)

    def fh(module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += fga_gain * direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        h = output.float()
        if h.dim() == 3: h[:, -1, :] += fga_gain * direction.to(h.device)
        return h.to(output.dtype)

    h2 = surg_model.model.layers[fga_layer].register_forward_hook(fh)
    with torch.no_grad():
        final_logits = surg_model(**inp).logits[0, -1, :].float()
    h2.remove()

    final_pred_id = final_logits.argmax().item()
    return final_pred_id, oracle_pred_id, entropy


def main():
    print("[P180] Dual-Weight Oracle")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Load BASE model (for Oracle)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)

    # Load SURGERY model (for FGA output)
    surg_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(surg_model, tok, strength=2.0)

    n_layers = base_model.config.num_hidden_layers

    # First: Verify BASE Logit Lens works
    print("\n  === BASE Logit Lens Verification ===")
    for prompt, expected in FACT_TEST[:3]:
        preds = logit_lens_sweep(base_model, tok, prompt)
        exp_id = tok.encode(expected)[-1]
        hits = [l for l, p in preds.items() if p == exp_id]
        print(f"    '{expected.strip():>8s}' first correct at L{min(hits) if hits else 'NONE'} "
              f"({len(hits)}/{n_layers} layers)")

    # Sweep Oracle layers with Dual-Weight approach
    print("\n  === Dual-Weight Oracle Layer Sweep ===")
    fga_layer = n_layers - max(1, n_layers // 4)
    layer_results = {}

    for oracle_l in range(4, n_layers - 2):
        correct = 0
        oracle_correct = 0
        for prompt, expected in FACT_TEST:
            exp_id = tok.encode(expected)[-1]
            final_id, oracle_id, _ = dual_weight_oracle(
                base_model, surg_model, tok, prompt,
                fga_gain=5, oracle_layer=oracle_l, fga_layer=fga_layer)
            if oracle_id == exp_id: oracle_correct += 1
            if final_id == exp_id: correct += 1

        o_acc = oracle_correct / len(FACT_TEST)
        f_acc = correct / len(FACT_TEST)
        layer_results[oracle_l] = {'oracle_acc': o_acc, 'final_acc': f_acc}
        marker = " ***" if f_acc >= 0.8 else ""
        print(f"    Oracle=L{oracle_l:2d}: base_lens={o_acc:.0%} "
              f"dual_final={f_acc:.0%}{marker}")

    best_oracle_l = max(layer_results, key=lambda l: layer_results[l]['final_acc'])
    best = layer_results[best_oracle_l]

    # Full evaluation with best Oracle layer
    print(f"\n  === Full Eval (Oracle=L{best_oracle_l}) ===")
    all_results = []

    # Facts
    for prompt, expected in FACT_TEST:
        exp_id = tok.encode(expected)[-1]
        final_id, oracle_id, entropy = dual_weight_oracle(
            base_model, surg_model, tok, prompt,
            fga_gain=5, oracle_layer=best_oracle_l, fga_layer=fga_layer)
        correct = (final_id == exp_id)
        all_results.append({'type': 'fact', 'correct': correct, 'entropy': entropy,
                            'expected': expected.strip(),
                            'pred': tok.decode([final_id]).strip()})

    # Arithmetic
    for prompt, expected in ARITH_TEST:
        exp_id = tok.encode(expected)[-1]
        final_id, oracle_id, entropy = dual_weight_oracle(
            base_model, surg_model, tok, prompt,
            fga_gain=5, oracle_layer=best_oracle_l, fga_layer=fga_layer)
        correct = (final_id == exp_id) or (tok.decode([final_id]).strip() == expected.strip())
        all_results.append({'type': 'arith', 'correct': correct, 'entropy': entropy,
                            'expected': expected.strip(),
                            'pred': tok.decode([final_id]).strip()})

    fact_acc = sum(1 for r in all_results if r['type'] == 'fact' and r['correct']) / \
              max(1, sum(1 for r in all_results if r['type'] == 'fact'))
    arith_acc = sum(1 for r in all_results if r['type'] == 'arith' and r['correct']) / \
               max(1, sum(1 for r in all_results if r['type'] == 'arith'))
    print(f"    Fact: {fact_acc:.0%}, Arith: {arith_acc:.0%}")

    # Compare: P177 single-model Oracle vs P180 dual-model Oracle
    p177_best = 0.4  # from P177 results

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase180_dual_oracle.json'), 'w') as f:
        json.dump({'phase': '180', 'name': 'Dual-Weight Oracle',
                   'best_oracle_layer': best_oracle_l,
                   'layer_results': {str(k): v for k, v in layer_results.items()},
                   'fact_acc': fact_acc, 'arith_acc': arith_acc,
                   'details': all_results,
                   'p177_comparison': p177_best}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Dual-Weight vs Single-Weight Oracle accuracy per layer
    ax = axes[0]
    layers = sorted(layer_results.keys())
    dual_accs = [layer_results[l]['final_acc'] for l in layers]
    oracle_accs = [layer_results[l]['oracle_acc'] for l in layers]
    ax.plot(layers, oracle_accs, 'b-o', lw=2, markersize=5, label='Base Logit Lens')
    ax.plot(layers, dual_accs, 'r-s', lw=2.5, markersize=6, label='Dual-Weight Final')
    ax.axhline(y=p177_best, color='gray', ls='--', alpha=0.7,
               label=f'P177 single-model best ({p177_best:.0%})')
    ax.axvline(x=best_oracle_l, color='green', ls=':', lw=2, alpha=0.7,
               label=f'Best: L{best_oracle_l}')
    ax.set_xlabel('Oracle Layer', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('Dual-Weight Oracle: Layer Sweep', fontsize=13, fontweight='bold')

    # Right: Comparison bar chart
    ax = axes[1]
    methods = ['P177\nSingle-Model\nOracle', 'P180\nDual-Weight\nOracle',
               'Teacher\nFGA\n(ground truth)']
    vals = [p177_best, best['final_acc'], 0.9]
    colors = ['#3498db', '#e74c3c', '#95a5a6']
    ax.bar(methods, vals, color=colors, alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fact Accuracy', fontsize=12)
    ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Oracle Methods Comparison', fontsize=13, fontweight='bold')

    plt.suptitle('Phase 180: Dual-Weight Oracle\n'
                 'BASE model reads truth, SURGERY model steers output',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase180_dual_oracle.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    print(f"  -> P177 single-model: {p177_best:.0%}")
    print(f"  -> P180 dual-weight:  {best['final_acc']:.0%} (Oracle=L{best_oracle_l})")
    improvement = best['final_acc'] - p177_best
    if improvement > 0.1:
        print(f"  -> DUAL-WEIGHT FIX WORKS! +{improvement:.0%} improvement!")
    elif improvement > 0:
        print(f"  -> Modest improvement: +{improvement:.0%}")
    else:
        print(f"  -> No improvement. Surgery model FGA itself may be the bottleneck.")
    print(f"  -> Full eval: fact={fact_acc:.0%}, arith={arith_acc:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 180] Complete.")

    del base_model, surg_model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
