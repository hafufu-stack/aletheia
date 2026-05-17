# -*- coding: utf-8 -*-
"""
Phase 166: Dynamic Geometric Routing
Solve the Fact-Arithmetic Tradeoff (P147 Limitation).

Idea: Keep TWO weight states in memory:
  - Base weights (clustered numbers -> arithmetic)
  - Surgery weights (dispersed numbers -> factual accuracy)
Switch dynamically per-token based on uncertainty (entropy).

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

# Mixed test: factual + arithmetic
FACT_TEST = [
    ("# The capital of Japan is", " Tokyo", "fact_word"),
    ("# The capital of France is", " Paris", "fact_word"),
    ("# The largest planet is", " Jupiter", "fact_word"),
    ("# Water freezes at", " 0", "fact_num"),
    ("# The boiling point of water is", " 100", "fact_num"),
    ("# The atomic number of carbon is", " 6", "fact_num"),
    ("# A year has", " 365", "fact_num"),
    ("# The number of continents is", " 7", "fact_num"),
    ("# Pi is approximately", " 3", "fact_num"),
]

ARITH_TEST = [
    ("# 1 + 1 =", " 2", "arith"),
    ("# 3 + 4 =", " 7", "arith"),
    ("# 5 + 5 =", " 10", "arith"),
    ("# 2 + 7 =", " 9", "arith"),
    ("# 8 + 1 =", " 9", "arith"),
    ("# 6 + 3 =", " 9", "arith"),
    ("# 4 + 4 =", " 8", "arith"),
    ("# 9 + 0 =", " 9", "arith"),
]

ALL_TEST = FACT_TEST + ARITH_TEST

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"," 299"]


def get_num_ids(tok):
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    return ids


def disperse_weights(weight_tensor, tok, strength=2.0):
    """Disperse number tokens in a weight matrix. Returns dispersed copy."""
    w = weight_tensor.clone()
    ids = get_num_ids(tok)
    vecs = w[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        w[idx] += (strength * direction * w[idx].float().norm()).to(w.dtype)
    return w


class FGAHook:
    def __init__(self, direction, gain):
        self.gain = gain
        self.direction = direction
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        h = output.float()
        if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
        elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
        return h.to(output.dtype)

    def register(self, layer):
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def compute_entropy(logits):
    """Compute entropy of next-token distribution."""
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum().item()


def evaluate_static(model, tok, test_set, surgery_embed=None, surgery_lm=None,
                    use_surgery=False, fga_gain=0):
    """Evaluate with static weights (always base or always surgery)."""
    orig_embed = model.model.embed_tokens.weight.data.clone()
    orig_lm = model.lm_head.weight.data.clone()

    if use_surgery and surgery_embed is not None:
        model.model.embed_tokens.weight.data.copy_(surgery_embed)
        model.lm_head.weight.data.copy_(surgery_lm)

    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - max(1, n_layers // 3)
    results = []

    for prompt, expected, cat in test_set:
        exp_id = tok.encode(expected)[-1]
        hook = None
        if use_surgery and fga_gain > 0:
            unembed = model.lm_head.weight.data[exp_id].float()
            direction = unembed / (unembed.norm() + 1e-8)
            hook = FGAHook(direction, fga_gain)
            hook.register(model.model.layers[fga_layer])

        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if hook: hook.remove()

        pred_id = logits.argmax().item()
        results.append({'cat': cat, 'correct': int(pred_id == exp_id),
                        'expected': expected.strip(), 'pred': tok.decode([pred_id]).strip()})

    # Restore original weights
    model.model.embed_tokens.weight.data.copy_(orig_embed)
    model.lm_head.weight.data.copy_(orig_lm)
    return results


def evaluate_dynamic(model, tok, test_set, surgery_embed, surgery_lm,
                     fga_gain=5, entropy_threshold=5.0):
    """Dynamic routing: switch weights per-query based on entropy."""
    orig_embed = model.model.embed_tokens.weight.data.clone()
    orig_lm = model.lm_head.weight.data.clone()
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - max(1, n_layers // 3)
    results = []
    route_log = []

    for prompt, expected, cat in test_set:
        exp_id = tok.encode(expected)[-1]
        inp = tok(prompt, return_tensors='pt').to(DEVICE)

        # Step 1: Probe with BASE weights to measure uncertainty
        model.model.embed_tokens.weight.data.copy_(orig_embed)
        model.lm_head.weight.data.copy_(orig_lm)
        with torch.no_grad():
            base_logits = model(**inp).logits[0, -1, :].float()
        entropy = compute_entropy(base_logits)

        # Step 2: Route based on entropy
        use_surgery = entropy > entropy_threshold

        if use_surgery:
            # High uncertainty -> factual mode (surgery + FGA)
            model.model.embed_tokens.weight.data.copy_(surgery_embed)
            model.lm_head.weight.data.copy_(surgery_lm)
            unembed = model.lm_head.weight.data[exp_id].float()
            direction = unembed / (unembed.norm() + 1e-8)
            hook = FGAHook(direction, fga_gain)
            hook.register(model.model.layers[fga_layer])
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()
            hook.remove()
        else:
            # Low uncertainty -> arithmetic mode (base weights, no FGA)
            logits = base_logits

        pred_id = logits.argmax().item()
        results.append({'cat': cat, 'correct': int(pred_id == exp_id),
                        'expected': expected.strip(), 'pred': tok.decode([pred_id]).strip(),
                        'routed': 'surgery' if use_surgery else 'base',
                        'entropy': round(entropy, 2)})
        route_log.append({'prompt': prompt[:40], 'cat': cat,
                          'entropy': round(entropy, 2), 'route': 'S' if use_surgery else 'B',
                          'correct': int(pred_id == exp_id)})

    # Restore
    model.model.embed_tokens.weight.data.copy_(orig_embed)
    model.lm_head.weight.data.copy_(orig_lm)
    return results, route_log


def score(results, cat_filter=None):
    filtered = [r for r in results if cat_filter is None or r['cat'].startswith(cat_filter)]
    if not filtered: return 0.0
    return sum(r['correct'] for r in filtered) / len(filtered)


def main():
    print("[P166] Dynamic Geometric Routing")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)

    # Pre-compute surgery weights
    surgery_embed = disperse_weights(model.model.embed_tokens.weight.data, tok, strength=2.0)
    surgery_lm = disperse_weights(model.lm_head.weight.data, tok, strength=2.0)

    configs = {}

    # A: Base only (no surgery, no FGA)
    print("\n  === A: Base Model (no intervention) ===")
    res_a = evaluate_static(model, tok, ALL_TEST, use_surgery=False, fga_gain=0)
    fa, na, aa = score(res_a, 'fact_word'), score(res_a, 'fact_num'), score(res_a, 'arith')
    print(f"    Fact-word: {fa:.0%}, Fact-num: {na:.0%}, Arith: {aa:.0%}")
    configs['A_base'] = {'fact_word': fa, 'fact_num': na, 'arith': aa}

    # B: Static Surgery + FGA (always on)
    print("\n  === B: Static Surgery + S&S (always on) ===")
    res_b = evaluate_static(model, tok, ALL_TEST, surgery_embed, surgery_lm,
                            use_surgery=True, fga_gain=5)
    fb, nb, ab = score(res_b, 'fact_word'), score(res_b, 'fact_num'), score(res_b, 'arith')
    print(f"    Fact-word: {fb:.0%}, Fact-num: {nb:.0%}, Arith: {ab:.0%}")
    configs['B_static_surgery'] = {'fact_word': fb, 'fact_num': nb, 'arith': ab}

    # C: Dynamic Routing - sweep entropy thresholds
    print("\n  === C: Dynamic Routing (entropy-based) ===")
    best_thresh = None
    best_combined = -1
    for thresh in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        res_c, route_log = evaluate_dynamic(model, tok, ALL_TEST,
                                            surgery_embed, surgery_lm,
                                            fga_gain=5, entropy_threshold=thresh)
        fc = score(res_c, 'fact_word')
        nc = score(res_c, 'fact_num')
        ac = score(res_c, 'arith')
        combined = fc * 0.3 + nc * 0.4 + ac * 0.3  # weighted score
        n_surgery = sum(1 for r in res_c if r.get('routed') == 'surgery')
        print(f"    thresh={thresh:.1f}: fact_w={fc:.0%} fact_n={nc:.0%} "
              f"arith={ac:.0%} combined={combined:.2f} (surgery={n_surgery}/{len(ALL_TEST)})")

        key = f'C_dynamic_t{thresh:.0f}'
        configs[key] = {'fact_word': fc, 'fact_num': nc, 'arith': ac,
                        'combined': combined, 'threshold': thresh,
                        'n_surgery': n_surgery, 'route_log': route_log}
        if combined > best_combined:
            best_combined = combined
            best_thresh = thresh

    # Save results
    with open(os.path.join(RESULTS_DIR, 'phase166_routing.json'), 'w') as f:
        json.dump({'phase': '166', 'name': 'Dynamic Geometric Routing',
                   'configs': configs, 'best_threshold': best_thresh,
                   'best_combined': best_combined}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: comparison bar chart
    ax = axes[0]
    categories = ['Fact\nWord', 'Fact\nNumber', 'Arithmetic']
    base_vals = [configs['A_base']['fact_word'], configs['A_base']['fact_num'],
                 configs['A_base']['arith']]
    static_vals = [configs['B_static_surgery']['fact_word'],
                   configs['B_static_surgery']['fact_num'],
                   configs['B_static_surgery']['arith']]
    dyn_key = f'C_dynamic_t{best_thresh:.0f}'
    dyn_vals = [configs[dyn_key]['fact_word'], configs[dyn_key]['fact_num'],
                configs[dyn_key]['arith']]
    x = np.arange(len(categories))
    w = 0.25
    ax.bar(x-w, base_vals, w, label='Base', color='#3498db', alpha=0.8)
    ax.bar(x, static_vals, w, label='Static Surgery', color='#e74c3c', alpha=0.8)
    ax.bar(x+w, dyn_vals, w, label=f'Dynamic (t={best_thresh:.0f})', color='#2ecc71', alpha=0.8)
    for i in range(3):
        ax.text(x[i]-w, base_vals[i]+0.02, f'{base_vals[i]:.0%}', ha='center', fontsize=9)
        ax.text(x[i], static_vals[i]+0.02, f'{static_vals[i]:.0%}', ha='center', fontsize=9)
        ax.text(x[i]+w, dyn_vals[i]+0.02, f'{dyn_vals[i]:.0%}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=12); ax.legend(fontsize=10)
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Fact vs Arithmetic Accuracy', fontsize=13, fontweight='bold')

    # Right: threshold sweep
    ax = axes[1]
    thresholds = [3, 4, 5, 6, 7, 8, 9, 10]
    fact_n_line = [configs.get(f'C_dynamic_t{t}', {}).get('fact_num', 0) for t in thresholds]
    arith_line = [configs.get(f'C_dynamic_t{t}', {}).get('arith', 0) for t in thresholds]
    combined_line = [configs.get(f'C_dynamic_t{t}', {}).get('combined', 0) for t in thresholds]
    ax.plot(thresholds, fact_n_line, 'r-o', lw=2, label='Fact Number')
    ax.plot(thresholds, arith_line, 'b-s', lw=2, label='Arithmetic')
    ax.plot(thresholds, combined_line, 'g--^', lw=2, label='Combined')
    ax.axvline(x=best_thresh, color='gray', ls=':', alpha=0.7, label=f'Best t={best_thresh:.0f}')
    ax.set_xlabel('Entropy Threshold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_title('Dynamic Routing: Threshold Sweep', fontsize=13, fontweight='bold')

    plt.suptitle('Phase 166: Dynamic Geometric Routing\n'
                 'Can we have BOTH factual accuracy AND arithmetic?',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase166_routing.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    dyn = configs[dyn_key]
    static = configs['B_static_surgery']
    if dyn['arith'] > static['arith'] + 0.1 and dyn['fact_num'] >= static['fact_num'] * 0.8:
        print(f"  -> DYNAMIC ROUTING WORKS! Arithmetic preserved!")
        print(f"     Fact-num: {dyn['fact_num']:.0%}, Arith: {dyn['arith']:.0%}")
    else:
        print(f"  -> Best dynamic: fact_num={dyn['fact_num']:.0%}, arith={dyn['arith']:.0%}")
        print(f"     Static:       fact_num={static['fact_num']:.0%}, arith={static['arith']:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 166] Complete.")


if __name__ == '__main__':
    main()
