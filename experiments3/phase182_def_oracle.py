# -*- coding: utf-8 -*-
"""
Phase 182: The Def-Oracle Synergy
P181 achieved Autopoiesis (Oracle=Teacher at 100% ratio).
But base accuracy was Fact=50%, Arith=38%.

Fix: Apply "def f(): return " to BOTH base and surgery models.
P176 proved "def" gives 100% arithmetic protection.

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
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The largest planet is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("The boiling point of water is", " 100"),
    ("The atomic number of carbon is", " 6"),
    ("A year has", " 365"),
    ("The number of continents is", " 7"),
    ("Pi is approximately", " 3"),
]
ARITH_TEST = [
    ("1 + 1 =", " 2"), ("3 + 4 =", " 7"), ("5 + 5 =", " 10"),
    ("2 + 7 =", " 9"), ("8 + 1 =", " 9"), ("6 + 3 =", " 9"),
    ("4 + 4 =", " 8"), ("9 + 0 =", " 9"), ("7 + 2 =", " 9"),
    ("2 + 6 =", " 8"),
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

def run_config(base_model, surg_model, tok, test_data, prefix, fga_gain, fga_layer):
    correct = 0
    details = []
    for prompt, expected in test_data:
        text = f"{prefix}{prompt}"
        exp_id = tok.encode(expected)[-1]
        inp = tok(text, return_tensors='pt').to(DEVICE)
        # Phase 1: BASE prediction
        with torch.no_grad():
            base_logits = base_model(**inp).logits[0, -1, :].float()
        base_pred_id = base_logits.argmax().item()
        # Phase 2: SURGERY + FGA toward base prediction
        unembed = surg_model.lm_head.weight.data[base_pred_id].float()
        direction = unembed / (unembed.norm() + 1e-8)
        def mk(d, g):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].float()
                    if h.dim() == 3: h[:, -1, :] += g * d.to(h.device)
                    return (h.to(output[0].dtype),) + output[1:]
                return output
            return fn
        handle = surg_model.model.layers[fga_layer].register_forward_hook(mk(direction, fga_gain))
        with torch.no_grad():
            final_logits = surg_model(**inp).logits[0, -1, :].float()
        handle.remove()
        pred_id = final_logits.argmax().item()
        ok = (pred_id == exp_id) or (tok.decode([pred_id]).strip() == expected.strip())
        if ok: correct += 1
        details.append({'expected': expected.strip(), 'pred': tok.decode([pred_id]).strip(),
                        'base_pred': tok.decode([base_pred_id]).strip(), 'correct': ok})
    return correct / len(test_data), details

def main():
    print("[P182] The Def-Oracle Synergy")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    surg = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(surg, tok, strength=2.0)
    n_layers = base.config.num_hidden_layers
    fga_layer = n_layers - max(1, n_layers // 4)
    configs = {}
    prefixes = [("# ", "hash"), ("def f(): return ", "def"), (">>> ", "repl"), ("print(", "print")]
    gains = [5, 10]
    for prefix, pname in prefixes:
        for g in gains:
            fa, fd = run_config(base, surg, tok, FACT_TEST, prefix, g, fga_layer)
            aa, ad = run_config(base, surg, tok, ARITH_TEST, prefix, g, fga_layer)
            key = f'{pname}_g{g}'
            configs[key] = {'prefix': prefix, 'gain': g, 'fact': fa, 'arith': aa}
            print(f"  {key:15s}: fact={fa:.0%} arith={aa:.0%}")
    # Also test: base-only with def prefix (no surgery, no FGA)
    for prefix, pname in prefixes:
        c = 0
        for prompt, expected in FACT_TEST + ARITH_TEST:
            text = f"{prefix}{prompt}"
            exp_id = tok.encode(expected)[-1]
            inp = tok(text, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = base(**inp).logits[0, -1, :].float()
            if logits.argmax().item() == exp_id: c += 1
        total = len(FACT_TEST) + len(ARITH_TEST)
        configs[f'{pname}_base_only'] = {'prefix': prefix, 'gain': 0, 'overall': c/total}
        print(f"  {pname}_base_only: overall={c/total:.0%}")

    with open(os.path.join(RESULTS_DIR, 'phase182_def_oracle.json'), 'w') as f:
        json.dump({'phase': '182', 'name': 'Def-Oracle Synergy', 'configs': configs}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    keys = [k for k in configs if '_g' in k]
    labels = [k.replace('_', '\n') for k in keys]
    fact_v = [configs[k]['fact'] for k in keys]
    arith_v = [configs[k]['arith'] for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x-w/2, fact_v, w, label='Factual', color='#e74c3c', alpha=0.8)
    ax.bar(x+w/2, arith_v, w, label='Arithmetic', color='#3498db', alpha=0.8)
    for i in range(len(keys)):
        ax.text(x[i]-w/2, fact_v[i]+0.02, f'{fact_v[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i]+w/2, arith_v[i]+0.02, f'{arith_v[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11); ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase 182: Def-Oracle Synergy\nFinal Oracle + Optimal Prefix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase182_def_oracle.png'), dpi=150)
    plt.close()

    best_key = max([k for k in configs if '_g' in k],
                   key=lambda k: configs[k]['fact'] + configs[k]['arith'])
    best = configs[best_key]
    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    print(f"  -> Best: {best_key} fact={best['fact']:.0%} arith={best['arith']:.0%}")
    p181_fact, p181_arith = 0.5, 0.38
    print(f"  -> P181 (# prefix): fact={p181_fact:.0%} arith={p181_arith:.0%}")
    delta_f = best['fact'] - p181_fact
    delta_a = best['arith'] - p181_arith
    if delta_f > 0 or delta_a > 0:
        print(f"  -> IMPROVEMENT: fact+{delta_f:.0%}, arith+{delta_a:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 182] Complete.")
    del base, surg; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
