# -*- coding: utf-8 -*-
"""
Phase 192: Code-Booted Abacus
P190: Carry=87-100%, but ones_digit=0%, full_sum=0%.
Does the "def" prefix boot a BETTER abacus?

Compare linear probe accuracy with:
  - "# " prefix (P190 baseline)
  - "def f(): return " prefix
  - No prefix (natural language)

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def gen_additions():
    data = []
    for a in range(10):
        for b in range(10):
            s = a + b
            data.append((a, b, s, 1 if s >= 10 else 0, s % 10))
    return data

def collect_and_probe(model, tok, prefix, additions, target_layers):
    hiddens = {l: [] for l in target_layers}
    labels_carry = []; labels_ones = []; labels_sum = []
    for a, b, s, carry, ones in additions:
        prompt = f"{prefix}{a} + {b} ="
        hs = {}; hooks = []
        def make_hook(l):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    hs[l] = output[0][:, -1, :].detach().cpu().numpy().flatten()
                else:
                    hs[l] = output[:, -1, :].detach().cpu().numpy().flatten()
            return fn
        for l in target_layers:
            hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            _ = model(**inp)
        for h in hooks: h.remove()
        for l in target_layers:
            if l in hs: hiddens[l].append(hs[l])
        labels_carry.append(carry); labels_ones.append(ones); labels_sum.append(s)

    labels_carry = np.array(labels_carry)
    labels_ones = np.array(labels_ones)
    labels_sum = np.array(labels_sum)
    results = {}
    for l in target_layers:
        X = np.array(hiddens[l])
        n = len(X)
        idx = np.random.RandomState(42).permutation(n)
        tr, te = idx[:70], idx[70:]
        # Carry
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X[tr], labels_carry[tr])
        carry_acc = accuracy_score(labels_carry[te], clf.predict(X[te]))
        # Ones
        clf2 = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
        clf2.fit(X[tr], labels_ones[tr])
        ones_acc = accuracy_score(labels_ones[te], clf2.predict(X[te]))
        # Sum
        clf3 = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
        clf3.fit(X[tr], labels_sum[tr])
        sum_acc = accuracy_score(labels_sum[te], clf3.predict(X[te]))
        results[l] = {'carry': carry_acc, 'ones': ones_acc, 'sum': sum_acc}
    return results

def main():
    print("[P192] Code-Booted Abacus")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    additions = gen_additions()
    target_layers = [10, 14, 15, 18, 20, 22]
    prefixes = [("", "natural"), ("# ", "hash"), ("def f(): return ", "def")]
    all_results = {}

    for prefix, pname in prefixes:
        print(f"\n  === Prefix: '{prefix}' ({pname}) ===")
        results = collect_and_probe(model, tok, prefix, additions, target_layers)
        all_results[pname] = results
        for l in target_layers:
            r = results[l]
            marker = " ***" if r['ones'] > 0.2 else ""
            print(f"    L{l:2d}: carry={r['carry']:.0%} ones={r['ones']:.0%} sum={r['sum']:.0%}{marker}")

    with open(os.path.join(RESULTS_DIR, 'phase192_code_abacus.json'), 'w') as f:
        json.dump({'phase': '192', 'name': 'Code-Booted Abacus',
                   'results': {p: {str(l): v for l, v in r.items()}
                               for p, r in all_results.items()}}, f, indent=2, default=str)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (metric, title) in enumerate([('carry', 'Carry Bit'), ('ones', 'Ones Digit'), ('sum', 'Full Sum')]):
        ax = axes[i]
        for pname in ['natural', 'hash', 'def']:
            vals = [all_results[pname][l][metric] for l in target_layers]
            ax.plot(target_layers, vals, '-o', lw=2, markersize=6, label=pname)
        ax.set_xlabel('Layer', fontsize=12); ax.set_ylabel('Probe Accuracy', fontsize=12)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'{title} Probe', fontsize=13, fontweight='bold')
    plt.suptitle('Phase 192: Code-Booted Abacus\nDoes "def" prefix improve internal computation?',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase192_code_abacus.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    for pname in ['natural', 'hash', 'def']:
        best_ones = max(all_results[pname][l]['ones'] for l in target_layers)
        best_carry = max(all_results[pname][l]['carry'] for l in target_layers)
        print(f"  -> {pname}: best carry={best_carry:.0%} best ones={best_ones:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 192] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
