# -*- coding: utf-8 -*-
"""
Phase 190: The Abacus Probe
Does the LLM create "virtual registers" for arithmetic?

Linear probe on L14-15 hidden states during arithmetic.
Train a tiny probe to predict:
  1. The carry bit (is there a carry in this addition?)
  2. The expected answer digit

If successful: LLM literally creates a virtual abacus.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time
import torch.nn.functional as F
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

# Generate all single-digit additions
def gen_additions():
    data = []
    for a in range(10):
        for b in range(10):
            s = a + b
            carry = 1 if s >= 10 else 0
            prompt = f"# {a} + {b} ="
            data.append((prompt, s, carry, a, b))
    return data

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

def main():
    print("[P190] The Abacus Probe")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    n_layers = model.config.num_hidden_layers
    target_layers = [10, 12, 14, 15, 16, 18, 20]
    additions = gen_additions()  # 100 examples

    # Collect hidden states
    print(f"\n  Collecting hidden states for {len(additions)} additions...")
    all_hiddens = {l: [] for l in target_layers}
    labels_sum = []
    labels_carry = []
    labels_ones = []  # ones digit of sum

    for prompt, s, carry, a, b in additions:
        hooks = []
        hiddens = {}
        def make_hook(l):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    hiddens[l] = output[0][:, -1, :].detach().cpu().numpy().flatten()
                else:
                    hiddens[l] = output[:, -1, :].detach().cpu().numpy().flatten()
            return fn
        for l in target_layers:
            hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            _ = model(**inp)
        for h in hooks:
            h.remove()
        for l in target_layers:
            if l in hiddens:
                all_hiddens[l].append(hiddens[l])
        labels_sum.append(s)
        labels_carry.append(carry)
        labels_ones.append(s % 10)

    labels_sum = np.array(labels_sum)
    labels_carry = np.array(labels_carry)
    labels_ones = np.array(labels_ones)

    # Train linear probes
    results = {}
    print("\n  === Linear Probing Results ===")

    for l in target_layers:
        X = np.array(all_hiddens[l])
        # Split
        n = len(X)
        idx = np.random.RandomState(42).permutation(n)
        tr, te = idx[:70], idx[70:]

        # Probe 1: Carry bit (binary)
        clf_carry = LogisticRegression(max_iter=1000, random_state=42)
        clf_carry.fit(X[tr], labels_carry[tr])
        carry_acc = accuracy_score(labels_carry[te], clf_carry.predict(X[te]))

        # Probe 2: Ones digit (10 classes)
        clf_ones = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
        clf_ones.fit(X[tr], labels_ones[tr])
        ones_acc = accuracy_score(labels_ones[te], clf_ones.predict(X[te]))

        # Probe 3: Full sum (19 classes: 0-18)
        clf_sum = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
        clf_sum.fit(X[tr], labels_sum[tr])
        sum_acc = accuracy_score(labels_sum[te], clf_sum.predict(X[te]))

        results[str(l)] = {'carry': carry_acc, 'ones': ones_acc, 'sum': sum_acc}
        marker = " ***" if carry_acc > 0.7 else ""
        print(f"    L{l:2d}: carry={carry_acc:.0%} ones_digit={ones_acc:.0%} "
              f"full_sum={sum_acc:.0%}{marker}")

    with open(os.path.join(RESULTS_DIR, 'phase190_abacus.json'), 'w') as f:
        json.dump({'phase': '190', 'name': 'The Abacus Probe',
                   'results': results}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    layers = target_layers
    carry_vals = [results[str(l)]['carry'] for l in layers]
    ones_vals = [results[str(l)]['ones'] for l in layers]
    sum_vals = [results[str(l)]['sum'] for l in layers]
    ax.plot(layers, carry_vals, 'r-o', lw=2.5, markersize=8, label='Carry bit')
    ax.plot(layers, ones_vals, 'b-s', lw=2, markersize=7, label='Ones digit')
    ax.plot(layers, sum_vals, 'g-^', lw=2, markersize=7, label='Full sum')
    ax.axhline(y=0.5, color='gray', ls=':', alpha=0.5, label='Chance (carry)')
    ax.axhline(y=0.1, color='lightgray', ls=':', alpha=0.5, label='Chance (digit)')
    ax.axvspan(13.5, 15.5, alpha=0.1, color='orange', label='Code-Math (P172)')
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Probe Accuracy', fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('Phase 190: The Abacus Probe\n'
                 'Does the LLM create virtual registers for arithmetic?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase190_abacus.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    best_carry = max(results.values(), key=lambda x: x['carry'])
    best_carry_l = [l for l in results if results[l]['carry'] == best_carry['carry']][0]
    print(f"\n  === VERDICT ===")
    print(f"  -> Best carry probe: L{best_carry_l} ({best_carry['carry']:.0%})")
    if best_carry['carry'] > 0.7:
        print("  -> VIRTUAL ABACUS FOUND! Carry information is encoded!")
    if float(best_carry_l) in [14, 15]:
        print("  -> Located in CODE-MATH NEURON region (L14-15)!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 190] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
