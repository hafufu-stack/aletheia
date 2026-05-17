# -*- coding: utf-8 -*-
"""
Phase 198: The Operand Register (Opus Addition)
P190/192: Carry/ones/sum registers found at 87-100% accuracy.
But do the INPUT operands (A, B) also have registers?

If A and B are stored as separate vectors, the LLM is truly
implementing a=input1, b=input2, c=a+b as a virtual program.

Probe: Can we read A and B separately from hidden states
during "A + B =" computation?

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

def main():
    print("[P198] The Operand Register")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    target_layers = [10, 12, 14, 15, 18, 20, 22]

    # Generate all single-digit additions
    data = []
    for a in range(10):
        for b in range(10):
            data.append((a, b, a+b))

    # Collect hidden states at "=" position
    print(f"  Collecting hidden states for {len(data)} additions...")
    hiddens = {l: [] for l in target_layers}
    labels_a = []; labels_b = []; labels_sum = []
    captured = [None]
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured[0] = output[0][:, -1, :].detach().cpu().numpy().flatten()
        else:
            captured[0] = output[:, -1, :].detach().cpu().numpy().flatten()

    for a, b, s in data:
        prompt = f"def f(): return {a} + {b} ="
        for l in target_layers:
            h = model.model.layers[l].register_forward_hook(hook_fn)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                _ = model(**inp)
            h.remove()
            hiddens[l].append(captured[0].copy())
        labels_a.append(a)
        labels_b.append(b)
        labels_sum.append(s)

    labels_a = np.array(labels_a)
    labels_b = np.array(labels_b)
    labels_sum = np.array(labels_sum)

    # Train probes for: A, B, A+B
    results = {}
    print("\n  === Probing Results ===")
    for l in target_layers:
        X = np.array(hiddens[l])
        idx = np.random.RandomState(42).permutation(len(X))
        tr, te = idx[:70], idx[70:]

        # Operand A (10 classes)
        clf_a = LogisticRegression(max_iter=1000, random_state=42)
        clf_a.fit(X[tr], labels_a[tr])
        acc_a = accuracy_score(labels_a[te], clf_a.predict(X[te]))

        # Operand B (10 classes)
        clf_b = LogisticRegression(max_iter=1000, random_state=42)
        clf_b.fit(X[tr], labels_b[tr])
        acc_b = accuracy_score(labels_b[te], clf_b.predict(X[te]))

        # Sum (19 classes)
        clf_s = LogisticRegression(max_iter=1000, random_state=42)
        clf_s.fit(X[tr], labels_sum[tr])
        acc_s = accuracy_score(labels_sum[te], clf_s.predict(X[te]))

        results[str(l)] = {'operand_A': acc_a, 'operand_B': acc_b, 'sum': acc_s}
        a_marker = " ***" if acc_a > 0.3 else ""
        b_marker = " ***" if acc_b > 0.3 else ""
        print(f"    L{l:2d}: A={acc_a:.0%}{a_marker} B={acc_b:.0%}{b_marker} Sum={acc_s:.0%}")

    with open(os.path.join(RESULTS_DIR, 'phase198_operand.json'), 'w') as f:
        json.dump({'phase': '198', 'name': 'The Operand Register',
                   'results': results}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    layers = target_layers
    a_vals = [results[str(l)]['operand_A'] for l in layers]
    b_vals = [results[str(l)]['operand_B'] for l in layers]
    s_vals = [results[str(l)]['sum'] for l in layers]
    ax.plot(layers, a_vals, 'r-o', lw=2.5, markersize=8, label='Operand A')
    ax.plot(layers, b_vals, 'b-s', lw=2.5, markersize=8, label='Operand B')
    ax.plot(layers, s_vals, 'g-^', lw=2, markersize=7, label='Sum (A+B)')
    ax.axhline(y=0.1, color='gray', ls=':', alpha=0.5, label='Chance (10-class)')
    ax.axvspan(13.5, 15.5, alpha=0.1, color='orange', label='Code-Math (P172)')
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Probe Accuracy', fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_title('Phase 198: The Operand Register\n'
                 'Can we read A, B, and A+B separately from hidden states?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase198_operand.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    best_a = max(results[str(l)]['operand_A'] for l in layers)
    best_b = max(results[str(l)]['operand_B'] for l in layers)
    best_s = max(results[str(l)]['sum'] for l in layers)
    print(f"\n  === VERDICT ===")
    print(f"  -> Best Operand A probe: {best_a:.0%}")
    print(f"  -> Best Operand B probe: {best_b:.0%}")
    print(f"  -> Best Sum probe: {best_s:.0%}")
    if best_a > 0.3 and best_b > 0.3:
        print("  -> OPERAND REGISTERS FOUND! A and B stored separately")
        print("  -> LLM implements: reg_a=A, reg_b=B, reg_c=A+B")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 198] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
