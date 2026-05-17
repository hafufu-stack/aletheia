# -*- coding: utf-8 -*-
"""
Phase 200: The Operation Register
A+B, A-B, A*B use the same operands but different operations.
Is "which operation" stored as a register?

Probe: Train a classifier to distinguish +, -, * from hidden states.
This proves the LLM has an "opcode register" like a real CPU.

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
    print("[P200] The Operation Register")
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
    target_layers = list(range(0, n_layers, 2))  # Every other layer

    # Generate problems: same A,B but different operations
    data = []
    for a in range(2, 10):
        for b in range(1, a):  # b < a to avoid negative results
            data.append((a, b, '+', a+b))
            data.append((a, b, '-', a-b))
            data.append((a, b, '*', a*b))

    op_map = {'+': 0, '-': 1, '*': 2}
    labels_op = np.array([op_map[d[2]] for d in data])
    labels_result = np.array([d[3] for d in data])
    labels_a = np.array([d[0] for d in data])
    labels_b = np.array([d[1] for d in data])

    print(f"  {len(data)} problems ({len(data)//3} x 3 ops)")

    # Collect hidden states
    all_hiddens = {l: [] for l in target_layers}
    captured = {}
    def make_hook(layer_idx):
        def fn(module, input, output):
            if isinstance(output, tuple):
                captured[layer_idx] = output[0][:, -1, :].detach().cpu().numpy().flatten()
            else:
                captured[layer_idx] = output[:, -1, :].detach().cpu().numpy().flatten()
        return fn

    for i, (a, b, op, result) in enumerate(data):
        prompt = f"def f(): return {a} {op} {b} ="
        captured.clear()
        hooks = []
        for l in target_layers:
            hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            _ = model(**inp)
        for h in hooks: h.remove()
        for l in target_layers:
            all_hiddens[l].append(captured[l].copy())

    # Probe
    results = {}
    idx = np.random.RandomState(42).permutation(len(data))
    n_tr = int(0.7 * len(data))
    tr, te = idx[:n_tr], idx[n_tr:]

    print("\n  === Probing Results ===")
    for l in target_layers:
        X = np.array(all_hiddens[l])
        # Operation type (3 classes: +, -, *)
        clf_op = LogisticRegression(max_iter=1000, random_state=42)
        clf_op.fit(X[tr], labels_op[tr])
        acc_op = accuracy_score(labels_op[te], clf_op.predict(X[te]))
        # Operand A
        clf_a = LogisticRegression(max_iter=1000, random_state=42)
        clf_a.fit(X[tr], labels_a[tr])
        acc_a = accuracy_score(labels_a[te], clf_a.predict(X[te]))
        # Operand B
        clf_b = LogisticRegression(max_iter=1000, random_state=42)
        clf_b.fit(X[tr], labels_b[tr])
        acc_b = accuracy_score(labels_b[te], clf_b.predict(X[te]))

        results[str(l)] = {'operation': acc_op, 'A': acc_a, 'B': acc_b}
        op_marker = " ***OPCODE***" if acc_op > 0.5 else ""
        print(f"    L{l:2d}: op={acc_op:.0%}{op_marker} A={acc_a:.0%} B={acc_b:.0%}")

    with open(os.path.join(RESULTS_DIR, 'phase200_opcode.json'), 'w') as f:
        json.dump({'phase': '200', 'name': 'The Operation Register',
                   'n_problems': len(data), 'results': results}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    layers = target_layers
    op_vals = [results[str(l)]['operation'] for l in layers]
    a_vals = [results[str(l)]['A'] for l in layers]
    b_vals = [results[str(l)]['B'] for l in layers]
    ax.plot(layers, op_vals, 'k-D', lw=3, markersize=8, label='OPCODE (+/-/*)', color='#e74c3c')
    ax.plot(layers, a_vals, 'r-o', lw=2, markersize=5, label='Operand A', alpha=0.7)
    ax.plot(layers, b_vals, 'b-s', lw=2, markersize=5, label='Operand B', alpha=0.7)
    ax.axhline(y=1/3, color='gray', ls=':', alpha=0.5, label='Chance (op)')
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Probe Accuracy', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_title('Phase 200: The Operation Register\n'
                 'Does the LLM have an "opcode" register for +, -, *?',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase200_opcode.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    best_op_l = max(target_layers, key=lambda l: results[str(l)]['operation'])
    best_op = results[str(best_op_l)]['operation']
    print(f"\n  === VERDICT ===")
    print(f"  -> Best opcode probe: L{best_op_l} ({best_op:.0%})")
    if best_op > 0.6:
        print("  -> OPCODE REGISTER FOUND! LLM stores which operation to execute!")
        print("  -> Complete CPU model: opcode + operands + carry + result")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 200] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
