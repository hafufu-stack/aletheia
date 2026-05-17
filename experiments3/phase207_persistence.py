# -*- coding: utf-8 -*-
"""
Phase 207: Register Persistence
For 7+8=15, model outputs "1" then "5" autoregressively.
Does REG_A (=7) persist from token 1 to token 2?
Or is it replaced by intermediate results?

This tests if the CPU has "memory" across clock cycles.

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
    print("[P207] Register Persistence")
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
    target_layers = [0, 4, 8, 12, 15, 18, 20, 22]

    # Token 1: "def f(): return A + B =" -> model outputs first token
    # Token 2: "def f(): return A + B = 1" -> model outputs second token
    # Compare register readability between these two contexts

    data = [(a, b, a+b) for a in range(2, 10) for b in range(2, 10) if a+b >= 10]
    print(f"  {len(data)} carry problems")

    # Collect hidden states for both token positions
    hiddens_t1 = {l: [] for l in target_layers}  # at "=" position
    hiddens_t2 = {l: [] for l in target_layers}  # at "1" position (after "= 1")
    labels_a = []; labels_b = []; labels_ones = []

    captured = {}
    def make_hook(layer_idx):
        def fn(module, input, output):
            if isinstance(output, tuple):
                captured[layer_idx] = output[0][:, -1, :].detach().cpu().numpy().flatten()
            else:
                captured[layer_idx] = output[:, -1, :].detach().cpu().numpy().flatten()
        return fn

    for a, b, s in data:
        ones_digit = s % 10
        labels_a.append(a); labels_b.append(b); labels_ones.append(ones_digit)

        # Token 1 context
        prompt1 = f"def f(): return {a} + {b} ="
        captured.clear()
        hooks = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in target_layers]
        inp = tok(prompt1, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): _ = model(**inp)
        for h in hooks: h.remove()
        for l in target_layers:
            hiddens_t1[l].append(captured[l].copy())

        # Token 2 context (append " 1" since all carry sums start with 1)
        prompt2 = f"def f(): return {a} + {b} = 1"
        captured.clear()
        hooks = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in target_layers]
        inp = tok(prompt2, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): _ = model(**inp)
        for h in hooks: h.remove()
        for l in target_layers:
            hiddens_t2[l].append(captured[l].copy())

    labels_a = np.array(labels_a)
    labels_b = np.array(labels_b)
    labels_ones = np.array(labels_ones)

    # Probe each layer for A, B, ones_digit at both token positions
    results = {'token1': {}, 'token2': {}}
    idx = np.random.RandomState(42).permutation(len(data))
    n_tr = int(0.7 * len(data))
    tr, te = idx[:n_tr], idx[n_tr:]

    print("\n  === Register Persistence ===")
    print(f"  {'Layer':>6} | {'T1:A':>6} {'T1:B':>6} {'T1:ones':>8} | "
          f"{'T2:A':>6} {'T2:B':>6} {'T2:ones':>8}")
    print("  " + "-" * 65)

    for l in target_layers:
        X1 = np.array(hiddens_t1[l])
        X2 = np.array(hiddens_t2[l])

        r1 = {}; r2 = {}
        for label_name, labels in [('A', labels_a), ('B', labels_b), ('ones', labels_ones)]:
            # Token 1
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X1[tr], labels[tr])
            r1[label_name] = accuracy_score(labels[te], clf.predict(X1[te]))
            # Token 2
            clf2 = LogisticRegression(max_iter=1000, random_state=42)
            clf2.fit(X2[tr], labels[tr])
            r2[label_name] = accuracy_score(labels[te], clf2.predict(X2[te]))

        results['token1'][str(l)] = r1
        results['token2'][str(l)] = r2
        print(f"  L{l:2d}    | {r1['A']:5.0%}  {r1['B']:5.0%}  {r1['ones']:7.0%}  | "
              f"{r2['A']:5.0%}  {r2['B']:5.0%}  {r2['ones']:7.0%}")

    with open(os.path.join(RESULTS_DIR, 'phase207_persistence.json'), 'w') as f:
        json.dump({'phase': '207', 'name': 'Register Persistence',
                   'results': results}, f, indent=2, default=str)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, (token_name, token_data) in enumerate([('Token 1 ("=")', results['token1']),
                                                    ('Token 2 ("= 1 ...")', results['token2'])]):
        ax = axes[i]
        layers = target_layers
        a_vals = [token_data[str(l)]['A'] for l in layers]
        b_vals = [token_data[str(l)]['B'] for l in layers]
        o_vals = [token_data[str(l)]['ones'] for l in layers]
        ax.plot(layers, a_vals, 'r-o', lw=2.5, markersize=7, label='Operand A')
        ax.plot(layers, b_vals, 'b-s', lw=2.5, markersize=7, label='Operand B')
        ax.plot(layers, o_vals, 'g-^', lw=2, markersize=7, label='Ones digit')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Probe Accuracy', fontsize=12)
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        ax.set_title(token_name, fontsize=13, fontweight='bold')
    plt.suptitle('Phase 207: Register Persistence\nDo registers survive across autoregressive tokens?',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase207_persistence.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 207] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
