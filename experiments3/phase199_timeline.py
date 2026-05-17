# -*- coding: utf-8 -*-
"""
Phase 199: The Full Execution Timeline
P198 showed: A,B loaded at L10-15, Sum computed at L20-22.
But only 7 layers were tested. Map ALL 24 layers to get the
complete "execution timeline" of LLM arithmetic.

Model: Qwen2.5-0.5B (GPU, 24 layers)
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
    print("[P199] The Full Execution Timeline")
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
    all_layers = list(range(n_layers))

    # Generate data
    data = []
    for a in range(10):
        for b in range(10):
            data.append((a, b, a+b, 1 if a+b >= 10 else 0))

    labels_a = np.array([d[0] for d in data])
    labels_b = np.array([d[1] for d in data])
    labels_sum = np.array([d[2] for d in data])
    labels_carry = np.array([d[3] for d in data])

    # Collect hidden states for ALL layers in one pass
    print(f"  Collecting {n_layers}-layer hidden states for {len(data)} problems...")
    all_hiddens = {l: [] for l in all_layers}
    captured = {}
    def make_hook(layer_idx):
        def fn(module, input, output):
            if isinstance(output, tuple):
                captured[layer_idx] = output[0][:, -1, :].detach().cpu().numpy().flatten()
            else:
                captured[layer_idx] = output[:, -1, :].detach().cpu().numpy().flatten()
        return fn

    for i, (a, b, s, c) in enumerate(data):
        prompt = f"def f(): return {a} + {b} ="
        captured.clear()
        hooks = []
        for l in all_layers:
            hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            _ = model(**inp)
        for h in hooks:
            h.remove()
        for l in all_layers:
            all_hiddens[l].append(captured[l].copy())
        if (i+1) % 25 == 0:
            print(f"    [{i+1}/{len(data)}]")

    # Probe each layer
    results = {}
    print("\n  === Full Timeline ===")
    idx = np.random.RandomState(42).permutation(len(data))
    tr, te = idx[:70], idx[70:]

    for l in all_layers:
        X = np.array(all_hiddens[l])
        # Operand A
        clf_a = LogisticRegression(max_iter=1000, random_state=42)
        clf_a.fit(X[tr], labels_a[tr])
        acc_a = accuracy_score(labels_a[te], clf_a.predict(X[te]))
        # Operand B
        clf_b = LogisticRegression(max_iter=1000, random_state=42)
        clf_b.fit(X[tr], labels_b[tr])
        acc_b = accuracy_score(labels_b[te], clf_b.predict(X[te]))
        # Sum
        clf_s = LogisticRegression(max_iter=1000, random_state=42)
        clf_s.fit(X[tr], labels_sum[tr])
        acc_s = accuracy_score(labels_sum[te], clf_s.predict(X[te]))
        # Carry
        clf_c = LogisticRegression(max_iter=1000, random_state=42)
        clf_c.fit(X[tr], labels_carry[tr])
        acc_c = accuracy_score(labels_carry[te], clf_c.predict(X[te]))

        results[str(l)] = {'A': acc_a, 'B': acc_b, 'Sum': acc_s, 'Carry': acc_c}
        print(f"    L{l:2d}: A={acc_a:.0%} B={acc_b:.0%} Sum={acc_s:.0%} Carry={acc_c:.0%}")

    with open(os.path.join(RESULTS_DIR, 'phase199_timeline.json'), 'w') as f:
        json.dump({'phase': '199', 'name': 'Full Execution Timeline',
                   'n_layers': n_layers, 'results': results}, f, indent=2, default=str)

    # Beautiful timeline visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    layers = all_layers
    a_vals = [results[str(l)]['A'] for l in layers]
    b_vals = [results[str(l)]['B'] for l in layers]
    s_vals = [results[str(l)]['Sum'] for l in layers]
    c_vals = [results[str(l)]['Carry'] for l in layers]
    ax.plot(layers, a_vals, 'r-o', lw=2.5, markersize=5, label='Operand A', alpha=0.9)
    ax.plot(layers, b_vals, 'b-s', lw=2.5, markersize=5, label='Operand B', alpha=0.9)
    ax.plot(layers, s_vals, 'g-^', lw=2.5, markersize=5, label='Sum (A+B)', alpha=0.9)
    ax.plot(layers, c_vals, 'm-D', lw=2, markersize=5, label='Carry bit', alpha=0.8)
    # Annotate phases
    ax.axvspan(-0.5, 5.5, alpha=0.06, color='blue', label='Phase 1: Input')
    ax.axvspan(5.5, 15.5, alpha=0.06, color='orange', label='Phase 2: Compute')
    ax.axvspan(15.5, n_layers-0.5, alpha=0.06, color='green', label='Phase 3: Output')
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Linear Probe Accuracy', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(layers)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_title('Phase 199: The Full Execution Timeline\n'
                 'LLM as a Von Neumann Machine: Load -> Compute -> Store',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase199_timeline.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    # Find crossover points
    a_peak = max(range(n_layers), key=lambda l: results[str(l)]['A'])
    b_peak = max(range(n_layers), key=lambda l: results[str(l)]['B'])
    s_peak = max(range(n_layers), key=lambda l: results[str(l)]['Sum'])
    c_peak = max(range(n_layers), key=lambda l: results[str(l)]['Carry'])
    print(f"\n  === VERDICT ===")
    print(f"  -> Operand A peaks at L{a_peak} ({results[str(a_peak)]['A']:.0%})")
    print(f"  -> Operand B peaks at L{b_peak} ({results[str(b_peak)]['B']:.0%})")
    print(f"  -> Sum peaks at L{s_peak} ({results[str(s_peak)]['Sum']:.0%})")
    print(f"  -> Carry peaks at L{c_peak} ({results[str(c_peak)]['Carry']:.0%})")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 199] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
