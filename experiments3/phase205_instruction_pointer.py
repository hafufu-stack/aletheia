# -*- coding: utf-8 -*-
"""
Phase 205: The Instruction Pointer (Opus Addition)
In a real CPU, there's an instruction pointer tracking execution progress.
During multi-step computation like "3 + 4 + 5 = 12", can we detect
WHICH STEP the model is currently executing?

If LLM computes (3+4)=7 first, then 7+5=12, the "step counter"
should shift between these sub-computations across layers.

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
    print("[P205] The Instruction Pointer")
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
    target_layers = list(range(n_layers))

    # Compare 1-step vs 2-step computations with same operands
    # If model is at "step 1" vs "step 2", can we tell?
    data_1step = []  # a + b = (1 step)
    data_2step = []  # a + b + c = (2 steps)
    for a in range(2, 6):
        for b in range(2, 6):
            data_1step.append((f"def f(): return {a} + {b} =", 1, a+b))
            for c in range(2, 6):
                data_2step.append((f"def f(): return {a} + {b} + {c} =", 2, a+b+c))

    # Limit 2-step for balance
    np.random.seed(42)
    idx = np.random.permutation(len(data_2step))[:len(data_1step)]
    data_2step = [data_2step[i] for i in idx]

    all_data = data_1step + data_2step
    labels = np.array([d[1] for d in all_data])  # 1 or 2 steps

    # Collect hidden states
    print(f"  Collecting states for {len(all_data)} problems...")
    all_hiddens = {l: [] for l in target_layers}
    captured = {}
    def make_hook(layer_idx):
        def fn(module, input, output):
            if isinstance(output, tuple):
                captured[layer_idx] = output[0][:, -1, :].detach().cpu().numpy().flatten()
            else:
                captured[layer_idx] = output[:, -1, :].detach().cpu().numpy().flatten()
        return fn

    for prompt, steps, result in all_data:
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

    # Probe: can we tell if this is a 1-step or 2-step computation?
    results = {}
    idx_perm = np.random.RandomState(42).permutation(len(all_data))
    n_tr = int(0.7 * len(all_data))
    tr, te = idx_perm[:n_tr], idx_perm[n_tr:]

    print("\n  === Instruction Pointer Probe ===")
    for l in target_layers:
        X = np.array(all_hiddens[l])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X[tr], labels[tr])
        acc = accuracy_score(labels[te], clf.predict(X[te]))
        results[str(l)] = acc
        marker = " ***" if acc > 0.7 else ""
        print(f"    L{l:2d}: step_probe={acc:.0%}{marker}")

    # Also probe: can we detect the INTERMEDIATE result (a+b) in 2-step problems?
    print("\n  === Intermediate Result Probe (2-step only) ===")
    data_2 = [(d, i) for i, d in enumerate(all_data) if d[1] == 2]
    if len(data_2) > 10:
        inter_results = {}
        for l in [10, 14, 18, 20, 22]:
            X_2 = np.array([all_hiddens[l][i] for _, i in data_2])
            # The intermediate result is hard to probe without knowing which it is
            # Instead, probe if the hidden state encodes "complexity"
            # Use the total sum as a proxy
            sums_2 = np.array([d[0][2] for d, _ in data_2])
            if len(np.unique(sums_2)) > 1:
                from sklearn.metrics import r2_score
                from sklearn.linear_model import Ridge
                idx2 = np.random.RandomState(42).permutation(len(X_2))
                n2 = int(0.7 * len(X_2))
                clf2 = Ridge(alpha=1.0)
                clf2.fit(X_2[idx2[:n2]], sums_2[idx2[:n2]])
                preds = clf2.predict(X_2[idx2[n2:]])
                r2 = r2_score(sums_2[idx2[n2:]], preds)
                inter_results[str(l)] = r2
                print(f"    L{l:2d}: sum_r2={r2:.2f}")

    with open(os.path.join(RESULTS_DIR, 'phase205_instruction_pointer.json'), 'w') as f:
        json.dump({'phase': '205', 'name': 'The Instruction Pointer',
                   'step_probe': results,
                   'n_1step': len(data_1step), 'n_2step': len(data_2step)},
                  f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    layers = target_layers
    vals = [results[str(l)] for l in layers]
    ax.plot(layers, vals, 'purple', lw=2.5, marker='D', markersize=5, label='1-step vs 2-step')
    ax.axhline(y=0.5, color='gray', ls=':', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Classification Accuracy', fontsize=14)
    ax.set_ylim(0.3, 1.05); ax.set_xticks(layers)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_title('Phase 205: The Instruction Pointer\n'
                 'Can the model distinguish 1-step from 2-step computation?',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase205_instruction_pointer.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    best = max(results[str(l)] for l in layers)
    print(f"\n  === VERDICT ===")
    print(f"  -> Best step probe: {best:.0%}")
    if best > 0.7:
        print("  -> INSTRUCTION POINTER FOUND!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 205] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
