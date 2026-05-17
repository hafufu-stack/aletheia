# -*- coding: utf-8 -*-
"""
Phase 197: Turing Completeness Probe
P190/192: LLM has arithmetic registers (carry, ones, sum).
Does it also have LOGIC registers (boolean, comparison)?

Probe hidden states during conditional reasoning:
  "If 5 > 3, output 1, else 0" -> Is "5>3=True" stored as a register?

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

def gen_comparisons():
    """Generate A > B comparison problems."""
    data = []
    for a in range(1, 10):
        for b in range(1, 10):
            if a == b: continue
            result = 1 if a > b else 0
            # "def f(): return 1 if A > B else 0" style
            prompt = f"def f(): return 1 if {a} > {b} else 0\n# f() ="
            data.append((prompt, result, a, b))
    return data

def gen_even_odd():
    """Is N even?"""
    data = []
    for n in range(20):
        is_even = 1 if n % 2 == 0 else 0
        prompt = f"def f(): return {n} % 2 ==\n# f() ="
        data.append((prompt, is_even, n))
    return data

def main():
    print("[P197] Turing Completeness Probe")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    target_layers = [10, 14, 15, 18, 20, 22]

    # Task 1: Comparison (A > B ?)
    print("\n  === Task 1: Comparison Register (A > B ?) ===")
    comparisons = gen_comparisons()
    hiddens = {l: [] for l in target_layers}
    labels = []
    captured = [None]
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured[0] = output[0][:, -1, :].detach().cpu().numpy().flatten()
        else:
            captured[0] = output[:, -1, :].detach().cpu().numpy().flatten()

    for prompt, result, a, b in comparisons:
        for l in target_layers:
            h = model.model.layers[l].register_forward_hook(hook_fn)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                _ = model(**inp)
            h.remove()
            hiddens[l].append(captured[0].copy())
        labels.append(result)

    labels = np.array(labels)
    comp_results = {}
    for l in target_layers:
        X = np.array(hiddens[l])
        idx = np.random.RandomState(42).permutation(len(X))
        n_tr = int(0.7 * len(X))
        tr, te = idx[:n_tr], idx[n_tr:]
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X[tr], labels[tr])
        acc = accuracy_score(labels[te], clf.predict(X[te]))
        comp_results[l] = acc
        marker = " ***" if acc > 0.7 else ""
        print(f"    L{l:2d}: A>B probe = {acc:.0%}{marker}")

    # Task 2: Even/Odd
    print("\n  === Task 2: Even/Odd Register ===")
    evens = gen_even_odd()
    hiddens2 = {l: [] for l in target_layers}
    labels2 = []
    for prompt, is_even, n in evens:
        for l in target_layers:
            h = model.model.layers[l].register_forward_hook(hook_fn)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                _ = model(**inp)
            h.remove()
            hiddens2[l].append(captured[0].copy())
        labels2.append(is_even)
    labels2 = np.array(labels2)
    eo_results = {}
    for l in target_layers:
        X = np.array(hiddens2[l])
        idx = np.random.RandomState(42).permutation(len(X))
        n_tr = int(0.7 * len(X))
        tr, te = idx[:n_tr], idx[n_tr:]
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X[tr], labels2[tr])
        acc = accuracy_score(labels2[te], clf.predict(X[te]))
        eo_results[l] = acc
        marker = " ***" if acc > 0.7 else ""
        print(f"    L{l:2d}: Even/Odd probe = {acc:.0%}{marker}")

    all_results = {'comparison': {str(l): v for l, v in comp_results.items()},
                   'even_odd': {str(l): v for l, v in eo_results.items()}}

    with open(os.path.join(RESULTS_DIR, 'phase197_turing.json'), 'w') as f:
        json.dump({'phase': '197', 'name': 'Turing Completeness Probe',
                   'results': all_results}, f, indent=2, default=str)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, (task, task_results, title) in enumerate([
        ('comparison', comp_results, 'A > B (Boolean Register)'),
        ('even_odd', eo_results, 'Even/Odd (Modular Register)')
    ]):
        ax = axes[i]
        layers = target_layers
        vals = [task_results[l] for l in layers]
        colors = ['#e74c3c' if v > 0.7 else '#3498db' for v in vals]
        ax.bar([f'L{l}' for l in layers], vals, color=colors, alpha=0.8)
        for j, v in enumerate(vals):
            ax.text(j, v+0.02, f'{v:.0%}', ha='center', fontsize=11, fontweight='bold')
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5, label='Chance')
        ax.set_ylabel('Probe Accuracy', fontsize=12)
        ax.legend(fontsize=9); ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(title, fontsize=13, fontweight='bold')
    plt.suptitle('Phase 197: Turing Completeness Probe\n'
                 'Does the LLM have boolean & modular registers?',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase197_turing.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    best_comp = max(comp_results.values())
    best_eo = max(eo_results.values())
    print(f"\n  === VERDICT ===")
    print(f"  -> Best comparison probe: {best_comp:.0%}")
    print(f"  -> Best even/odd probe: {best_eo:.0%}")
    if best_comp > 0.7:
        print("  -> BOOLEAN REGISTER FOUND! LLM has IF/ELSE capability")
    if best_eo > 0.7:
        print("  -> MODULAR REGISTER FOUND! LLM can compute mod()")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 197] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
