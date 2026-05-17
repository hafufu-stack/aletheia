# -*- coding: utf-8 -*-
"""
Phase 208: Natural vs Code VM
P192 showed "def" activates Code-VM. Does this change WHERE
the registers are located?

Compare register maps for:
  "3 + 4 =" (natural)
  "def f(): return 3 + 4 =" (code VM)

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
    print("[P208] Natural vs Code VM")
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

    data = [(a, b, a+b, 1 if a+b >= 10 else 0)
            for a in range(10) for b in range(10)]
    labels_a = np.array([d[0] for d in data])
    labels_b = np.array([d[1] for d in data])
    labels_carry = np.array([d[3] for d in data])

    templates = {
        'natural': "{a} + {b} =",
        'code': "def f(): return {a} + {b} =",
        'math': "Calculate: {a} + {b} =",
    }

    results = {}
    captured = {}
    def make_hook(layer_idx):
        def fn(module, input, output):
            if isinstance(output, tuple):
                captured[layer_idx] = output[0][:, -1, :].detach().cpu().numpy().flatten()
            else:
                captured[layer_idx] = output[:, -1, :].detach().cpu().numpy().flatten()
        return fn

    for tname, template in templates.items():
        print(f"\n  === Template: '{tname}' ===")
        hiddens = {l: [] for l in all_layers}
        for a, b, s, c in data:
            prompt = template.format(a=a, b=b)
            captured.clear()
            hooks = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in all_layers]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad(): _ = model(**inp)
            for h in hooks: h.remove()
            for l in all_layers:
                hiddens[l].append(captured[l].copy())

        idx = np.random.RandomState(42).permutation(len(data))
        tr, te = idx[:70], idx[70:]
        template_results = {}
        for l in all_layers:
            X = np.array(hiddens[l])
            # A
            clf_a = LogisticRegression(max_iter=1000, random_state=42)
            clf_a.fit(X[tr], labels_a[tr])
            acc_a = accuracy_score(labels_a[te], clf_a.predict(X[te]))
            # B
            clf_b = LogisticRegression(max_iter=1000, random_state=42)
            clf_b.fit(X[tr], labels_b[tr])
            acc_b = accuracy_score(labels_b[te], clf_b.predict(X[te]))
            # Carry
            clf_c = LogisticRegression(max_iter=1000, random_state=42)
            clf_c.fit(X[tr], labels_carry[tr])
            acc_c = accuracy_score(labels_carry[te], clf_c.predict(X[te]))
            template_results[str(l)] = {'A': acc_a, 'B': acc_b, 'Carry': acc_c}

        results[tname] = template_results
        # Print peak layers
        a_peak = max(all_layers, key=lambda l: template_results[str(l)]['A'])
        b_peak = max(all_layers, key=lambda l: template_results[str(l)]['B'])
        c_peak = max(all_layers, key=lambda l: template_results[str(l)]['Carry'])
        print(f"    A peaks at L{a_peak} ({template_results[str(a_peak)]['A']:.0%})")
        print(f"    B peaks at L{b_peak} ({template_results[str(b_peak)]['B']:.0%})")
        print(f"    Carry peaks at L{c_peak} ({template_results[str(c_peak)]['Carry']:.0%})")

    with open(os.path.join(RESULTS_DIR, 'phase208_vm_compare.json'), 'w') as f:
        json.dump({'phase': '208', 'name': 'Natural vs Code VM',
                   'results': results}, f, indent=2, default=str)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = {'natural': '#e74c3c', 'code': '#3498db', 'math': '#2ecc71'}
    for i, reg in enumerate(['A', 'B', 'Carry']):
        ax = axes[i]
        for tname in templates:
            vals = [results[tname][str(l)][reg] for l in all_layers]
            ax.plot(all_layers, vals, '-o', lw=2, markersize=4,
                    color=colors[tname], label=tname, alpha=0.8)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Probe Accuracy', fontsize=12)
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        ax.set_title(f'Register: {reg}', fontsize=13, fontweight='bold')
    plt.suptitle('Phase 208: Natural vs Code VM\nDoes "def" change register locations?',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase208_vm_compare.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 208] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
