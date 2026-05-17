# -*- coding: utf-8 -*-
"""
Phase 196: Liquid Circuit Lesioning
If "def" activates Code-Math neurons (P172: 1229 neurons at L14-15),
what happens when we MUTE everything EXCEPT those neurons?

Dynamic lesioning: During arithmetic, zero out all MLP neurons
except the top-N most arithmetic-responsive ones.

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

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"]
ARITH_TEST = [
    ("def f(): return 1 + 1 =", " 2"),
    ("def f(): return 3 + 4 =", " 7"),
    ("def f(): return 5 + 5 =", " 10"),
    ("def f(): return 8 + 1 =", " 9"),
    ("def f(): return 6 + 3 =", " 9"),
    ("def f(): return 4 + 4 =", " 8"),
    ("def f(): return 7 + 2 =", " 9"),
    ("def f(): return 2 + 6 =", " 8"),
    ("def f(): return 9 + 0 =", " 9"),
    ("def f(): return 2 + 7 =", " 9"),
]
FACT_TEST = [
    ("def f(): return The capital of Japan is", " Tokyo"),
    ("def f(): return The capital of France is", " Paris"),
    ("def f(): return Water freezes at", " 0"),
]

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
    print("[P196] Liquid Circuit Lesioning")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    target_layers = [14, 15]  # Code-Math layer region
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - max(1, n_layers // 4)

    # Step 1: Profile MLP activations during arithmetic
    print("  Profiling MLP activations...")
    mlp_activations = {l: [] for l in target_layers}
    for prompt, expected in ARITH_TEST:
        for l in target_layers:
            captured = [None]
            def hook_mlp(module, input, output):
                captured[0] = output.detach().float()
            # Hook the MLP's output (gate_proj or up_proj output)
            h = model.model.layers[l].mlp.register_forward_hook(hook_mlp)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                _ = model(**inp)
            h.remove()
            if captured[0] is not None:
                # Get activation magnitude per neuron (last token)
                act = captured[0][:, -1, :].abs().squeeze().cpu().numpy()
                mlp_activations[l].append(act)

    # Find top-N most active neurons during arithmetic
    results = {}
    for l in target_layers:
        acts = np.array(mlp_activations[l])
        mean_act = acts.mean(axis=0)
        n_neurons = len(mean_act)
        print(f"    L{l}: {n_neurons} MLP neurons, top activation={mean_act.max():.4f}")
        results[f'L{l}_n_neurons'] = n_neurons

    # Step 2: Evaluate with lesioning (mute bottom-N% of neurons)
    configs = {}
    for keep_pct in [100, 50, 25, 10, 5, 1]:
        hooks = []
        for l in target_layers:
            acts = np.array(mlp_activations[l])
            mean_act = acts.mean(axis=0)
            n_neurons = len(mean_act)
            n_keep = max(1, int(n_neurons * keep_pct / 100))
            top_indices = np.argsort(mean_act)[-n_keep:]
            mask = torch.zeros(n_neurons, device=DEVICE)
            mask[top_indices] = 1.0
            def mk_mask(m):
                def fn(module, input, output):
                    return output * m.unsqueeze(0).unsqueeze(0)
                return fn
            hooks.append(model.model.layers[l].mlp.register_forward_hook(mk_mask(mask)))

        correct_arith = 0
        for prompt, expected in ARITH_TEST:
            exp_id = tok.encode(expected)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()
            if logits.argmax().item() == exp_id: correct_arith += 1

        correct_fact = 0
        for prompt, expected in FACT_TEST:
            exp_id = tok.encode(expected)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()
            if logits.argmax().item() == exp_id: correct_fact += 1

        for h in hooks: h.remove()
        arith_acc = correct_arith / len(ARITH_TEST)
        fact_acc = correct_fact / len(FACT_TEST)
        configs[f'keep{keep_pct}'] = {'keep_pct': keep_pct, 'arith': arith_acc, 'fact': fact_acc}
        print(f"  Keep {keep_pct:3d}%: arith={arith_acc:.0%} fact={fact_acc:.0%}")

    results['configs'] = configs
    with open(os.path.join(RESULTS_DIR, 'phase196_lesion.json'), 'w') as f:
        json.dump({'phase': '196', 'name': 'Liquid Circuit Lesioning',
                   'results': results}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    pcts = [100, 50, 25, 10, 5, 1]
    arith_vals = [configs[f'keep{p}']['arith'] for p in pcts]
    fact_vals = [configs[f'keep{p}']['fact'] for p in pcts]
    x = np.arange(len(pcts))
    w = 0.35
    ax.bar(x-w/2, arith_vals, w, label='Arithmetic', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, fact_vals, w, label='Factual', color='#e74c3c', alpha=0.8)
    for i in range(len(pcts)):
        ax.text(x[i]-w/2, arith_vals[i]+0.02, f'{arith_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i]+w/2, fact_vals[i]+0.02, f'{fact_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'{p}%' for p in pcts], fontsize=11)
    ax.set_xlabel('% of MLP Neurons Kept (L14-15)', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.legend(fontsize=11); ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase 196: Liquid Circuit Lesioning\n'
                 'How many Code-Math neurons are needed?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase196_lesion.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    print(f"  -> 100% neurons: arith={configs['keep100']['arith']:.0%}")
    print(f"  ->   5% neurons: arith={configs['keep5']['arith']:.0%}")
    print(f"  ->   1% neurons: arith={configs['keep1']['arith']:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 196] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
