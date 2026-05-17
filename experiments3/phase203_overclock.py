# -*- coding: utf-8 -*-
"""
Phase 203: Neural Overclocking (Simplified)
Instead of physically looping layers (which requires complex state management),
use FGA at MULTIPLE execute-phase layers simultaneously to amplify computation.

Hypothesis: If the execute phase (L11-17) is too short, applying FGA
at ALL execute layers (not just one) should give a stronger effect.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time
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

# Test: all single-digit additions (the hard ones are carry problems)
def gen_additions():
    return [(a, b, a+b) for a in range(10) for b in range(10)]

def main():
    print("[P203] Neural Overclocking (Multi-Layer FGA)")
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
    additions = gen_additions()

    configs = {}

    # Config A: Baseline (no FGA)
    print("  Baseline...")
    correct = 0
    for a, b, s in additions:
        prompt = f"def f(): return {a} + {b} ="
        exp_id = tok.encode(f" {s}")[-1]
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if logits.argmax().item() == exp_id: correct += 1
    configs['baseline'] = correct / len(additions)
    print(f"    Baseline: {configs['baseline']:.0%}")

    # Config B: FGA at single layer (L18, traditional)
    print("  Single-layer FGA (L18)...")
    correct = 0
    for a, b, s in additions:
        prompt = f"def f(): return {a} + {b} ="
        exp_id = tok.encode(f" {s}")[-1]
        unembed = model.lm_head.weight.data[exp_id].float()
        d = unembed / (unembed.norm() + 1e-8)
        def mk(dd, gg):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].float()
                    if h.dim() == 3: h[:, -1, :] += gg * dd.to(h.device)
                    return (h.to(output[0].dtype),) + output[1:]
                return output
            return fn
        handle = model.model.layers[18].register_forward_hook(mk(d, 5))
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        handle.remove()
        if logits.argmax().item() == exp_id: correct += 1
    configs['fga_single'] = correct / len(additions)
    print(f"    Single FGA: {configs['fga_single']:.0%}")

    # Config C: FGA at ALL execute layers (L11-L17) - "overclocked"
    for n_exec_layers in [3, 5, 7]:
        exec_layers = list(range(18-n_exec_layers, 18+1))  # centered on L18
        label = f'fga_{n_exec_layers}layers'
        print(f"  Multi-layer FGA ({n_exec_layers} layers: L{exec_layers[0]}-L{exec_layers[-1]})...")
        correct = 0
        for a, b, s in additions:
            prompt = f"def f(): return {a} + {b} ="
            exp_id = tok.encode(f" {s}")[-1]
            unembed = model.lm_head.weight.data[exp_id].float()
            d = unembed / (unembed.norm() + 1e-8)
            # Apply FGA at each layer with reduced strength (to avoid oversaturation)
            g = 5 / n_exec_layers  # distribute gain across layers
            hooks = []
            for el in exec_layers:
                if 0 <= el < n_layers:
                    hooks.append(model.model.layers[el].register_forward_hook(mk(d, g)))
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()
            for h in hooks: h.remove()
            if logits.argmax().item() == exp_id: correct += 1
        configs[label] = correct / len(additions)
        print(f"    {label}: {configs[label]:.0%}")

    # Config D: FGA at ALL layers (L0-L23) - "maximum overclock"
    print("  Full-stack FGA (all layers)...")
    correct = 0
    for a, b, s in additions:
        prompt = f"def f(): return {a} + {b} ="
        exp_id = tok.encode(f" {s}")[-1]
        unembed = model.lm_head.weight.data[exp_id].float()
        d = unembed / (unembed.norm() + 1e-8)
        g = 5 / n_layers
        hooks = []
        for el in range(n_layers):
            hooks.append(model.model.layers[el].register_forward_hook(mk(d, g)))
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        for h in hooks: h.remove()
        if logits.argmax().item() == exp_id: correct += 1
    configs['fga_all'] = correct / len(additions)
    print(f"    Full-stack: {configs['fga_all']:.0%}")

    with open(os.path.join(RESULTS_DIR, 'phase203_overclock.json'), 'w') as f:
        json.dump({'phase': '203', 'name': 'Neural Overclocking',
                   'results': configs}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    methods = list(configs.keys())
    vals = [configs[m] for m in methods]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c', '#1abc9c']
    ax.bar(methods, vals, color=colors[:len(methods)], alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (100 additions)', fontsize=12)
    ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(methods, rotation=15, fontsize=10)
    ax.set_title('Phase 203: Neural Overclocking\n'
                 'Does distributing FGA across more layers improve accuracy?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase203_overclock.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    for m, v in configs.items():
        print(f"  -> {m}: {v:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 203] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
