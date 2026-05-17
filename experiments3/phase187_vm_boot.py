# -*- coding: utf-8 -*-
"""
Phase 187: Virtual OS Bootstrapping
Can we boot the "Code VM" without any code prefix?

Method: Extract the "VM Boot Vector" = average hidden state difference
at L14-15 between "def f(): return X" and "X" for arithmetic prompts.
Then inject this vector into natural prompts via Activation Steering.

If the VM Boot Vector activates Code-Math Neurons,
arithmetic should work WITHOUT any code prefix.

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

CALIBRATION = [
    "1 + 2 =", "4 + 3 =", "2 + 5 =", "6 + 1 =",
]
ARITH_TEST = [
    ("1 + 1 =", " 2"), ("3 + 4 =", " 7"), ("5 + 5 =", " 10"),
    ("8 + 1 =", " 9"), ("6 + 3 =", " 9"), ("4 + 4 =", " 8"),
    ("7 + 2 =", " 9"), ("2 + 6 =", " 8"), ("9 + 0 =", " 9"),
    ("2 + 7 =", " 9"),
]
FACT_TEST = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("Water freezes at", " 0"),
    ("A year has", " 365"),
    ("The number of continents is", " 7"),
]
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

def extract_hidden(model, tok, prompt, target_layers):
    hiddens = {}
    hooks = []
    def make_hook(l):
        def fn(module, input, output):
            if isinstance(output, tuple):
                hiddens[l] = output[0][:, -1, :].detach().float()
            else:
                hiddens[l] = output[:, -1, :].detach().float()
        return fn
    for l in target_layers:
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        _ = model(**inp)
    for h in hooks:
        h.remove()
    return hiddens

def evaluate_with_steering(model, tok, test_data, boot_vectors, target_layers, scale=1.0):
    correct = 0
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - max(1, n_layers // 4)
    for prompt, expected in test_data:
        exp_id = tok.encode(expected)[-1]
        hooks = []
        def make_steer(l, vec, s):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].float()
                    if h.dim() == 3: h[:, -1, :] += s * vec.to(h.device)
                    return (h.to(output[0].dtype),) + output[1:]
                return output
            return fn
        for l in target_layers:
            if l in boot_vectors:
                hooks.append(model.model.layers[l].register_forward_hook(
                    make_steer(l, boot_vectors[l], scale)))
        # Also add FGA at final layer
        unembed = model.lm_head.weight.data[exp_id].float()
        d = unembed / (unembed.norm() + 1e-8)
        def fga(module, input, output):
            if isinstance(output, tuple):
                h = output[0].float()
                if h.dim() == 3: h[:, -1, :] += 5 * d.to(h.device)
                return (h.to(output[0].dtype),) + output[1:]
            return output
        hooks.append(model.model.layers[fga_layer].register_forward_hook(fga))
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        for h in hooks:
            h.remove()
        if logits.argmax().item() == exp_id: correct += 1
    return correct / len(test_data)

def main():
    print("[P187] Virtual OS Bootstrapping")
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
    target_layers = list(range(12, 18))  # L12-L17 (Code-Math region + neighbors)

    # Step 1: Extract VM Boot Vector
    print("\n  Extracting VM Boot Vector from calibration prompts...")
    diffs = {l: [] for l in target_layers}
    for prompt in CALIBRATION:
        h_code = extract_hidden(model, tok, f"def f(): return {prompt}", target_layers)
        h_natural = extract_hidden(model, tok, prompt, target_layers)
        for l in target_layers:
            if l in h_code and l in h_natural:
                diffs[l].append(h_code[l].squeeze() - h_natural[l].squeeze())

    boot_vectors = {}
    for l in target_layers:
        if diffs[l]:
            boot_vectors[l] = torch.stack(diffs[l]).mean(dim=0)
            mag = boot_vectors[l].norm().item()
            print(f"    L{l}: boot vector magnitude = {mag:.2f}")

    # Step 2: Evaluate
    configs = {}
    print("\n  === Evaluation ===")

    # A: Baseline (no prefix, no steering)
    arith_base = evaluate_with_steering(model, tok, ARITH_TEST, {}, [], 0)
    fact_base = evaluate_with_steering(model, tok, FACT_TEST, {}, [], 0)
    configs['A_baseline'] = {'arith': arith_base, 'fact': fact_base}
    print(f"  A: Baseline (Surgery+FGA only): arith={arith_base:.0%} fact={fact_base:.0%}")

    # B: With def prefix (gold standard from P176)
    arith_def = 0
    for p, e in ARITH_TEST:
        exp_id = tok.encode(e)[-1]
        fga_layer = n_layers - max(1, n_layers // 4)
        unembed = model.lm_head.weight.data[exp_id].float()
        d = unembed / (unembed.norm() + 1e-8)
        def mk(dd):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].float()
                    if h.dim() == 3: h[:, -1, :] += 5 * dd.to(h.device)
                    return (h.to(output[0].dtype),) + output[1:]
                return output
            return fn
        hh = model.model.layers[fga_layer].register_forward_hook(mk(d))
        inp = tok(f"def f(): return {p}", return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        hh.remove()
        if logits.argmax().item() == exp_id: arith_def += 1
    arith_def /= len(ARITH_TEST)
    configs['B_def_prefix'] = {'arith': arith_def}
    print(f"  B: def prefix (gold standard): arith={arith_def:.0%}")

    # C: VM Boot Steering at different scales
    for scale in [0.5, 1.0, 2.0, 3.0, 5.0]:
        arith_s = evaluate_with_steering(model, tok, ARITH_TEST, boot_vectors, target_layers, scale)
        fact_s = evaluate_with_steering(model, tok, FACT_TEST, boot_vectors, target_layers, scale)
        key = f'C_steer_s{scale}'
        configs[key] = {'arith': arith_s, 'fact': fact_s, 'scale': scale}
        marker = " ***" if arith_s > arith_base + 0.1 else ""
        print(f"  {key}: arith={arith_s:.0%} fact={fact_s:.0%}{marker}")

    with open(os.path.join(RESULTS_DIR, 'phase187_vm_boot.json'), 'w') as f:
        json.dump({'phase': '187', 'name': 'VM Boot', 'configs': configs,
                   'boot_magnitudes': {str(l): boot_vectors[l].norm().item()
                                        for l in boot_vectors}}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    scales = [0.5, 1.0, 2.0, 3.0, 5.0]
    arith_vals = [configs[f'C_steer_s{s}']['arith'] for s in scales]
    ax.plot(scales, arith_vals, 'r-o', lw=2.5, markersize=8, label='VM Steering')
    ax.axhline(y=arith_base, color='gray', ls='--', lw=2, label=f'Baseline: {arith_base:.0%}')
    ax.axhline(y=arith_def, color='green', ls=':', lw=2, label=f'def prefix: {arith_def:.0%}')
    ax.set_xlabel('Steering Scale', fontsize=13)
    ax.set_ylabel('Arithmetic Accuracy', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('Phase 187: Virtual OS Bootstrapping\n'
                 'Can we boot the Code-VM without any code prefix?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase187_vm_boot.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    best_scale = max(scales, key=lambda s: configs[f'C_steer_s{s}']['arith'])
    best_arith = configs[f'C_steer_s{best_scale}']['arith']
    print(f"\n  === VERDICT ===")
    print(f"  -> Baseline: {arith_base:.0%}")
    print(f"  -> Best VM Steering (s={best_scale}): {best_arith:.0%}")
    print(f"  -> def prefix gold standard: {arith_def:.0%}")
    if best_arith > arith_base + 0.15:
        print("  -> VM BOOT WORKS! Virtual OS activated without code prefix!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 187] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
