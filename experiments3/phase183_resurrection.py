# -*- coding: utf-8 -*-
"""
Phase 183: Logit Lens Resurrection at Scale
P177/P180: Logit Lens = 0% on 0.5B (all layers). Too shallow.
Test: Does Logit Lens work on Qwen2.5-1.5B?

Model: Qwen2.5-1.5B (GPU, float16 to fit in VRAM)
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

FACT_TEST = [
    ("# The capital of Japan is", " Tokyo"),
    ("# The capital of France is", " Paris"),
    ("# The largest planet is", " Jupiter"),
    ("# Water freezes at", " 0"),
    ("# A year has", " 365"),
    ("# The number of continents is", " 7"),
    ("# Pi is approximately", " 3"),
    ("# The boiling point of water is", " 100"),
]

def logit_lens_all_layers(model, tok, prompt):
    n_layers = model.config.num_hidden_layers
    hiddens = {}
    hooks = []
    def make_hook(l):
        def fn(module, input, output):
            if isinstance(output, tuple):
                hiddens[l] = output[0][:, -1, :].detach().float()
            else:
                hiddens[l] = output[:, -1, :].detach().float()
        return fn
    for l in range(n_layers):
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        final_logits = model(**inp).logits[0, -1, :].float()
    for h in hooks:
        h.remove()
    layer_preds = {}
    for l in range(n_layers):
        if l in hiddens:
            logits = model.lm_head(hiddens[l].to(model.lm_head.weight.dtype)).float()
            layer_preds[l] = logits.argmax(dim=-1).item()
    final_pred = final_logits.argmax().item()
    return layer_preds, final_pred

def main():
    print("[P183] Logit Lens Resurrection at Scale")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Test on 1.5B
    model_id = 'Qwen/Qwen2.5-1.5B'
    print(f"\n  Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float16).to(DEVICE)
    n_layers = model.config.num_hidden_layers

    print(f"  d={model.config.hidden_size}, layers={n_layers}")

    # Logit Lens scan
    layer_accuracy = {l: 0 for l in range(n_layers)}
    first_correct = {}

    print(f"\n  === Logit Lens Scan ({n_layers} layers) ===")
    for prompt, expected in FACT_TEST:
        exp_id = tok.encode(expected)[-1]
        preds, final_pred = logit_lens_all_layers(model, tok, prompt)
        hits = []
        for l in range(n_layers):
            if l in preds and preds[l] == exp_id:
                layer_accuracy[l] += 1
                hits.append(l)
        first_hit = min(hits) if hits else -1
        first_correct[expected.strip()] = first_hit
        final_ok = "OK" if final_pred == exp_id else "MISS"
        print(f"    '{expected.strip():>8s}': first@L{first_hit if first_hit>=0 else 'NONE':>4s} "
              f"total={len(hits)}/{n_layers} final={final_ok}")

    # Normalize
    for l in layer_accuracy:
        layer_accuracy[l] /= len(FACT_TEST)

    # Find resurrection layer
    resurrection_layer = None
    for l in range(n_layers):
        if layer_accuracy[l] >= 0.5:
            resurrection_layer = l
            break

    print(f"\n  Resurrection layer (>=50%): {'L'+str(resurrection_layer) if resurrection_layer else 'NONE'}")

    # Compare with 0.5B
    print("\n  === 0.5B Comparison ===")
    model_05 = 'Qwen/Qwen2.5-0.5B'
    del model; gc.collect(); torch.cuda.empty_cache()
    tok05 = AutoTokenizer.from_pretrained(model_05, local_files_only=True)
    if tok05.pad_token is None: tok05.pad_token = tok05.eos_token
    m05 = AutoModelForCausalLM.from_pretrained(model_05, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    n05 = m05.config.num_hidden_layers
    layer_acc_05 = {l: 0 for l in range(n05)}
    for prompt, expected in FACT_TEST:
        exp_id = tok05.encode(expected)[-1]
        preds, _ = logit_lens_all_layers(m05, tok05, prompt)
        for l in range(n05):
            if l in preds and preds[l] == exp_id:
                layer_acc_05[l] += 1
    for l in layer_acc_05:
        layer_acc_05[l] /= len(FACT_TEST)
    del m05; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase183_resurrection.json'), 'w') as f:
        json.dump({'phase': '183', 'name': 'Logit Lens Resurrection',
                   'layer_accuracy_1_5B': {str(k): v for k, v in layer_accuracy.items()},
                   'layer_accuracy_0_5B': {str(k): v for k, v in layer_acc_05.items()},
                   'resurrection_layer': resurrection_layer,
                   'first_correct': first_correct,
                   'n_layers_1_5B': n_layers, 'n_layers_0_5B': n05}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    # Normalize x-axis to [0, 1] for comparison
    layers_15 = sorted(layer_accuracy.keys())
    layers_05 = sorted(layer_acc_05.keys())
    x_15 = [l / (n_layers - 1) for l in layers_15]
    x_05 = [l / (n05 - 1) for l in layers_05]
    acc_15 = [layer_accuracy[l] for l in layers_15]
    acc_05 = [layer_acc_05[l] for l in layers_05]
    ax.plot(x_15, acc_15, 'r-o', lw=2.5, markersize=5, label=f'Qwen-1.5B ({n_layers}L)')
    ax.plot(x_05, acc_05, 'b-s', lw=2, markersize=4, alpha=0.7, label=f'Qwen-0.5B ({n05}L)')
    if resurrection_layer is not None:
        ax.axvline(x=resurrection_layer/(n_layers-1), color='green', ls='--', lw=2, alpha=0.7,
                   label=f'Resurrection: L{resurrection_layer}')
    ax.set_xlabel('Relative Layer Position (0=input, 1=output)', fontsize=12)
    ax.set_ylabel('Logit Lens Accuracy', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Phase 183: Logit Lens Resurrection at Scale\n'
                 'Does deeper = better internal truth?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase183_resurrection.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    if resurrection_layer is not None:
        print(f"  -> LOGIT LENS RESURRECTED at L{resurrection_layer}!")
        print(f"     1.5B peak accuracy: {max(layer_accuracy.values()):.0%}")
    else:
        peak = max(layer_accuracy.values()) if layer_accuracy else 0
        print(f"  -> No resurrection (peak: {peak:.0%})")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 183] Complete.")

if __name__ == '__main__':
    main()
