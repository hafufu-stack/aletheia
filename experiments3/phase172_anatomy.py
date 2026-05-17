# -*- coding: utf-8 -*-
"""
Phase 172: Anatomy of the Trinity
WHY does Code Mode protect arithmetic from Surgery damage?

Hypothesis: "# " activates "Code-Math Neurons" that bypass
the damaged number-clustering pathway.

Method: Compare MLP activations (Natural vs Code Mode) on arithmetic,
identify neurons that ONLY fire in Code Mode.

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

ARITH_PROMPTS = [
    "1 + 1 =", "3 + 4 =", "5 + 5 =", "2 + 7 =",
    "8 + 1 =", "6 + 3 =", "4 + 4 =", "9 + 0 =",
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


def capture_mlp_activations(model, tok, prompts, code_mode=False):
    """Capture MLP output activations for all layers."""
    n_layers = model.config.num_hidden_layers
    all_activations = {i: [] for i in range(n_layers)}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # MLP output is the output of the MLP block
            if isinstance(output, torch.Tensor):
                all_activations[layer_idx].append(output[:, -1, :].detach().float().cpu())
            elif isinstance(output, tuple):
                all_activations[layer_idx].append(output[0][:, -1, :].detach().float().cpu())
        return hook_fn

    # Register hooks on MLP modules
    for i in range(n_layers):
        h = model.model.layers[i].mlp.register_forward_hook(make_hook(i))
        hooks.append(h)

    for prompt in prompts:
        text = f"# {prompt}" if code_mode else prompt
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)

    for h in hooks:
        h.remove()

    # Average activations across prompts
    avg_acts = {}
    for i in range(n_layers):
        if all_activations[i]:
            avg_acts[i] = torch.stack(all_activations[i]).mean(dim=0).squeeze()
    return avg_acts


def main():
    print("[P172] Anatomy of the Trinity")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    # Capture activations in both modes
    print("\n  Capturing Natural Mode activations...")
    natural_acts = capture_mlp_activations(model, tok, ARITH_PROMPTS, code_mode=False)

    print("  Capturing Code Mode activations...")
    code_acts = capture_mlp_activations(model, tok, ARITH_PROMPTS, code_mode=True)

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.intermediate_size  # MLP intermediate dim

    # Compute per-layer and per-neuron differences
    layer_diffs = {}
    code_exclusive_neurons = {}
    total_code_exclusive = 0

    print("\n  === Per-Layer MLP Activation Analysis ===")
    for i in range(n_layers):
        if i not in natural_acts or i not in code_acts:
            continue
        nat = natural_acts[i]
        code = code_acts[i]
        diff = (code - nat).abs()
        # Normalize by natural magnitude
        nat_mag = nat.abs().mean().item()
        code_mag = code.abs().mean().item()
        diff_mag = diff.mean().item()
        relative_change = diff_mag / max(nat_mag, 1e-8)

        # Find "Code-exclusive" neurons: active in Code, dormant in Natural
        # (absolute activation > threshold in Code, < threshold in Natural)
        threshold = nat.abs().mean() + 2 * nat.abs().std()
        code_active = (code.abs() > threshold)
        nat_dormant = (nat.abs() < threshold * 0.5)
        exclusive = (code_active & nat_dormant).sum().item()
        total_neurons = code.shape[-1]

        layer_diffs[i] = {
            'nat_mag': nat_mag, 'code_mag': code_mag,
            'diff_mag': diff_mag, 'relative_change': relative_change,
            'code_exclusive': exclusive, 'total_neurons': total_neurons
        }
        code_exclusive_neurons[i] = exclusive
        total_code_exclusive += exclusive

        if i % 4 == 0 or exclusive > 0:
            print(f"    L{i:2d}: nat={nat_mag:.3f} code={code_mag:.3f} "
                  f"delta={relative_change:.2%} exclusive={exclusive}")

    # Find layers with most Code-exclusive neurons
    top_layers = sorted(code_exclusive_neurons.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Total Code-exclusive neurons: {total_code_exclusive}")
    print(f"  Top layers: {[(f'L{l}', n) for l, n in top_layers]}")

    # Also measure: which layers show the BIGGEST activation shift?
    top_delta_layers = sorted(layer_diffs.items(),
                               key=lambda x: x[1]['relative_change'], reverse=True)[:5]
    delta_str = ', '.join(f'L{l}={d["relative_change"]:.2%}' for l, d in top_delta_layers)
    print(f"  Top delta layers: {delta_str}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase172_anatomy.json'), 'w') as f:
        json.dump({'phase': '172', 'name': 'Anatomy of the Trinity',
                   'layer_diffs': {str(k): v for k, v in layer_diffs.items()},
                   'total_code_exclusive': total_code_exclusive,
                   'top_layers': [(l, n) for l, n in top_layers],
                   'top_delta': [(l, d['relative_change']) for l, d in top_delta_layers]},
                  f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    layers = sorted(layer_diffs.keys())

    # Left: Activation magnitudes
    ax = axes[0]
    nat_mags = [layer_diffs[l]['nat_mag'] for l in layers]
    code_mags = [layer_diffs[l]['code_mag'] for l in layers]
    ax.plot(layers, nat_mags, 'b-o', lw=2, markersize=4, label='Natural Mode')
    ax.plot(layers, code_mags, 'r-s', lw=2, markersize=4, label='Code Mode')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean |Activation|', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_title('MLP Activation Magnitude', fontsize=13, fontweight='bold')

    # Middle: Relative change per layer
    ax = axes[1]
    rel_changes = [layer_diffs[l]['relative_change'] for l in layers]
    colors = ['#e74c3c' if r > 0.1 else '#3498db' for r in rel_changes]
    ax.bar(layers, rel_changes, color=colors, alpha=0.8)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Relative Change', fontsize=12)
    ax.set_title('Code vs Natural: Activation Shift', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Code-exclusive neurons
    ax = axes[2]
    exclusives = [code_exclusive_neurons.get(l, 0) for l in layers]
    colors2 = ['#e74c3c' if e > 0 else '#3498db' for e in exclusives]
    ax.bar(layers, exclusives, color=colors2, alpha=0.8)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('# Code-Exclusive Neurons', fontsize=12)
    ax.set_title('Neurons Active ONLY in Code Mode', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Phase 172: Anatomy of the Trinity\n'
                 'Why does Code Mode protect arithmetic from Surgery?',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase172_anatomy.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    if total_code_exclusive > 10:
        print(f"  -> CODE-MATH NEURONS FOUND! {total_code_exclusive} exclusive neurons")
        print(f"     Concentrated in layers: {[f'L{l}' for l, n in top_layers if n > 0]}")
    else:
        top_l, top_d = top_delta_layers[0]
        print(f"  -> No exclusive neurons, but L{top_l} shows {top_d['relative_change']:.1%} activation shift")
        print(f"     Code Mode REDIRECTS existing circuits rather than activating new ones")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 172] Complete.")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
