# -*- coding: utf-8 -*-
"""
Phase 108: The MLP Autopsy
Why did Frankenstein Surgery (P106) fail? Because Alignment Tax
lives in MLPs, not just Attention. Compare MLP activations
between Qwen-0.5B Base and Instruct.
GPU, float32.
"""
import torch, json, os, gc, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FACTS = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("The chemical symbol for gold is", " Au"),
    ("The tallest mountain in the world is", " Mount"),
    ("The first president of the United States was", " George"),
    ("The chemical formula for water is", " H"),
]

def get_mlp_activations(model, tok, facts, n_layers):
    """Capture MLP output norms per layer."""
    layer_norms = {l: [] for l in range(n_layers)}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # MLP output norm for last token
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            norm = h[:, -1, :].float().norm().item()
            layer_norms[layer_idx].append(norm)
        return hook_fn

    for l in range(n_layers):
        h = model.model.layers[l].mlp.register_forward_hook(make_hook(l))
        hooks.append(h)

    for prompt, answer in facts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)

    for h in hooks:
        h.remove()

    return {l: np.mean(v) for l, v in layer_norms.items()}

def main():
    print(f"[P108] The MLP Autopsy (device={DEVICE})")

    results = {}
    for model_id, label in [('Qwen/Qwen2.5-0.5B', 'Base'), ('Qwen/Qwen2.5-0.5B-Instruct', 'Instruct')]:
        print(f"  Loading {label}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32
        ).eval().to(DEVICE)
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        n_layers = model.config.num_hidden_layers

        norms = get_mlp_activations(model, tok, FACTS, n_layers)
        results[label] = {str(k): v for k, v in norms.items()}
        print(f"    Mean norm: {np.mean(list(norms.values())):.2f}")

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Compute delta
    base_norms = np.array([results['Base'][str(l)] for l in range(n_layers)])
    inst_norms = np.array([results['Instruct'][str(l)] for l in range(n_layers)])
    delta = inst_norms - base_norms
    mid = n_layers // 2

    print(f"\n  === MLP DELTA (Instruct - Base) ===")
    print(f"  Front half: {delta[:mid].mean():+.2f}")
    print(f"  Back half:  {delta[mid:].mean():+.2f}")
    print(f"  Last 3 layers: {delta[-3:].mean():+.2f}")

    # Find most amplified MLP layers (the MLPolice)
    most_amplified = np.argsort(delta)[::-1]
    print(f"  Most AMPLIFIED by Instruct (MLPolice):")
    for i in range(5):
        l = most_amplified[i]
        print(f"    L{l}: Base={base_norms[l]:.2f} -> Inst={inst_norms[l]:.2f} (delta={delta[l]:+.2f})")

    out = {
        'phase': 108, 'name': 'The MLP Autopsy',
        'base_norms': base_norms.tolist(),
        'instruct_norms': inst_norms.tolist(),
        'delta': delta.tolist(),
        'front_delta': float(delta[:mid].mean()),
        'back_delta': float(delta[mid:].mean()),
        'mlpolice_layers': [int(l) for l in most_amplified[:5]],
    }
    with open(os.path.join(RESULTS_DIR, 'phase108_mlp_autopsy.json'), 'w') as f:
        json.dump(out, f, indent=2)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    layers = range(n_layers)
    axes[0].plot(layers, base_norms, 'o-', color='#2ecc71', label='Base')
    axes[0].plot(layers, inst_norms, 's-', color='#e74c3c', label='Instruct')
    axes[0].set_xlabel('Layer'); axes[0].set_ylabel('MLP Output Norm')
    axes[0].set_title('MLP Activation Norms'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in delta]
    axes[1].bar(layers, delta, color=colors)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].axvline(x=mid, color='red', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Layer'); axes[1].set_ylabel('Delta (Inst-Base)')
    axes[1].set_title('MLP Alignment Tax per Layer')

    axes[2].bar(['Front\n(Semantic)', 'Back\n(Syntactic)', 'Last 3\n(MLPolice)'],
               [delta[:mid].mean(), delta[mid:].mean(), delta[-3:].mean()],
               color=['#2ecc71', '#e74c3c', '#c0392b'])
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].set_ylabel('Mean Delta'); axes[2].set_title('MLP Tax by Region')

    fig.suptitle('Phase 108: The MLP Autopsy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase108_mlp_autopsy.png'), dpi=150)
    plt.close()
    print("[Phase 108] Complete.")

if __name__ == '__main__':
    main()
