# -*- coding: utf-8 -*-
"""
Phase 111: The Aletheia Decoder
THE practical application: inject entropy noise into suppressor layers
to improve factual accuracy WITHOUT changing the prompt.
If this works, it's a publishable algorithm.
GPU, Qwen-0.5B.
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
    ("The boiling point of water is", " 100"),
    ("The atomic number of carbon is", " 6"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("The speed of sound is approximately", " 343"),
    ("Photosynthesis converts sunlight into", " chemical"),
]

def eval_accuracy(model, tok, facts):
    correct = 0
    total_rank = 0
    for prompt, answer in facts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        fact_tokens = tok.encode(answer)
        fact_id = fact_tokens[-1] if fact_tokens else 0
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
        rank = (logits.argsort(descending=True) == fact_id).nonzero()
        rank = rank.item() + 1 if rank.numel() > 0 else logits.shape[0]
        if rank == 1: correct += 1
        total_rank += rank
    return correct / len(facts), total_rank / len(facts)

def main():
    print(f"[P111] The Aletheia Decoder (device={DEVICE})")

    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', local_files_only=True, torch_dtype=torch.float32
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', local_files_only=True)
    n_layers = model.config.num_hidden_layers

    # Baseline
    base_acc, base_rank = eval_accuracy(model, tok, FACTS)
    print(f"  Baseline: acc={base_acc:.0%}, rank={base_rank:.1f}")

    # Test different entropy injection strategies
    results = {'baseline': {'accuracy': base_acc, 'mean_rank': base_rank}}

    # Strategy: inject noise into attention output of back-half layers
    # P105 showed L20-L23 are most affected by alignment
    for noise_scale in [0.01, 0.05, 0.1, 0.5, 1.0]:
        hooks = []
        # Inject noise into last 4 layers' attention output
        target_layers = list(range(n_layers - 4, n_layers))

        def make_hook(scale):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    noise = torch.randn_like(h) * scale
                    return (h + noise,) + output[1:]
                else:
                    noise = torch.randn_like(output) * scale
                    return output + noise
            return hook_fn

        for l in target_layers:
            h = model.model.layers[l].self_attn.register_forward_hook(make_hook(noise_scale))
            hooks.append(h)

        acc, rank = eval_accuracy(model, tok, FACTS)
        for h in hooks:
            h.remove()

        label = f"noise_{noise_scale}"
        results[label] = {'accuracy': acc, 'mean_rank': rank, 'noise_scale': noise_scale,
                         'target_layers': target_layers}
        delta = acc - base_acc
        print(f"  Noise={noise_scale:.2f} (L{target_layers[0]}-L{target_layers[-1]}): "
              f"acc={acc:.0%} ({'+' if delta >= 0 else ''}{delta:.0%}), rank={rank:.1f}")

    # Strategy 2: Scale down MLP output of back layers
    for scale_factor in [0.5, 0.3, 0.1, 0.0]:
        hooks = []
        target_layers = list(range(n_layers - 4, n_layers))

        def make_mlp_hook(factor):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (output[0] * factor,) + output[1:]
                return output * factor
            return hook_fn

        for l in target_layers:
            h = model.model.layers[l].mlp.register_forward_hook(make_mlp_hook(scale_factor))
            hooks.append(h)

        acc, rank = eval_accuracy(model, tok, FACTS)
        for h in hooks:
            h.remove()

        label = f"mlp_scale_{scale_factor}"
        results[label] = {'accuracy': acc, 'mean_rank': rank, 'scale_factor': scale_factor}
        delta = acc - base_acc
        print(f"  MLP scale={scale_factor:.1f} (L{target_layers[0]}-L{target_layers[-1]}): "
              f"acc={acc:.0%} ({'+' if delta >= 0 else ''}{delta:.0%}), rank={rank:.1f}")

    # Best result
    best_key = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\n  BEST: {best_key} -> {results[best_key]['accuracy']:.0%}")
    print(f"  Improvement: {results[best_key]['accuracy'] - base_acc:+.0%}")

    out = {'phase': 111, 'name': 'The Aletheia Decoder',
           'model': 'Qwen/Qwen2.5-0.5B', 'n_layers': n_layers, 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase111_aletheia_decoder.json'), 'w') as f:
        json.dump(out, f, indent=2)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Noise sweep
    noise_keys = [k for k in results if k.startswith('noise_')]
    noise_vals = [results[k]['noise_scale'] for k in noise_keys]
    noise_accs = [results[k]['accuracy'] for k in noise_keys]
    axes[0].plot(noise_vals, noise_accs, 'o-', color='#e74c3c', linewidth=2)
    axes[0].axhline(y=base_acc, color='gray', linestyle='--', label=f'Baseline ({base_acc:.0%})')
    axes[0].set_xlabel('Noise Scale'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Attention Noise Injection'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # MLP scale sweep
    mlp_keys = [k for k in results if k.startswith('mlp_')]
    mlp_vals = [results[k]['scale_factor'] for k in mlp_keys]
    mlp_accs = [results[k]['accuracy'] for k in mlp_keys]
    axes[1].plot(mlp_vals, mlp_accs, 's-', color='#2ecc71', linewidth=2)
    axes[1].axhline(y=base_acc, color='gray', linestyle='--', label=f'Baseline ({base_acc:.0%})')
    axes[1].set_xlabel('MLP Scale Factor'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('MLP Suppression Surgery'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Phase 111: The Aletheia Decoder', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase111_aletheia_decoder.png'), dpi=150)
    plt.close()

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("[Phase 111] Complete.")

if __name__ == '__main__':
    main()
