# -*- coding: utf-8 -*-
"""
Phase 113: Temperature Physics of Truth (Opus's addition)
Does the 0.94 constant change with sampling temperature?
If it's truly architectural, temperature shouldn't affect the BEST LAYER.
But temperature should affect which facts are recalled at the output.
Tests multiple temperatures on Qwen-1.5B Logit Lens.
GPU.
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
    ("The largest planet in the solar system is", " Jupiter"),
    ("The chemical symbol for gold is", " Au"),
    ("The tallest mountain in the world is", " Mount"),
    ("The first president of the United States was", " George"),
    ("The chemical formula for water is", " H"),
]

TEMPERATURES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

def logit_lens_layer(model, tok, prompt, layer_idx):
    hidden = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden['h'] = output[0][:, -1, :].detach()
        else:
            hidden['h'] = output[:, -1, :].detach()
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**tok(prompt, return_tensors='pt').to(DEVICE))
    handle.remove()
    h = model.model.norm(hidden['h'])
    logits = model.lm_head(h).squeeze()
    return logits

def main():
    print(f"[P113] Temperature Physics of Truth (device={DEVICE})")

    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B', local_files_only=True, torch_dtype=torch.float16
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', local_files_only=True)
    n_layers = model.config.num_hidden_layers

    results = {}
    for temp in TEMPERATURES:
        layer_accs = {}
        for l in range(n_layers):
            correct = 0
            for prompt, answer in FACTS:
                fact_tokens = tok.encode(answer)
                fact_id = fact_tokens[-1] if fact_tokens else 0
                logits = logit_lens_layer(model, tok, prompt, l)
                # Apply temperature
                scaled_logits = logits / temp
                if scaled_logits.argmax().item() == fact_id:
                    correct += 1
            layer_accs[l] = correct / len(FACTS)

        best_layer = max(layer_accs, key=layer_accs.get)
        best_rel = best_layer / n_layers
        output_acc = layer_accs[n_layers - 1]

        results[str(temp)] = {
            'temperature': temp,
            'best_layer': best_layer,
            'best_relative': best_rel,
            'best_accuracy': layer_accs[best_layer],
            'output_accuracy': output_acc,
        }
        print(f"  T={temp:.1f}: best=L{best_layer} ({best_rel:.3f}), "
              f"best_acc={layer_accs[best_layer]:.0%}, out_acc={output_acc:.0%}")

    # Analysis: does best_layer change with temperature?
    best_layers = [results[str(t)]['best_layer'] for t in TEMPERATURES]
    invariant = len(set(best_layers)) == 1

    print(f"\n  0.94 invariant to temperature: {'YES' if invariant else 'NO'}")
    print(f"  Best layers: {best_layers}")

    out = {
        'phase': 113, 'name': 'Temperature Physics of Truth',
        'model': 'Qwen/Qwen2.5-1.5B', 'results': results,
        'temperature_invariant': invariant,
    }
    with open(os.path.join(RESULTS_DIR, 'phase113_temperature.json'), 'w') as f:
        json.dump(out, f, indent=2)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    temps = TEMPERATURES
    best_rels = [results[str(t)]['best_relative'] for t in temps]
    out_accs = [results[str(t)]['output_accuracy'] for t in temps]
    best_accs = [results[str(t)]['best_accuracy'] for t in temps]

    axes[0].plot(temps, best_rels, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    axes[0].axhline(y=0.94, color='gold', linestyle='--', linewidth=2, label='0.94')
    axes[0].set_xscale('log'); axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Best Layer (relative)'); axes[0].set_title('0.94 Constant vs Temperature')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(temps, out_accs, 'o-', color='#3498db', label='Output', linewidth=2)
    axes[1].plot(temps, best_accs, 's-', color='#2ecc71', label='Best Layer', linewidth=2)
    axes[1].set_xscale('log'); axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('Accuracy'); axes[1].set_title('Accuracy vs Temperature')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Phase 113: Temperature Physics of Truth',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase113_temperature.png'), dpi=150)
    plt.close()

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("[Phase 113] Complete.")

if __name__ == '__main__':
    main()
