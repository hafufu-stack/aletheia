# -*- coding: utf-8 -*-
"""
Phase 95: The Grand Scaling Law - GSF Across 4 GPT-2 Scales
Test GSF universality across Small (124M), Medium (345M), Large (774M), XL (1.5B).
This is THE definitive test of the Grammatical Suppression scaling law.
"""
import torch, json, os, sys, gc, numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

FACTS = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("The chemical symbol for gold is", " Au"),
    ("The speed of light is approximately", " 299"),
    ("Albert Einstein was born in", " Ul"),
    ("The tallest mountain in the world is", " Mount"),
    ("The currency of the United Kingdom is the", " pound"),
    ("The author of Romeo and Juliet is", " William"),
    ("The first president of the United States was", " George"),
    ("The chemical formula for water is", " H"),
    ("The boiling point of water is", " 100"),
    ("The atomic number of carbon is", " 6"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("The speed of sound is approximately", " 343"),
    ("Photosynthesis converts sunlight into", " chemical"),
]

MODELS = [
    ('gpt2',        'Small',  124e6,  12),
    ('gpt2-medium', 'Medium', 345e6,  24),
    ('gpt2-large',  'Large',  774e6,  36),
    ('gpt2-xl',     'XL',     1558e6, 48),
]

def logit_lens_layer(model, tok, prompt, layer_idx):
    hidden = {}
    def hook_fn(module, input, output):
        hidden['h'] = output[0][:, -1, :].detach()
    handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**tok(prompt, return_tensors='pt'))
    handle.remove()
    h = model.transformer.ln_f(hidden['h'])
    logits = model.lm_head(h).squeeze()
    return logits

def main():
    print("[P95] The Grand Scaling Law")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

    all_results = {}

    for model_name, label, n_params, n_layers in MODELS:
        print(f"\n  === {label} ({model_name}, {n_layers} layers) ===")
        try:
            model = GPT2LMHeadModel.from_pretrained(model_name, local_files_only=True).eval()
            model_tok = GPT2Tokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        # Per-layer Logit Lens
        layer_data = {}
        for layer in range(n_layers):
            ranks = []
            for prompt, answer in FACTS:
                fact_id = model_tok.encode(answer)[0]
                logits = logit_lens_layer(model, model_tok, prompt, layer)
                rank = (logits.argsort(descending=True) == fact_id).nonzero().item() + 1
                ranks.append(rank)
            median_rank = float(np.median(ranks))
            acc = sum(1 for r in ranks if r == 1) / len(FACTS)
            layer_data[layer] = {'median_rank': median_rank, 'accuracy': acc}

        # Find best and worst layers
        best_layer = min(layer_data, key=lambda l: layer_data[l]['median_rank'])
        output_layer = n_layers - 1

        # Baseline (output layer) accuracy
        baseline_acc = layer_data[output_layer]['accuracy']
        best_acc = layer_data[best_layer]['accuracy']

        # GSF magnitude: rank degradation from best to output
        gsf_mag = layer_data[output_layer]['median_rank'] - layer_data[best_layer]['median_rank']

        # Code Mode test
        code_correct = 0
        nat_correct = 0
        for prompt, answer in FACTS:
            fact_id = model_tok.encode(answer)[0]
            # Natural
            with torch.no_grad():
                logits = model(**model_tok(prompt, return_tensors='pt')).logits[0, -1, :]
            if logits.argmax().item() == fact_id:
                nat_correct += 1
            # Comment
            with torch.no_grad():
                logits = model(**model_tok(f"# {prompt}", return_tensors='pt')).logits[0, -1, :]
            if logits.argmax().item() == fact_id:
                code_correct += 1

        nat_acc = nat_correct / len(FACTS)
        code_acc = code_correct / len(FACTS)

        all_results[label] = {
            'model': model_name,
            'params': n_params,
            'n_layers': n_layers,
            'best_layer': best_layer,
            'best_layer_relative': best_layer / n_layers,
            'best_accuracy': best_acc,
            'output_accuracy': baseline_acc,
            'natural_accuracy': nat_acc,
            'code_mode_accuracy': code_acc,
            'gsf_magnitude': gsf_mag,
            'layer_data': {str(k): v for k, v in layer_data.items()},
        }
        print(f"    Best layer: L{best_layer} (relative: {best_layer/n_layers:.2f})")
        print(f"    Best acc: {best_acc:.0%}, Output acc: {baseline_acc:.0%}")
        print(f"    Natural: {nat_acc:.0%}, Code Mode: {code_acc:.0%}")
        print(f"    GSF magnitude: {gsf_mag:.1f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {'phase': 95, 'name': 'Grand Scaling Law', 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase95_grand_scaling.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Normalized layer trajectories (all models overlaid)
    for label, data in all_results.items():
        n_layers = data['n_layers']
        layers = list(range(n_layers))
        medians = [data['layer_data'][str(l)]['median_rank'] for l in layers]
        norm_layers = [l / n_layers for l in layers]
        axes[0].plot(norm_layers, medians, 'o-', label=f"{label} ({data['params']/1e6:.0f}M)",
                    markersize=3, linewidth=1.5)
    axes[0].set_xlabel('Normalized Layer Position (0=first, 1=last)')
    axes[0].set_ylabel('Median Fact Rank')
    axes[0].set_title('GSF Trajectory Across Scales')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Scaling law: GSF magnitude vs parameters
    params = [d['params'] for d in all_results.values()]
    gsf_mags = [d['gsf_magnitude'] for d in all_results.values()]
    labels_list = list(all_results.keys())
    axes[1].scatter(params, gsf_mags, s=150, c='#e74c3c', zorder=5, edgecolors='black')
    for i, lbl in enumerate(labels_list):
        axes[1].annotate(lbl, (params[i], gsf_mags[i]),
                        textcoords="offset points", xytext=(10, 5), fontsize=10)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Parameters')
    axes[1].set_ylabel('GSF Magnitude (rank degradation)')
    axes[1].set_title('GSF vs Model Scale')
    axes[1].grid(True, alpha=0.3)

    # 3. Code Mode effect across scales
    x = np.arange(len(all_results))
    w = 0.25
    nat_accs = [d['natural_accuracy'] for d in all_results.values()]
    code_accs = [d['code_mode_accuracy'] for d in all_results.values()]
    best_accs = [d['best_accuracy'] for d in all_results.values()]

    axes[2].bar(x - w, nat_accs, w, label='Natural', color='#e74c3c', alpha=0.8)
    axes[2].bar(x, code_accs, w, label='Code Mode (#)', color='#2ecc71', alpha=0.8)
    axes[2].bar(x + w, best_accs, w, label='Best Layer (Logit Lens)', color='#3498db', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels_list)
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Accuracy Across Scales')
    axes[2].legend()

    fig.suptitle('Phase 95: The Grand Scaling Law - GSF Across 4 GPT-2 Scales',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase95_grand_scaling.png'), dpi=150)
    plt.close()
    print("[Phase 95] Complete.")

if __name__ == '__main__':
    main()
