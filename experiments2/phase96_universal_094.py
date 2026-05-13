# -*- coding: utf-8 -*-
"""
Phase 96: The Universal 0.94 Constant
Prove that the 0.94 convergence holds across different architectures.
Test Qwen2.5-0.5B and Qwen2.5-1.5B alongside GPT-2 family.
GPU-accelerated.
"""
import torch, json, os, sys, gc, numpy as np

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

def logit_lens_scan(model, tok, facts, n_layers, model_type='gpt2'):
    """Run Logit Lens across all layers."""
    layer_data = {}
    for layer_idx in range(n_layers):
        hidden_states = {}
        def make_hook(target):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states['h'] = output[0][:, -1, :].detach()
                else:
                    hidden_states['h'] = output[:, -1, :].detach()
            return hook_fn

        if model_type == 'gpt2':
            handle = model.transformer.h[layer_idx].register_forward_hook(make_hook(layer_idx))
            ln_f = model.transformer.ln_f
            lm_head = model.lm_head
        else:
            handle = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            ln_f = model.model.norm
            lm_head = model.lm_head

        ranks = []
        for prompt, answer in facts:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            fact_id = tok.encode(answer)
            if len(fact_id) > 0:
                # Some tokenizers add special tokens
                fact_id = fact_id[-1] if model_type != 'gpt2' else fact_id[0]
            else:
                fact_id = 0

            with torch.no_grad():
                model(**inp)
            h = ln_f(hidden_states['h'])
            logits = lm_head(h).squeeze()
            rank = (logits.argsort(descending=True) == fact_id).nonzero()
            if rank.numel() > 0:
                rank = rank.item() + 1
            else:
                rank = logits.shape[0]
            ranks.append(rank)

        handle.remove()
        median_rank = float(np.median(ranks))
        acc = sum(1 for r in ranks if r == 1) / len(facts)
        layer_data[layer_idx] = {'median_rank': median_rank, 'accuracy': acc}

    return layer_data

def eval_baseline(model, tok, facts, model_type='gpt2'):
    """Evaluate baseline and Code Mode accuracy."""
    results = {}
    for mode_name, prefix in [('natural', ''), ('code', '# ')]:
        correct = 0
        for prompt, answer in facts:
            full = prefix + prompt
            inp = tok(full, return_tensors='pt').to(DEVICE)
            fact_id = tok.encode(answer)
            if len(fact_id) > 0:
                fact_id = fact_id[-1] if model_type != 'gpt2' else fact_id[0]
            else:
                fact_id = 0
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :]
            if logits.argmax().item() == fact_id:
                correct += 1
        results[mode_name] = correct / len(facts)
    return results


MODELS = [
    # Already tested in P95, re-run for completeness with GPU
    ('gpt2',        'GPT2-Small',  124e6,  12, 'gpt2'),
    ('gpt2-medium', 'GPT2-Medium', 345e6,  24, 'gpt2'),
    ('gpt2-large',  'GPT2-Large',  774e6,  36, 'gpt2'),
    ('gpt2-xl',     'GPT2-XL',    1558e6,  48, 'gpt2'),
    # New architectures!
    ('Qwen/Qwen2.5-0.5B', 'Qwen2.5-0.5B', 494e6, 24, 'qwen'),
    ('Qwen/Qwen2.5-1.5B', 'Qwen2.5-1.5B', 1544e6, 28, 'qwen'),
]

def main():
    print(f"[P96] The Universal 0.94 Constant (device={DEVICE})")

    all_results = {}

    for model_id, label, n_params, n_layers, model_type in MODELS:
        print(f"\n  === {label} ({model_id}, {n_layers} layers) ===")
        try:
            if model_type == 'gpt2':
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                model = GPT2LMHeadModel.from_pretrained(model_id, local_files_only=True).eval().to(DEVICE)
                tok = GPT2Tokenizer.from_pretrained(model_id, local_files_only=True)
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.float16).eval().to(DEVICE)
                tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        # Logit Lens scan
        layer_data = logit_lens_scan(model, tok, FACTS, n_layers, model_type)

        # Find best layer
        best_layer = min(layer_data, key=lambda l: layer_data[l]['median_rank'])
        best_relative = best_layer / n_layers

        # Baseline & Code Mode
        mode_accs = eval_baseline(model, tok, FACTS, model_type)

        # GSF magnitude
        output_layer = n_layers - 1
        gsf_mag = layer_data[output_layer]['median_rank'] - layer_data[best_layer]['median_rank']

        all_results[label] = {
            'model': model_id,
            'params': n_params,
            'n_layers': n_layers,
            'architecture': model_type,
            'best_layer': best_layer,
            'best_layer_relative': best_relative,
            'best_accuracy': layer_data[best_layer]['accuracy'],
            'output_accuracy': layer_data[output_layer]['accuracy'],
            'natural_accuracy': mode_accs['natural'],
            'code_mode_accuracy': mode_accs['code'],
            'gsf_magnitude': gsf_mag,
            'layer_data': {str(k): v for k, v in layer_data.items()},
        }
        print(f"    Best layer: L{best_layer} (relative: {best_relative:.3f})")
        print(f"    Best acc: {layer_data[best_layer]['accuracy']:.0%}, Output: {layer_data[output_layer]['accuracy']:.0%}")
        print(f"    Natural: {mode_accs['natural']:.0%}, Code Mode: {mode_accs['code']:.0%}")
        print(f"    GSF magnitude: {gsf_mag:.1f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {'phase': 96, 'name': 'Universal 0.94 Constant', 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase96_universal_094.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Normalized layer trajectories
    for label, data in all_results.items():
        n_layers = data['n_layers']
        layers = list(range(n_layers))
        medians = [data['layer_data'][str(l)]['median_rank'] for l in layers]
        norm_layers = [l / n_layers for l in layers]
        marker = 'o' if 'GPT2' in label else 's'
        axes[0].plot(norm_layers, medians, f'{marker}-', label=label, markersize=3, linewidth=1.5)
    axes[0].axvline(x=0.94, color='red', linestyle='--', alpha=0.7, label='0.94 constant')
    axes[0].set_xlabel('Normalized Layer Position')
    axes[0].set_ylabel('Median Fact Rank')
    axes[0].set_title('GSF Trajectory: GPT-2 vs Qwen')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # 2. Best layer relative position vs params
    params_list = [d['params'] for d in all_results.values()]
    best_rels = [d['best_layer_relative'] for d in all_results.values()]
    archs = [d['architecture'] for d in all_results.values()]
    colors = ['#3498db' if a == 'gpt2' else '#e74c3c' for a in archs]
    axes[1].scatter(params_list, best_rels, c=colors, s=150, edgecolors='black', zorder=5)
    axes[1].axhline(y=0.94, color='red', linestyle='--', alpha=0.7, label='0.94')
    for i, lbl in enumerate(all_results.keys()):
        axes[1].annotate(lbl, (params_list[i], best_rels[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Parameters')
    axes[1].set_ylabel('Best Layer / Total Layers')
    axes[1].set_title('The 0.94 Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Accuracy comparison
    labels = list(all_results.keys())
    x = np.arange(len(labels))
    w = 0.25
    nat = [d['natural_accuracy'] for d in all_results.values()]
    code = [d['code_mode_accuracy'] for d in all_results.values()]
    best = [d['best_accuracy'] for d in all_results.values()]
    axes[2].bar(x - w, nat, w, label='Natural', color='#e74c3c', alpha=0.8)
    axes[2].bar(x, code, w, label='Code #', color='#2ecc71', alpha=0.8)
    axes[2].bar(x + w, best, w, label='Best Layer', color='#3498db', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha='right', fontsize=7)
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Cross-Architecture Comparison')
    axes[2].legend(fontsize=8)

    fig.suptitle('Phase 96: The Universal 0.94 Constant', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase96_universal_094.png'), dpi=150)
    plt.close()
    print("[Phase 96] Complete.")

if __name__ == '__main__':
    main()
