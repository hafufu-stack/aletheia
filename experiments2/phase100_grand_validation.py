# -*- coding: utf-8 -*-
"""
Phase 100: The Grand Validation
THE definitive test: Does a 7B model achieve ~90% factual accuracy
as predicted by the Scaling Law (P99)?
Qwen2.5-7B-Instruct (7B params, cached, fits in 17GB VRAM in fp16).
Also tests the 0.94 constant at 7B scale.
GPU-accelerated.
"""
import torch, json, os, sys, gc, numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Extended fact set for rigorous testing (30 facts)
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
    ("The longest river in the world is the", " N"),
    ("The chemical symbol for silver is", " Ag"),
    ("The freezing point of water in Fahrenheit is", " 32"),
    ("The number of continents on Earth is", " seven"),
    ("The element with atomic number 1 is", " hydrogen"),
    ("The planet closest to the Sun is", " Mercury"),
    ("The author of Hamlet is", " William"),
    ("The capital of Australia is", " Canberra"),
    ("The chemical symbol for iron is", " Fe"),
    ("The tallest animal in the world is the", " gir"),
]

TEMPLATES = {
    'natural': "{prompt}",
    'code_hash': "# {prompt}",
    'code_slash': "// {prompt}",
    'cot_think': "Let's think step by step. {prompt}",
    'cot_answer': "The answer is: {prompt}",
    'bullet': "- {prompt}",
}

def logit_lens_layer(model, tok, prompt, layer_idx):
    """Extract logits at a specific layer via Logit Lens."""
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
    print(f"[P100] The Grand Validation (device={DEVICE})")
    print("  Loading Qwen2.5-7B-Instruct...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-7B-Instruct',
        local_files_only=True,
        torch_dtype=torch.float16,
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2.5-7B-Instruct', local_files_only=True
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded: {n_params/1e9:.2f}B params, {n_layers} layers")

    # ===== 1. Template comparison =====
    print("\n  --- Template Comparison (30 facts) ---")
    template_results = {}
    for tmpl_name, tmpl in TEMPLATES.items():
        correct = 0
        total_rank = 0
        for prompt, answer in FACTS:
            full = tmpl.format(prompt=prompt)
            inp = tok(full, return_tensors='pt').to(DEVICE)
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if fact_tokens else 0

            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :]
            rank = (logits.argsort(descending=True) == fact_id).nonzero()
            rank = rank.item() + 1 if rank.numel() > 0 else logits.shape[0]
            if rank == 1:
                correct += 1
            total_rank += rank

        acc = correct / len(FACTS)
        mean_rank = total_rank / len(FACTS)
        template_results[tmpl_name] = {'accuracy': acc, 'mean_rank': mean_rank}
        print(f"    {tmpl_name:15s}: acc={acc:.0%} ({correct}/{len(FACTS)}), rank={mean_rank:.1f}")

    # ===== 2. Logit Lens scan =====
    print(f"\n  --- Logit Lens ({n_layers} layers) ---")
    layer_data = {}
    for layer_idx in range(n_layers):
        ranks = []
        for prompt, answer in FACTS:
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if fact_tokens else 0
            logits = logit_lens_layer(model, tok, prompt, layer_idx)
            rank = (logits.argsort(descending=True) == fact_id).nonzero()
            rank = rank.item() + 1 if rank.numel() > 0 else logits.shape[0]
            ranks.append(rank)
        median_rank = float(np.median(ranks))
        acc = sum(1 for r in ranks if r == 1) / len(FACTS)
        layer_data[layer_idx] = {'median_rank': median_rank, 'accuracy': acc}
        if layer_idx % 4 == 0 or layer_idx == n_layers - 1:
            print(f"    L{layer_idx}: rank={median_rank:.1f}, acc={acc:.0%}")

    best_layer = min(layer_data, key=lambda l: layer_data[l]['median_rank'])
    best_relative = best_layer / n_layers

    print(f"\n  BEST LAYER: L{best_layer} (relative: {best_relative:.3f})")
    print(f"  Best accuracy: {layer_data[best_layer]['accuracy']:.0%}")
    print(f"  Output accuracy: {layer_data[n_layers-1]['accuracy']:.0%}")

    # ===== 3. Scaling Law Validation =====
    # P99 prediction: Code Mode at 7B -> ~85-90%
    code_acc = template_results['code_hash']['accuracy']
    nat_acc = template_results['natural']['accuracy']

    # P99 formula: Code = 0.1616 * ln(N) + (-2.7887)
    predicted_code = 0.1616 * np.log(n_params) + (-2.7887)
    predicted_nat = 0.1562 * np.log(n_params) + (-2.7196)

    print(f"\n  === SCALING LAW VALIDATION ===")
    print(f"  P99 Prediction (Code): {predicted_code:.1%}")
    print(f"  Actual Code Mode:      {code_acc:.1%}")
    print(f"  Prediction Error:      {abs(code_acc - predicted_code):.1%}")
    print(f"  P99 Prediction (Nat):  {predicted_nat:.1%}")
    print(f"  Actual Natural:        {nat_acc:.1%}")
    print(f"  Prediction Error:      {abs(nat_acc - predicted_nat):.1%}")

    # Save
    out = {
        'phase': 100, 'name': 'The Grand Validation',
        'model': 'Qwen/Qwen2.5-7B-Instruct',
        'n_params': n_params,
        'n_layers': n_layers,
        'best_layer': best_layer,
        'best_layer_relative': best_relative,
        'template_results': template_results,
        'layer_data': {str(k): v for k, v in layer_data.items()},
        'scaling_validation': {
            'predicted_code': predicted_code,
            'actual_code': code_acc,
            'predicted_natural': predicted_nat,
            'actual_natural': nat_acc,
            'code_error': abs(code_acc - predicted_code),
            'natural_error': abs(nat_acc - predicted_nat),
        },
        'aletheia_094': best_relative,
    }
    with open(os.path.join(RESULTS_DIR, 'phase100_grand_validation.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Layer trajectory
    layers = list(range(n_layers))
    medians = [layer_data[l]['median_rank'] for l in layers]
    accs = [layer_data[l]['accuracy'] for l in layers]
    axes[0].plot([l/n_layers for l in layers], medians, 'b-o', markersize=3)
    axes[0].axvline(x=0.94, color='red', linestyle='--', alpha=0.7, label='0.94 constant')
    axes[0].axvline(x=best_relative, color='green', linestyle=':', alpha=0.7,
                   label=f'Best: {best_relative:.3f}')
    axes[0].set_xlabel('Normalized Layer')
    axes[0].set_ylabel('Median Rank')
    axes[0].set_title(f'Qwen2.5-7B: Logit Lens (Best=L{best_layer}={best_relative:.3f})')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Template comparison
    tmpl_names = list(template_results.keys())
    tmpl_accs = [template_results[t]['accuracy'] for t in tmpl_names]
    colors_bar = ['#95a5a6', '#2ecc71', '#27ae60', '#e74c3c', '#e67e22', '#3498db']
    axes[1].bar(tmpl_names, tmpl_accs, color=colors_bar[:len(tmpl_names)])
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('7B: Template Comparison')
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].axhline(y=0.9, color='gold', linestyle=':', linewidth=2, label='90%')
    axes[1].legend()

    # 3. Scaling law + 7B validation point
    # Load P99 data
    p99_path = os.path.join(RESULTS_DIR, 'phase99_scaling_prediction.json')
    if os.path.exists(p99_path):
        p99 = json.load(open(p99_path))
        ps = [d['params'] for d in p99['data_points']]
        cs = [d['code_acc'] for d in p99['data_points']]
        ns = [d['natural_acc'] for d in p99['data_points']]
        axes[2].scatter(ps, cs, c='#2ecc71', marker='o', s=80, label='Code (prev)', zorder=3)
        axes[2].scatter(ps, ns, c='#e74c3c', marker='s', s=80, label='Natural (prev)', zorder=3)
    # Add 7B point
    axes[2].scatter([n_params], [code_acc], c='#2ecc71', marker='*', s=300,
                   edgecolors='black', linewidth=2, zorder=5, label=f'7B Code: {code_acc:.0%}')
    axes[2].scatter([n_params], [nat_acc], c='#e74c3c', marker='*', s=300,
                   edgecolors='black', linewidth=2, zorder=5, label=f'7B Natural: {nat_acc:.0%}')
    # Fit line
    param_range = np.logspace(np.log10(100e6), np.log10(20e9), 200)
    pred_code = np.clip(0.1616 * np.log(param_range) - 2.7887, 0, 1)
    pred_nat = np.clip(0.1562 * np.log(param_range) - 2.7196, 0, 1)
    axes[2].plot(param_range, pred_code, '--', color='#2ecc71', alpha=0.5, label='P99 Code fit')
    axes[2].plot(param_range, pred_nat, '--', color='#e74c3c', alpha=0.5, label='P99 Nat fit')
    axes[2].axhline(y=0.9, color='gold', linestyle=':', linewidth=2)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Parameters')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Scaling Law Validation')
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1.05)

    fig.suptitle('Phase 100: The Grand Validation - Qwen2.5-7B',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase100_grand_validation.png'), dpi=150)
    plt.close()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n[Phase 100] COMPLETE! Project Aletheia: 100 Phases Achieved!")

if __name__ == '__main__':
    main()
