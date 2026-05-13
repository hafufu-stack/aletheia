# -*- coding: utf-8 -*-
"""
Phase 102: The 14B Prophecy (GPU 4-bit)
Qwen2.5-14B with 4-bit quantization on GPU (~8GB VRAM).
Tests whether 14B exceeds the predicted 90% accuracy threshold.
"""
import torch, json, os, sys, gc, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

DEVICE = 'cuda'

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

TEMPLATES = {
    'natural': "{prompt}",
    'code_hash': "# {prompt}",
    'code_slash': "// {prompt}",
    'cot_answer': "The answer is: {prompt}",
    'bullet': "- {prompt}",
}

def main():
    print(f"[P102] The 14B Prophecy - GPU 4-bit")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    print("  Loading Qwen2.5-14B (4-bit quantized)...")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-14B',
        local_files_only=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B', local_files_only=True)
    n_params = 14e9  # approximate (quantized model reports different)
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded: ~14B params, {n_layers} layers, 4-bit quantized on GPU")

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

    # Scaling law validation
    predicted_code = 0.1616 * np.log(n_params) + (-2.7887)
    predicted_nat = 0.1562 * np.log(n_params) + (-2.7196)
    actual_code = template_results['code_hash']['accuracy']
    actual_nat = template_results['natural']['accuracy']
    best_acc = max(v['accuracy'] for v in template_results.values())
    best_tmpl = max(template_results, key=lambda t: template_results[t]['accuracy'])

    print(f"\n  === 14B SCALING LAW VALIDATION ===")
    print(f"  P99 Prediction (Code): {predicted_code:.1%}")
    print(f"  Actual Code Mode:      {actual_code:.0%}")
    print(f"  P99 Prediction (Nat):  {predicted_nat:.1%}")
    print(f"  Actual Natural:        {actual_nat:.0%}")
    print(f"  Best Template:         {best_tmpl} ({best_acc:.0%})")
    print(f"  *** 90% TARGET: {'ACHIEVED!' if best_acc >= 0.9 else 'NOT YET (' + f'{best_acc:.0%}' + ')'} ***")

    out = {
        'phase': 102, 'name': 'The 14B Prophecy',
        'model': 'Qwen/Qwen2.5-14B',
        'n_params': n_params,
        'n_layers': n_layers,
        'quantization': '4-bit NF4',
        'template_results': template_results,
        'scaling_validation': {
            'predicted_code': predicted_code,
            'actual_code': actual_code,
            'predicted_natural': predicted_nat,
            'actual_natural': actual_nat,
            'best_accuracy': best_acc,
            'best_template': best_tmpl,
        },
    }
    with open(os.path.join(RESULTS_DIR, 'phase102_14b_prophecy.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Template comparison
    tmpl_names = list(template_results.keys())
    tmpl_accs = [template_results[t]['accuracy'] for t in tmpl_names]
    colors = ['#95a5a6', '#2ecc71', '#27ae60', '#e67e22', '#3498db']
    axes[0].bar(tmpl_names, tmpl_accs, color=colors[:len(tmpl_names)])
    axes[0].axhline(y=0.9, color='gold', linestyle=':', linewidth=2, label='90% Target')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Qwen2.5-14B: Template Comparison')
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    # 2. Scaling law with 14B
    p99_path = os.path.join(RESULTS_DIR, 'phase99_scaling_prediction.json')
    if os.path.exists(p99_path):
        p99 = json.load(open(p99_path))
        ps = [d['params'] for d in p99['data_points']]
        cs = [d['code_acc'] for d in p99['data_points']]
        ns = [d['natural_acc'] for d in p99['data_points']]
        axes[1].scatter(ps, cs, c='#2ecc71', marker='o', s=60, label='Code (prev)')
        axes[1].scatter(ps, ns, c='#e74c3c', marker='s', s=60, label='Nat (prev)')
    axes[1].scatter([n_params], [actual_code], c='#2ecc71', marker='*', s=400,
                   edgecolors='black', linewidth=2, zorder=5, label=f'14B Code: {actual_code:.0%}')
    axes[1].scatter([n_params], [actual_nat], c='#e74c3c', marker='*', s=400,
                   edgecolors='black', linewidth=2, zorder=5, label=f'14B Natural: {actual_nat:.0%}')
    param_range = np.logspace(np.log10(100e6), np.log10(20e9), 200)
    axes[1].plot(param_range, np.clip(0.1616*np.log(param_range)-2.7887, 0, 1), '--', color='#2ecc71', alpha=0.5)
    axes[1].plot(param_range, np.clip(0.1562*np.log(param_range)-2.7196, 0, 1), '--', color='#e74c3c', alpha=0.5)
    axes[1].axhline(y=0.9, color='gold', linestyle=':', linewidth=2)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Parameters')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Scaling Law + 14B Validation')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    fig.suptitle('Phase 102: The 14B Prophecy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase102_14b_prophecy.png'), dpi=150)
    plt.close()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[Phase 102] Complete.")

if __name__ == '__main__':
    main()
