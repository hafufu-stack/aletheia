# -*- coding: utf-8 -*-
"""
Phase 101: The Base vs Instruct Divide
Test whether P100's scaling law gap is caused by Instruct tuning.
Compare Qwen2.5-0.5B Base vs 1.5B Base (both cached).
Also: "The answer is:" prefix investigation.
GPU-accelerated.
"""
import torch, json, os, sys, gc, numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Same 20 facts as P95/P96 for fair comparison
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
    'cot_think': "Let's think step by step. {prompt}",
    'cot_answer': "The answer is: {prompt}",
    'bullet': "- {prompt}",
}

MODELS = [
    # Base models (fair comparison with P95 GPT-2 data)
    ('Qwen/Qwen2.5-0.5B', 'Qwen2.5-0.5B-Base', 'base'),
    ('Qwen/Qwen2.5-1.5B', 'Qwen2.5-1.5B-Base', 'base'),
    # Instruct models (to measure the Instruct penalty)
    ('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen2.5-0.5B-Instruct', 'instruct'),
    ('Qwen/Qwen2.5-7B-Instruct', 'Qwen2.5-7B-Instruct', 'instruct'),
]

def main():
    print(f"[P101] Base vs Instruct Divide (device={DEVICE})")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_results = {}

    for model_id, label, variant in MODELS:
        print(f"\n  === {label} ({variant}) ===")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, torch_dtype=torch.float16
            ).eval().to(DEVICE)
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        n_params = sum(p.numel() for p in model.parameters())
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
            print(f"    {tmpl_name:15s}: acc={acc:.0%}, rank={mean_rank:.1f}")

        all_results[label] = {
            'model': model_id,
            'variant': variant,
            'params': n_params,
            'template_results': template_results,
        }

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {'phase': 101, 'name': 'Base vs Instruct Divide', 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase101_base_instruct.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Base vs Instruct accuracy for each template
    model_labels = list(all_results.keys())
    x = np.arange(len(model_labels))
    templates_to_show = ['natural', 'code_hash', 'cot_answer']
    colors = {'natural': '#95a5a6', 'code_hash': '#2ecc71', 'cot_answer': '#e67e22'}
    w = 0.25
    for i, tmpl in enumerate(templates_to_show):
        accs = [all_results[m]['template_results'].get(tmpl, {}).get('accuracy', 0)
                for m in model_labels]
        axes[0].bar(x + i*w - w, accs, w, label=tmpl, color=colors.get(tmpl, 'gray'))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_labels, rotation=15, ha='right', fontsize=8)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Base vs Instruct: Factual Accuracy')
    axes[0].legend()

    # 2. Mean rank comparison
    for i, tmpl in enumerate(templates_to_show):
        ranks = [all_results[m]['template_results'].get(tmpl, {}).get('mean_rank', 0)
                for m in model_labels]
        axes[1].bar(x + i*w - w, ranks, w, label=tmpl, color=colors.get(tmpl, 'gray'))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_labels, rotation=15, ha='right', fontsize=8)
    axes[1].set_ylabel('Mean Rank (lower=better)')
    axes[1].set_title('Base vs Instruct: Mean Rank')
    axes[1].legend()

    fig.suptitle('Phase 101: The Base vs Instruct Divide', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase101_base_instruct.png'), dpi=150)
    plt.close()
    print("[Phase 101] Complete.")

if __name__ == '__main__':
    main()
