# -*- coding: utf-8 -*-
"""
Phase 98: The CoT Capability Horizon
Test whether CoT hurts factual accuracy across ALL model scales.
Deep Think's Self-Pollution Theory: CoT adds noise to fact retrieval.
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

TEMPLATES = {
    'natural': "{prompt}",
    'code_hash': "# {prompt}",
    'cot_think': "Let's think step by step. {prompt}",
    'cot_answer': "The answer is: {prompt}",
}

MODELS = [
    ('gpt2',        'GPT2-Small',  12, 'gpt2'),
    ('gpt2-xl',     'GPT2-XL',     48, 'gpt2'),
    ('Qwen/Qwen2.5-0.5B', 'Qwen2.5-0.5B', 24, 'qwen'),
    ('Qwen/Qwen2.5-1.5B', 'Qwen2.5-1.5B', 28, 'qwen'),
]

def main():
    print(f"[P98] CoT Capability Horizon (device={DEVICE})")

    all_results = {}

    for model_id, label, n_layers, model_type in MODELS:
        print(f"\n  === {label} ===")
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

        model_results = {}
        for tmpl_name, tmpl in TEMPLATES.items():
            correct = 0
            total_rank = 0
            for prompt, answer in FACTS:
                full = tmpl.format(prompt=prompt)
                inp = tok(full, return_tensors='pt').to(DEVICE)
                fact_tokens = tok.encode(answer)
                if model_type != 'gpt2' and len(fact_tokens) > 0:
                    fact_id = fact_tokens[-1]
                else:
                    fact_id = fact_tokens[0] if fact_tokens else 0

                with torch.no_grad():
                    logits = model(**inp).logits[0, -1, :]
                rank = (logits.argsort(descending=True) == fact_id).nonzero()
                rank = rank.item() + 1 if rank.numel() > 0 else logits.shape[0]
                if rank == 1:
                    correct += 1
                total_rank += rank

            acc = correct / len(FACTS)
            mean_rank = total_rank / len(FACTS)
            model_results[tmpl_name] = {'accuracy': acc, 'mean_rank': mean_rank}
            print(f"    {tmpl_name:15s}: acc={acc:.0%}, rank={mean_rank:.1f}")

        all_results[label] = model_results
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {'phase': 98, 'name': 'CoT Capability Horizon', 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase98_cot_horizon.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    model_names = list(all_results.keys())
    templates = list(TEMPLATES.keys())
    x = np.arange(len(model_names))
    w = 0.2
    colors = {'natural': '#95a5a6', 'code_hash': '#2ecc71', 'cot_think': '#e74c3c', 'cot_answer': '#e67e22'}

    for i, tmpl in enumerate(templates):
        accs = [all_results[m].get(tmpl, {}).get('accuracy', 0) for m in model_names]
        axes[0].bar(x + i*w - w*1.5, accs, w, label=tmpl, color=colors.get(tmpl, 'gray'))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Factual Accuracy: Code Mode vs CoT')
    axes[0].legend()

    for i, tmpl in enumerate(templates):
        ranks = [all_results[m].get(tmpl, {}).get('mean_rank', 0) for m in model_names]
        axes[1].bar(x + i*w - w*1.5, ranks, w, label=tmpl, color=colors.get(tmpl, 'gray'))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names)
    axes[1].set_ylabel('Mean Rank')
    axes[1].set_title('Mean Rank: Code Mode vs CoT')
    axes[1].legend()

    fig.suptitle('Phase 98: The CoT Capability Horizon', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase98_cot_horizon.png'), dpi=150)
    plt.close()
    print("[Phase 98] Complete.")

if __name__ == '__main__':
    main()
