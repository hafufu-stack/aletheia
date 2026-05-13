# -*- coding: utf-8 -*-
"""
Phase 109: The Reasoning Horizon
Does Code Mode help reasoning (not just fact recall)?
Test on simple math and logic tasks.
GPU, Qwen-1.5B.
"""
import torch, json, os, gc, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Mix of fact, math, and logic tasks
TASKS = {
    'fact': [
        ("The capital of Japan is", " Tokyo"),
        ("The capital of France is", " Paris"),
        ("The chemical symbol for gold is", " Au"),
        ("The largest planet is", " Jupiter"),
        ("The first president of the US was", " George"),
    ],
    'math': [
        ("2 + 3 =", " 5"),
        ("7 * 8 =", " 56"),
        ("100 - 37 =", " 63"),
        ("12 * 12 =", " 144"),
        ("15 + 27 =", " 42"),
    ],
    'logic': [
        ("If it rains, the ground gets wet. It is raining. Therefore the ground is", " wet"),
        ("All dogs are animals. Rex is a dog. Therefore Rex is", " an"),
        ("The opposite of hot is", " cold"),
        ("The opposite of up is", " down"),
        ("Water at 200 degrees Celsius is", " steam"),
    ],
}

TEMPLATES = {
    'natural': "{prompt}",
    'code_hash': "# {prompt}",
    'answer_is': "The answer is: {prompt}",
    'cot_think': "Let's think step by step. {prompt}",
}

def main():
    print(f"[P109] The Reasoning Horizon (device={DEVICE})")

    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B', local_files_only=True, torch_dtype=torch.float16
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', local_files_only=True)

    results = {}
    for task_type, tasks in TASKS.items():
        print(f"\n  === {task_type.upper()} ===")
        task_results = {}
        for tmpl_name, tmpl in TEMPLATES.items():
            correct = 0
            total_rank = 0
            for prompt, answer in tasks:
                full = tmpl.format(prompt=prompt)
                inp = tok(full, return_tensors='pt').to(DEVICE)
                fact_tokens = tok.encode(answer)
                fact_id = fact_tokens[-1] if fact_tokens else 0
                with torch.no_grad():
                    logits = model(**inp).logits[0, -1, :]
                rank = (logits.argsort(descending=True) == fact_id).nonzero()
                rank = rank.item() + 1 if rank.numel() > 0 else logits.shape[0]
                if rank == 1: correct += 1
                total_rank += rank
            acc = correct / len(tasks)
            mean_rank = total_rank / len(tasks)
            task_results[tmpl_name] = {'accuracy': acc, 'mean_rank': mean_rank}
            print(f"    {tmpl_name:15s}: acc={acc:.0%}, rank={mean_rank:.1f}")
        results[task_type] = task_results

    # Analysis: which template is best for each task type?
    print(f"\n  === BEST TEMPLATE PER TASK ===")
    for task_type in TASKS:
        best = max(results[task_type], key=lambda t: results[task_type][t]['accuracy'])
        acc = results[task_type][best]['accuracy']
        print(f"  {task_type}: {best} ({acc:.0%})")

    out = {
        'phase': 109, 'name': 'The Reasoning Horizon',
        'model': 'Qwen/Qwen2.5-1.5B', 'results': results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase109_reasoning_horizon.json'), 'w') as f:
        json.dump(out, f, indent=2)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    task_types = list(TASKS.keys())
    for i, task_type in enumerate(task_types):
        tmpl_names = list(TEMPLATES.keys())
        accs = [results[task_type][t]['accuracy'] for t in tmpl_names]
        colors = ['#95a5a6', '#2ecc71', '#e67e22', '#3498db']
        axes[i].bar(tmpl_names, accs, color=colors)
        axes[i].set_ylabel('Accuracy')
        axes[i].set_title(f'{task_type.upper()} Tasks')
        axes[i].set_ylim(0, 1.1)
        axes[i].tick_params(axis='x', rotation=15)

    fig.suptitle('Phase 109: The Reasoning Horizon - Qwen-1.5B', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase109_reasoning_horizon.png'), dpi=150)
    plt.close()

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("[Phase 109] Complete.")

if __name__ == '__main__':
    main()
