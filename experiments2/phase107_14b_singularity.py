# -*- coding: utf-8 -*-
"""
Phase 107: The 14B Singularity
Qwen2.5-14B with device_map="auto" (GPU/CPU hybrid, fp16, NO quantization).
Tests: does 14B reach 90%+ accuracy as predicted by P99?
Uses 10 facts only for speed.
"""
import torch, json, os, gc, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

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

TEMPLATES = {
    'natural': "{prompt}",
    'code_hash': "# {prompt}",
    'answer_is': "The answer is: {prompt}",
    'bullet': "- {prompt}",
}

def main():
    print("[P107] The 14B Singularity (GPU/CPU hybrid)")

    print("  Loading Qwen2.5-14B (fp16, device_map=auto)...")
    print("  This will split model across GPU + CPU RAM...")

    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-14B',
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B', local_files_only=True)
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded: {n_layers} layers")

    # Show device map summary
    if hasattr(model, 'hf_device_map'):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        print(f"  Devices: {devices_used}")

    template_results = {}
    for tmpl_name, tmpl in TEMPLATES.items():
        correct = 0
        total_rank = 0
        for idx, (prompt, answer) in enumerate(FACTS):
            full = tmpl.format(prompt=prompt)
            inp = tok(full, return_tensors='pt').to(model.device)
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if fact_tokens else 0

            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()
            rank = (logits.argsort(descending=True) == fact_id).nonzero()
            rank = rank.item() + 1 if rank.numel() > 0 else logits.shape[0]
            if rank == 1:
                correct += 1
            total_rank += rank
            if idx == 0:
                print(f"    {tmpl_name}: first inference OK (rank={rank})")

        acc = correct / len(FACTS)
        mean_rank = total_rank / len(FACTS)
        template_results[tmpl_name] = {'accuracy': acc, 'mean_rank': mean_rank}
        print(f"    {tmpl_name:15s}: acc={acc:.0%} ({correct}/{len(FACTS)}), rank={mean_rank:.1f}")

    # Scaling law validation
    n_params = 14e9
    predicted_code = 0.1616 * np.log(n_params) + (-2.7887)
    predicted_nat = 0.1562 * np.log(n_params) + (-2.7196)
    actual_code = template_results['code_hash']['accuracy']
    actual_nat = template_results['natural']['accuracy']
    best_acc = max(v['accuracy'] for v in template_results.values())
    best_tmpl = max(template_results, key=lambda t: template_results[t]['accuracy'])

    print(f"\n  === 14B RESULTS ===")
    print(f"  P99 Prediction (Code): {predicted_code:.1%}")
    print(f"  Actual Code Mode:      {actual_code:.0%}")
    print(f"  P99 Prediction (Nat):  {predicted_nat:.1%}")
    print(f"  Actual Natural:        {actual_nat:.0%}")
    print(f"  Best template:         {best_tmpl} ({best_acc:.0%})")
    print(f"  *** 90% TARGET: {'ACHIEVED!' if best_acc >= 0.9 else f'NOT YET ({best_acc:.0%})'} ***")

    results = {
        'phase': 107, 'name': 'The 14B Singularity',
        'model': 'Qwen/Qwen2.5-14B',
        'n_params': n_params,
        'n_layers': n_layers,
        'method': 'fp16 device_map=auto (GPU/CPU hybrid)',
        'template_results': template_results,
        'scaling_validation': {
            'predicted_code': predicted_code,
            'actual_code': actual_code,
            'predicted_nat': predicted_nat,
            'actual_natural': actual_nat,
            'best_accuracy': best_acc,
            'best_template': best_tmpl,
        },
    }
    with open(os.path.join(RESULTS_DIR, 'phase107_14b_singularity.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Phase 107] Complete.")

if __name__ == '__main__':
    main()
