# -*- coding: utf-8 -*-
"""
Phase 106: Frankenstein Alignment Surgery
Take Qwen2.5-0.5B-Instruct and replace the most affected suppressor layers'
attention weights with Base model weights. Test if factual accuracy recovers
while dialog capability is preserved.
GPU-accelerated.
"""
import torch, json, os, gc, copy, numpy as np
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
    """Evaluate factual accuracy with natural + code + answer templates."""
    results = {}
    for tmpl_name, tmpl in [('natural', '{p}'), ('code', '# {p}'), ('answer', 'The answer is: {p}')]:
        correct = 0
        for prompt, answer in facts:
            full = tmpl.format(p=prompt)
            inp = tok(full, return_tensors='pt').to(DEVICE)
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if fact_tokens else 0
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :]
            if logits.argmax().item() == fact_id:
                correct += 1
        results[tmpl_name] = correct / len(facts)
    return results

def main():
    print(f"[P106] Frankenstein Alignment Surgery (device={DEVICE})")

    # Load P105 results to find most affected layers
    p105_path = os.path.join(RESULTS_DIR, 'phase105_alignment_autopsy.json')
    if os.path.exists(p105_path):
        p105 = json.load(open(p105_path))
        affected_layers = p105['most_affected_layers'][:5]
        print(f"  P105 most affected layers: {affected_layers}")
    else:
        print("  WARNING: P105 not found, using default layers [22, 23, 20, 21, 19]")
        affected_layers = [22, 23, 20, 21, 19]

    # Load Base model
    print("  Loading Base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', local_files_only=True, torch_dtype=torch.float16
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', local_files_only=True)

    # Load Instruct model
    print("  Loading Instruct model...")
    inst_model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True, torch_dtype=torch.float16
    ).eval().to(DEVICE)

    # Evaluate originals
    print("\n  --- Original Models ---")
    base_acc = eval_accuracy(base_model, tok, FACTS)
    inst_acc = eval_accuracy(inst_model, tok, FACTS)
    print(f"  Base:     nat={base_acc['natural']:.0%}, code={base_acc['code']:.0%}, answer={base_acc['answer']:.0%}")
    print(f"  Instruct: nat={inst_acc['natural']:.0%}, code={inst_acc['code']:.0%}, answer={inst_acc['answer']:.0%}")

    # Surgery: try different numbers of transplanted layers
    surgery_results = {}

    for n_transplant in [1, 2, 3, 5]:
        layers_to_fix = affected_layers[:n_transplant]
        print(f"\n  --- Surgery: transplant {n_transplant} layers {layers_to_fix} ---")

        # Create Frankenstein model (deep copy of Instruct)
        frank_model = copy.deepcopy(inst_model)

        # Replace attention weights in affected layers
        for layer_idx in layers_to_fix:
            try:
                # Replace self_attn weights
                base_layer = base_model.model.layers[layer_idx].self_attn
                frank_layer = frank_model.model.layers[layer_idx].self_attn
                frank_layer.q_proj.weight.data.copy_(base_layer.q_proj.weight.data)
                frank_layer.k_proj.weight.data.copy_(base_layer.k_proj.weight.data)
                frank_layer.v_proj.weight.data.copy_(base_layer.v_proj.weight.data)
                frank_layer.o_proj.weight.data.copy_(base_layer.o_proj.weight.data)
                # Copy biases if they exist
                if hasattr(base_layer.q_proj, 'bias') and base_layer.q_proj.bias is not None:
                    frank_layer.q_proj.bias.data.copy_(base_layer.q_proj.bias.data)
                    frank_layer.k_proj.bias.data.copy_(base_layer.k_proj.bias.data)
                    frank_layer.v_proj.bias.data.copy_(base_layer.v_proj.bias.data)
            except Exception as e:
                print(f"    Layer {layer_idx} transplant failed: {e}")

        frank_acc = eval_accuracy(frank_model, tok, FACTS)
        print(f"    Frankenstein: nat={frank_acc['natural']:.0%}, code={frank_acc['code']:.0%}, answer={frank_acc['answer']:.0%}")

        # Recovery percentage
        base_nat = base_acc['natural']
        inst_nat = inst_acc['natural']
        frank_nat = frank_acc['natural']
        if base_nat > inst_nat:
            recovery = (frank_nat - inst_nat) / (base_nat - inst_nat) * 100
        else:
            recovery = 0
        print(f"    Recovery: {recovery:.0f}% of Alignment Tax removed")

        surgery_results[f"transplant_{n_transplant}"] = {
            'layers': layers_to_fix,
            'accuracy': frank_acc,
            'recovery_pct': float(recovery),
        }

        del frank_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {
        'phase': 106, 'name': 'Frankenstein Alignment Surgery',
        'base_accuracy': base_acc,
        'instruct_accuracy': inst_acc,
        'surgery_results': surgery_results,
        'affected_layers': affected_layers,
    }
    with open(os.path.join(RESULTS_DIR, 'phase106_frankenstein.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Accuracy comparison
    labels = ['Base', 'Instruct'] + [f'Frank-{k.split("_")[1]}L' for k in surgery_results.keys()]
    nat_vals = [base_acc['natural'], inst_acc['natural']] + \
               [v['accuracy']['natural'] for v in surgery_results.values()]
    code_vals = [base_acc['code'], inst_acc['code']] + \
                [v['accuracy']['code'] for v in surgery_results.values()]
    ans_vals = [base_acc['answer'], inst_acc['answer']] + \
               [v['accuracy']['answer'] for v in surgery_results.values()]
    x = np.arange(len(labels))
    w = 0.25
    axes[0].bar(x - w, nat_vals, w, label='Natural', color='#95a5a6')
    axes[0].bar(x, code_vals, w, label='Code #', color='#2ecc71')
    axes[0].bar(x + w, ans_vals, w, label='Answer is', color='#e67e22')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Frankenstein Surgery: Accuracy Recovery')
    axes[0].legend()

    # 2. Recovery percentage
    n_transplants = [int(k.split('_')[1]) for k in surgery_results.keys()]
    recoveries = [v['recovery_pct'] for v in surgery_results.values()]
    axes[1].plot(n_transplants, recoveries, 'o-', color='#2ecc71', linewidth=2, markersize=10)
    axes[1].set_xlabel('Layers Transplanted')
    axes[1].set_ylabel('Alignment Tax Recovery (%)')
    axes[1].set_title('Surgery Dose-Response')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=100, color='gold', linestyle=':', label='Full Recovery')
    axes[1].legend()

    fig.suptitle('Phase 106: Frankenstein Alignment Surgery',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase106_frankenstein.png'), dpi=150)
    plt.close()

    del base_model, inst_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Phase 106] Complete.")

if __name__ == '__main__':
    main()
