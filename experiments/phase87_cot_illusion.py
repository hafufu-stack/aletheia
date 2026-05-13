# -*- coding: utf-8 -*-
"""
Phase 87: The Chain-of-Thought Illusion
Test if "Let's think step by step" activates the same Code Mode Switch
neurons (L11:N314) and suppressor entropy patterns as symbol prefixes.
"""
import torch, json, os, sys, numpy as np
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

# Prompt templates to test
TEMPLATES = {
    'natural':       "{prompt}",
    'cot_think':     "Let's think step by step. {prompt}",
    'cot_answer':    "Think carefully and answer: {prompt}",
    'cot_reason':    "Let me reason about this. {prompt}",
    'bullet':        "- Question: {prompt}",
    'numbered':      "1. {prompt}",
    'code_comment':  "# {prompt}",
    'code_slash':    "// {prompt}",
    'markdown':      "## {prompt}",
    'arrow':         "=> {prompt}",
}

def main():
    print("[P87] Chain-of-Thought Illusion")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True, attn_implementation='eager').eval()
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

    # Target neurons and heads
    code_mode_neurons = {
        'L11_N314': (11, 314),
        'L10_N496': (10, 496),
    }
    suppressor_heads = [(9, 6), (10, 7), (11, 7)]

    results = []
    for tmpl_name, tmpl in TEMPLATES.items():
        neuron_activations = {k: [] for k in code_mode_neurons}
        head_entropies = {f'L{l}H{h}': [] for l, h in suppressor_heads}
        correct = 0
        total_rank = 0

        # Hooks for neuron activations
        mlp_activations = {}
        def make_mlp_hook(layer_idx):
            def hook_fn(module, input, output):
                mlp_activations[layer_idx] = output.detach()
            return hook_fn

        attn_weights = {}
        def make_attn_hook(layer_idx):
            def hook_fn(module, input, output):
                # Get attention weights - need to re-run with output_attentions
                pass
            return hook_fn

        for prompt, answer in FACTS:
            full_prompt = tmpl.format(prompt=prompt)
            inp = tok(full_prompt, return_tensors='pt')
            fact_id = tok.encode(answer)[0]

            # Run 1: get logits and attentions
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
                logits = out.logits[0, -1, :]
                attentions = out.attentions

            # Run 2: get MLP neuron activations (separate pass to avoid interference)
            handles = []
            for name, (layer, neuron) in code_mode_neurons.items():
                def make_mlp_hook(li):
                    def hook_fn(module, input, output):
                        mlp_activations[li] = output.detach()
                    return hook_fn
                h = model.transformer.h[layer].mlp.register_forward_hook(make_mlp_hook(layer))
                handles.append(h)
            with torch.no_grad():
                model(**inp)
            for h in handles:
                h.remove()

            # Rank
            rank = (logits.argsort(descending=True) == fact_id).nonzero().item() + 1
            if rank == 1:
                correct += 1
            total_rank += rank

            # Neuron activations
            for name, (layer, neuron) in code_mode_neurons.items():
                if layer in mlp_activations:
                    act = mlp_activations[layer][0, -1, neuron].item()
                    neuron_activations[name].append(act)

            # Head entropies
            for layer, head in suppressor_heads:
                attn = attentions[layer][0, head, -1, :]  # last token's attention
                entropy = -(attn * torch.log(attn + 1e-10)).sum().item()
                head_entropies[f'L{layer}H{head}'].append(entropy)

            mlp_activations.clear()

        acc = correct / len(FACTS)
        mean_rank = total_rank / len(FACTS)
        result = {
            'template': tmpl_name,
            'accuracy': acc,
            'mean_rank': mean_rank,
            'neuron_activations': {k: float(np.mean(v)) if v else 0 for k, v in neuron_activations.items()},
            'head_entropies': {k: float(np.mean(v)) if v else 0 for k, v in head_entropies.items()},
        }
        results.append(result)
        n314 = result['neuron_activations'].get('L11_N314', 0)
        h7_ent = result['head_entropies'].get('L11H7', 0)
        print(f"  {tmpl_name:15s}: acc={acc:.0%}, rank={mean_rank:.1f}, N314={n314:.2f}, L11H7_ent={h7_ent:.3f}")

    out = {'phase': 87, 'name': 'Chain-of-Thought Illusion', 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase87_cot_illusion.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    names = [r['template'] for r in results]
    accs = [r['accuracy'] for r in results]
    n314s = [r['neuron_activations'].get('L11_N314', 0) for r in results]
    h7_ents = [r['head_entropies'].get('L11H7', 0) for r in results]

    # Color code: natural=gray, cot=blue, code=green, other=orange
    colors = []
    for n in names:
        if n == 'natural':
            colors.append('gray')
        elif n.startswith('cot'):
            colors.append('#3498db')
        elif n.startswith('code'):
            colors.append('#2ecc71')
        else:
            colors.append('#e67e22')

    axes[0].barh(names, accs, color=colors)
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Factual Accuracy by Template')

    axes[1].barh(names, n314s, color=colors)
    axes[1].set_xlabel('Mean Activation')
    axes[1].set_title('Code Mode Neuron (L11:N314)')
    axes[1].axvline(x=0, color='black', linewidth=0.5)

    axes[2].barh(names, h7_ents, color=colors)
    axes[2].set_xlabel('Mean Entropy')
    axes[2].set_title('L11H7 Attention Entropy')

    fig.suptitle('Phase 87: Is Chain-of-Thought a Code Mode Switch?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase87_cot_illusion.png'), dpi=150)
    plt.close()
    print("[Phase 87] Complete.")

if __name__ == '__main__':
    main()
