# -*- coding: utf-8 -*-
"""
Phase 93: Code Mode x Multi-Token Generation
Test whether Code Mode Switch maintains factual accuracy across multiple
generated tokens, not just the first token.
Addresses paper Limitation: "Multi-token facts"
"""
import torch, json, os, sys, numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

# Multi-token facts: expected substring in generation
MULTI_FACTS = [
    ("The capital of Japan is", "Tokyo"),
    ("The capital of France is", "Paris"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Italy is", "Rome"),
    ("The capital of Spain is", "Madrid"),
    ("The largest planet in the solar system is", "Jupiter"),
    ("Albert Einstein was born in", "Germany"),
    ("The author of Romeo and Juliet is", "Shakespeare"),
    ("The first president of the United States was", "Washington"),
    ("The Great Wall of China is located in", "China"),
    ("The chemical formula for water is", "H2O"),
    ("The speed of light is approximately", "300"),
    ("The theory of relativity was proposed by", "Einstein"),
    ("Shakespeare was born in", "Stratford"),
    ("The largest ocean on Earth is the", "Pacific"),
]

TEMPLATES = {
    'natural': "{prompt}",
    'comment': "# {prompt}",
    'slash': "// {prompt}",
    'bullet': "- {prompt}",
    'cot': "Let's think step by step. {prompt}",
}

def generate_text(model, tok, prompt, max_new=20):
    """Generate text and return it."""
    inp = tok(prompt, return_tensors='pt')
    with torch.no_grad():
        out = model.generate(
            inp['input_ids'], max_new_tokens=max_new,
            do_sample=False, pad_token_id=tok.eos_token_id
        )
    generated = tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated

def main():
    print("[P93] Code Mode x Multi-Token Generation")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

    results = []
    for tmpl_name, tmpl in TEMPLATES.items():
        correct = 0
        partial = 0
        generations = []

        for prompt, expected in MULTI_FACTS:
            full_prompt = tmpl.format(prompt=prompt)
            generated = generate_text(model, tok, full_prompt)

            # Check if expected substring appears in generation
            found = expected.lower() in generated.lower()
            if found:
                correct += 1

            # Partial: first token matches
            first_tok = generated.strip().split()[0] if generated.strip() else ""
            partial_match = expected.lower().startswith(first_tok.lower()) if first_tok else False

            generations.append({
                'prompt': prompt,
                'expected': expected,
                'generated': generated[:60].encode('ascii', 'replace').decode('ascii'),
                'found': found,
            })

        acc = correct / len(MULTI_FACTS)
        results.append({
            'template': tmpl_name,
            'multi_token_accuracy': acc,
            'correct': correct,
            'total': len(MULTI_FACTS),
            'samples': generations[:5],
        })
        print(f"  {tmpl_name:10s}: multi_acc={acc:.0%} ({correct}/{len(MULTI_FACTS)})")
        # Show a few samples
        for g in generations[:3]:
            gen_safe = g['generated'][:50]
            mark = "OK" if g['found'] else "MISS"
            print(f"    [{mark}] {g['prompt'][:30]}... -> {gen_safe}")

    out = {'phase': 93, 'name': 'Code Mode Multi-Token', 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase93_multi_token_code.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    names = [r['template'] for r in results]
    accs = [r['multi_token_accuracy'] for r in results]
    colors = ['gray', '#2ecc71', '#27ae60', '#e67e22', '#3498db']

    bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Multi-Token Factual Accuracy')
    ax.set_title('Phase 93: Does Code Mode Maintain Facts Across Multiple Tokens?')
    ax.set_ylim(0, max(accs) * 1.3 + 0.05)
    for bar, acc_val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc_val:.0%}', ha='center', va='bottom', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase93_multi_token_code.png'), dpi=150)
    plt.close()
    print("[Phase 93] Complete.")

if __name__ == '__main__':
    main()
