# -*- coding: utf-8 -*-
"""
Phase 92: The Perplexity Cost of Truth
When we lobotomize suppressors, how much does fluency/perplexity degrade?
Is there a sweet spot where facts improve but fluency is preserved?
"""
import torch, json, os, sys, math, numpy as np
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

FLUENCY_TEXTS = [
    "The weather today is beautiful and the sun is shining brightly.",
    "Once upon a time there was a young prince who lived in a castle.",
    "The president announced that the new economic plan would be implemented.",
    "Scientists have discovered a new species of deep-sea creatures.",
    "The stock market experienced significant volatility during the trading session.",
    "Researchers at the university published their findings in a peer-reviewed journal.",
    "The ancient Romans built roads that connected their vast empire.",
    "Modern technology has transformed the way people communicate with each other.",
]

def compute_perplexity(model, tok, text):
    """Compute perplexity of a text."""
    inp = tok(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inp, labels=inp['input_ids'])
    return math.exp(outputs.loss.item())

def evaluate_facts(model, tok, facts):
    correct = 0
    total_rank = 0
    for prompt, answer in facts:
        inp = tok(prompt, return_tensors='pt')
        fact_id = tok.encode(answer)[0]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
        rank = (logits.argsort(descending=True) == fact_id).nonzero().item() + 1
        if rank == 1:
            correct += 1
        total_rank += rank
    return correct / len(facts), total_rank / len(facts)

def main():
    print("[P92] The Perplexity Cost of Truth")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    hd = 64

    # Sweep: mute suppressors with different scales (0=full mute, 0.5=half, 1=normal)
    scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    suppressors = [(9, 6), (10, 7), (11, 7)]

    results = []
    for scale in scales:
        model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
        with torch.no_grad():
            for layer, head in suppressors:
                start = head * hd
                end = start + hd
                model.transformer.h[layer].attn.c_proj.weight[start:end, :] *= scale
                if model.transformer.h[layer].attn.c_proj.bias is not None:
                    model.transformer.h[layer].attn.c_proj.bias[start:end] *= scale

        acc, mean_rank = evaluate_facts(model, tok, FACTS)
        ppls = [compute_perplexity(model, tok, t) for t in FLUENCY_TEXTS]
        mean_ppl = float(np.mean(ppls))

        results.append({
            'scale': scale,
            'accuracy': acc,
            'mean_rank': mean_rank,
            'perplexity': mean_ppl,
        })
        print(f"  scale={scale:.1f}: acc={acc:.0%}, rank={mean_rank:.1f}, ppl={mean_ppl:.1f}")
        del model

    out = {'phase': 92, 'name': 'Perplexity Cost of Truth', 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase92_perplexity_cost.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ss = [r['scale'] for r in results]
    accs = [r['accuracy'] for r in results]
    ranks = [r['mean_rank'] for r in results]
    ppls = [r['perplexity'] for r in results]

    axes[0].plot(ss, accs, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    axes[0].set_xlabel('Suppressor Scale (1.0=normal, 0.0=full mute)')
    axes[0].set_ylabel('Factual Accuracy')
    axes[0].set_title('Accuracy vs Scale')
    axes[0].invert_xaxis()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ss, ppls, 's-', color='#e74c3c', linewidth=2, markersize=8)
    axes[1].set_xlabel('Suppressor Scale')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Fluency Cost (Perplexity)')
    axes[1].invert_xaxis()
    axes[1].grid(True, alpha=0.3)

    # Pareto frontier: accuracy vs perplexity
    axes[2].scatter(ppls, accs, c=ss, cmap='RdYlGn', s=100, edgecolors='black')
    for i, s in enumerate(ss):
        axes[2].annotate(f'{s:.1f}', (ppls[i], accs[i]), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
    axes[2].set_xlabel('Perplexity (lower=more fluent)')
    axes[2].set_ylabel('Factual Accuracy')
    axes[2].set_title('Pareto Frontier: Truth vs Fluency')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Phase 92: The Perplexity Cost of Truth', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase92_perplexity_cost.png'), dpi=150)
    plt.close()
    print("[Phase 92] Complete.")

if __name__ == '__main__':
    main()
