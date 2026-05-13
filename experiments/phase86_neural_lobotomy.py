# -*- coding: utf-8 -*-
"""
Phase 86: The Ultimate Neural Lobotomy
Permanently modify model weights: mute suppressors, boost helpers.
Create a "Hallucination-Zero" GPT-2 without any runtime intervention.
"""
import torch, json, os, sys, copy, numpy as np
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
    ("The Pythagorean theorem states that a squared plus b squared equals", " c"),
    ("The human body has a total of", " 206"),
    ("DNA stands for de", "oxy"),
    ("The Great Wall of China is located in", " China"),
    ("Shakespeare was born in", " Strat"),
]

def evaluate(model, tok, facts):
    """Evaluate accuracy and mean rank."""
    correct = 0
    total_rank = 0
    ranks = []
    for prompt, answer in facts:
        inp = tok(prompt, return_tensors='pt')
        fact_id = tok.encode(answer)[0]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
        rank = (logits.argsort(descending=True) == fact_id).nonzero().item() + 1
        if rank == 1:
            correct += 1
        total_rank += rank
        ranks.append(rank)
    return correct / len(facts), total_rank / len(facts), ranks

def fluency_check(model, tok, prompts, max_len=30):
    """Generate text and check perplexity / coherence."""
    results = []
    for p in prompts:
        inp = tok(p, return_tensors='pt')
        with torch.no_grad():
            out = model.generate(
                inp['input_ids'], max_length=max_len,
                do_sample=False, pad_token_id=tok.eos_token_id
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        results.append(text)
    return results

def main():
    print("[P86] The Ultimate Neural Lobotomy")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    hd = 64

    # Suppressors to mute (from P83) and Helpers to boost
    suppressors = [(9, 6), (10, 7), (11, 7), (9, 7), (11, 1)]
    helpers = [(2, 1)]  # L2H1 is the strongest helper

    configs = [
        ('baseline', [], [], 1.0),
        ('mute_top3', [(9,6),(10,7),(11,7)], [], 1.0),
        ('mute_all5', suppressors, [], 1.0),
        ('boost_helper_2x', [], helpers, 2.0),
        ('mute3_boost_helper', [(9,6),(10,7),(11,7)], helpers, 2.0),
        ('mute5_boost_helper', suppressors, helpers, 2.0),
        ('mute3_scale0.5', [(9,6),(10,7),(11,7)], [], 0.5),
    ]

    results = []
    fluency_prompts = [
        "The weather today is",
        "Once upon a time there was",
        "The president announced that",
    ]

    for config_name, mute_heads, boost_heads, boost_factor in configs:
        print(f"\n  Config: {config_name}")
        model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()

        # Apply surgical modifications
        with torch.no_grad():
            for layer, head in mute_heads:
                # Zero out W_O for this head
                start = head * hd
                end = start + hd
                if config_name.endswith('scale0.5'):
                    model.transformer.h[layer].attn.c_proj.weight[start:end, :] *= 0.5
                else:
                    model.transformer.h[layer].attn.c_proj.weight[start:end, :] = 0
                    # Also zero the bias contribution
                    if model.transformer.h[layer].attn.c_proj.bias is not None:
                        model.transformer.h[layer].attn.c_proj.bias[start:end] = 0

            for layer, head in boost_heads:
                start = head * hd
                end = start + hd
                model.transformer.h[layer].attn.c_proj.weight[start:end, :] *= boost_factor
                if model.transformer.h[layer].attn.c_proj.bias is not None:
                    model.transformer.h[layer].attn.c_proj.bias[start:end] *= boost_factor

        acc, mean_rank, ranks = evaluate(model, tok, FACTS)
        fluency = fluency_check(model, tok, fluency_prompts)

        results.append({
            'config': config_name,
            'accuracy': acc,
            'mean_rank': mean_rank,
            'median_rank': float(np.median(ranks)),
            'sample_fluency': fluency[:2],
        })
        print(f"    acc={acc:.0%}, mean_rank={mean_rank:.1f}, median={np.median(ranks):.0f}")
        for i, t in enumerate(fluency[:2]):
            safe_text = t.encode('ascii', 'replace').decode('ascii')
            print(f"    fluency[{i}]: {safe_text[:80]}...")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out = {'phase': 86, 'name': 'Ultimate Neural Lobotomy', 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase86_neural_lobotomy.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = [r['config'] for r in results]
    accs = [r['accuracy'] for r in results]
    mean_ranks = [r['mean_rank'] for r in results]
    med_ranks = [r['median_rank'] for r in results]
    colors = ['gray', '#e74c3c', '#c0392b', '#3498db', '#2ecc71', '#27ae60', '#f39c12']

    axes[0].barh(names, accs, color=colors)
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Factual Accuracy')
    axes[0].set_xlim(0, max(accs)*1.3 + 0.05)

    axes[1].barh(names, mean_ranks, color=colors)
    axes[1].set_xlabel('Mean Rank')
    axes[1].set_title('Mean Fact Rank (lower=better)')

    axes[2].barh(names, med_ranks, color=colors)
    axes[2].set_xlabel('Median Rank')
    axes[2].set_title('Median Fact Rank (lower=better)')

    fig.suptitle('Phase 86: The Ultimate Neural Lobotomy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase86_neural_lobotomy.png'), dpi=150)
    plt.close()
    print("[Phase 86] Complete.")

if __name__ == '__main__':
    main()
