# -*- coding: utf-8 -*-
"""
Phase 94: L2H1 - The Greatest Ally
P83 identified L2H1 as the strongest helper (-1268 suppression score).
Investigate: What does it attend to? What happens when we ablate it?
What happens when we boost it?
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
    ("The Pythagorean theorem states that a squared plus b squared equals", " c"),
    ("The human body has a total of", " 206"),
    ("DNA stands for de", "oxy"),
    ("The Great Wall of China is located in", " China"),
    ("Shakespeare was born in", " Strat"),
]

def evaluate(model, tok, facts):
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

def main():
    print("[P94] L2H1 - The Greatest Ally")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    hd = 64

    # === Part 1: Attention Pattern Analysis ===
    print("\n  === Part 1: L2H1 Attention Analysis ===")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True, attn_implementation='eager').eval()

    l2h1_patterns = []
    for prompt, answer in FACTS[:10]:
        inp = tok(prompt, return_tensors='pt')
        tokens = [tok.decode([t]) for t in inp['input_ids'][0]]
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        # L2H1 attention from last token
        attn = out.attentions[2][0, 1, -1, :].detach().numpy()
        top_indices = np.argsort(attn)[-3:][::-1]
        top_tokens = [(tokens[i], float(attn[i])) for i in top_indices]
        entropy = float(-(attn * np.log(attn + 1e-10)).sum())
        l2h1_patterns.append({
            'prompt': prompt[:40],
            'top_attended': top_tokens,
            'entropy': entropy,
        })
        top_str = ', '.join(f'{t}({w:.2f})' for t, w in top_tokens)
        safe_str = top_str.encode('ascii', 'replace').decode('ascii')
        print(f"    {prompt[:35]:35s} -> top: {safe_str}, H={entropy:.3f}")
    del model

    # === Part 2: Ablation & Boost Sweep ===
    print("\n  === Part 2: L2H1 Scale Sweep ===")
    scales = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    scale_results = []

    for scale in scales:
        model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
        with torch.no_grad():
            start = 1 * hd  # head 1
            end = start + hd
            model.transformer.h[2].attn.c_proj.weight[start:end, :] *= scale
            if model.transformer.h[2].attn.c_proj.bias is not None:
                model.transformer.h[2].attn.c_proj.bias[start:end] *= scale

        acc, mean_rank, ranks = evaluate(model, tok, FACTS)
        scale_results.append({
            'scale': scale,
            'accuracy': acc,
            'mean_rank': mean_rank,
            'median_rank': float(np.median(ranks)),
        })
        print(f"    scale={scale:.2f}: acc={acc:.0%}, rank={mean_rank:.1f}")
        del model

    # === Part 3: L2H1 vs Suppressor Interaction ===
    print("\n  === Part 3: Helper vs Suppressor Interaction ===")
    interaction_results = []
    configs = [
        ('baseline', [], []),
        ('ablate_L2H1', [(2, 1, 0.0)], []),
        ('boost_L2H1_3x', [(2, 1, 3.0)], []),
        ('mute_L9H6', [], [(9, 6, 0.0)]),
        ('mute_L9H6+boost_L2H1', [(2, 1, 3.0)], [(9, 6, 0.0)]),
        ('mute_top3+boost_L2H1_5x', [(2, 1, 5.0)], [(9, 6, 0.0), (10, 7, 0.0), (11, 7, 0.0)]),
    ]
    for config_name, helper_ops, suppressor_ops in configs:
        model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
        with torch.no_grad():
            for layer, head, scale in helper_ops + suppressor_ops:
                start = head * hd
                end = start + hd
                model.transformer.h[layer].attn.c_proj.weight[start:end, :] *= scale
                if model.transformer.h[layer].attn.c_proj.bias is not None:
                    model.transformer.h[layer].attn.c_proj.bias[start:end] *= scale

        acc, mean_rank, ranks = evaluate(model, tok, FACTS)
        interaction_results.append({
            'config': config_name,
            'accuracy': acc,
            'mean_rank': mean_rank,
        })
        print(f"    {config_name:35s}: acc={acc:.0%}, rank={mean_rank:.1f}")
        del model

    out = {
        'phase': 94, 'name': 'L2H1 - The Greatest Ally',
        'attention_patterns': l2h1_patterns,
        'scale_sweep': scale_results,
        'interactions': interaction_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase94_l2h1_ally.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scale sweep
    ss = [r['scale'] for r in scale_results]
    accs = [r['accuracy'] for r in scale_results]
    ranks = [r['mean_rank'] for r in scale_results]
    axes[0].plot(ss, accs, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    axes[0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='baseline')
    axes[0].set_xlabel('L2H1 Scale Factor')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('L2H1 Scale vs Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ss, ranks, 's-', color='#e74c3c', linewidth=2, markersize=8)
    axes[1].axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('L2H1 Scale Factor')
    axes[1].set_ylabel('Mean Rank')
    axes[1].set_title('L2H1 Scale vs Mean Rank')
    axes[1].grid(True, alpha=0.3)

    # Interactions
    int_names = [r['config'] for r in interaction_results]
    int_accs = [r['accuracy'] for r in interaction_results]
    colors = ['gray', '#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12']
    axes[2].barh(int_names, int_accs, color=colors)
    axes[2].set_xlabel('Accuracy')
    axes[2].set_title('Helper x Suppressor Interaction')

    fig.suptitle('Phase 94: L2H1 - The Greatest Ally', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase94_l2h1_ally.png'), dpi=150)
    plt.close()
    print("[Phase 94] Complete.")

if __name__ == '__main__':
    main()
