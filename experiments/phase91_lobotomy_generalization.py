# -*- coding: utf-8 -*-
"""
Phase 91: Lobotomized Model Generalization
P86 showed lobotomy works on training facts. Does it generalize to
completely new, unseen factual prompts?
"""
import torch, json, os, sys, numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

# Original facts used in P86 training
TRAIN_FACTS = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
]

# Completely new facts NOT in any previous experiment
TEST_FACTS = [
    ("The capital of Australia is", " Canberra"),
    ("The capital of Canada is", " Ottawa"),
    ("The capital of Brazil is", " Bras"),
    ("The capital of Egypt is", " Cairo"),
    ("The capital of South Korea is", " Seoul"),
    ("The capital of Sweden is", " Stockholm"),
    ("The capital of Poland is", " Warsaw"),
    ("The capital of Turkey is", " Ankara"),
    ("The chemical symbol for silver is", " Ag"),
    ("The chemical symbol for iron is", " Fe"),
    ("The chemical symbol for sodium is", " Na"),
    ("The chemical symbol for potassium is", " K"),
    ("The largest continent is", " Asia"),
    ("The smallest planet is", " Mercury"),
    ("The deepest ocean is the", " Pacific"),
    ("The longest river in the world is the", " N"),
    ("The inventor of the telephone is", " Alexander"),
    ("The theory of relativity was proposed by", " Albert"),
    ("The discoverer of penicillin is", " Alexander"),
    ("The author of Hamlet is", " William"),
    ("The first element in the periodic table is", " hydrogen"),
    ("The speed of light in a vacuum is", " 299"),
    ("The melting point of ice is", " 0"),
    ("The largest mammal is the", " blue"),
    ("The hardest natural substance is", " diamond"),
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
    print("[P91] Lobotomized Model Generalization")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    hd = 64

    configs = [
        ('baseline', [], []),
        ('mute_top3', [(9,6),(10,7),(11,7)], []),
        ('mute_top3+boost_L2H1', [(9,6),(10,7),(11,7)], [(2,1)]),
    ]

    results = []
    for config_name, mute_heads, boost_heads in configs:
        model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
        with torch.no_grad():
            for layer, head in mute_heads:
                start = head * hd
                end = start + hd
                model.transformer.h[layer].attn.c_proj.weight[start:end, :] = 0
                if model.transformer.h[layer].attn.c_proj.bias is not None:
                    model.transformer.h[layer].attn.c_proj.bias[start:end] = 0
            for layer, head in boost_heads:
                start = head * hd
                end = start + hd
                model.transformer.h[layer].attn.c_proj.weight[start:end, :] *= 2.0
                if model.transformer.h[layer].attn.c_proj.bias is not None:
                    model.transformer.h[layer].attn.c_proj.bias[start:end] *= 2.0

        train_acc, train_rank, train_ranks = evaluate(model, tok, TRAIN_FACTS)
        test_acc, test_rank, test_ranks = evaluate(model, tok, TEST_FACTS)

        results.append({
            'config': config_name,
            'train_accuracy': train_acc,
            'train_mean_rank': train_rank,
            'test_accuracy': test_acc,
            'test_mean_rank': test_rank,
            'train_median': float(np.median(train_ranks)),
            'test_median': float(np.median(test_ranks)),
        })
        print(f"  {config_name:30s}: train={train_acc:.0%}/rank={train_rank:.1f}, test={test_acc:.0%}/rank={test_rank:.1f}")
        del model

    out = {'phase': 91, 'name': 'Lobotomy Generalization', 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase91_lobotomy_generalization.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = [r['config'] for r in results]
    train_accs = [r['train_accuracy'] for r in results]
    test_accs = [r['test_accuracy'] for r in results]
    train_ranks = [r['train_mean_rank'] for r in results]
    test_ranks = [r['test_mean_rank'] for r in results]

    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w/2, train_accs, w, label='Train Facts', color='#3498db', alpha=0.8)
    axes[0].bar(x + w/2, test_accs, w, label='Test Facts (unseen)', color='#e74c3c', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha='right')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy: Train vs Test')
    axes[0].legend()

    axes[1].bar(x - w/2, train_ranks, w, label='Train Facts', color='#3498db', alpha=0.8)
    axes[1].bar(x + w/2, test_ranks, w, label='Test Facts (unseen)', color='#e74c3c', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha='right')
    axes[1].set_ylabel('Mean Rank')
    axes[1].set_title('Mean Rank: Train vs Test')
    axes[1].legend()

    fig.suptitle('Phase 91: Does Neural Lobotomy Generalize?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase91_lobotomy_generalization.png'), dpi=150)
    plt.close()
    print("[Phase 91] Complete.")

if __name__ == '__main__':
    main()
