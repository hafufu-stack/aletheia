# -*- coding: utf-8 -*-
"""
Phase 89: The Synergy Map - Combined Suppressor Ablation + Code Mode
Test whether Code Mode Switch and surgical ablation are additive or redundant.
If additive: the two mechanisms target different suppression pathways.
If redundant: they neutralize the same Grammar Police through different means.
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

def eval_with_config(model, tok, facts, use_comment=False, ablate_heads=None):
    """Evaluate with optional code mode and head ablation."""
    correct = 0
    total_rank = 0
    ranks = []
    hd = 64

    for prompt, answer in facts:
        if use_comment:
            full_prompt = f"# {prompt}"
        else:
            full_prompt = prompt

        inp = tok(full_prompt, return_tensors='pt')
        fact_id = tok.encode(answer)[0]

        handles = []
        if ablate_heads:
            for layer, head in ablate_heads:
                def make_hook(l, h):
                    def hook_fn(module, input, output):
                        if output.dim() == 2:
                            modified = output.clone()
                            start = h * hd
                            end = start + hd
                            modified[:, start:end] = 0
                            return modified
                        else:
                            modified = output.clone()
                            start = h * hd
                            end = start + hd
                            modified[:, :, start:end] = 0
                            return modified
                    return hook_fn
                handle = model.transformer.h[layer].attn.c_proj.register_forward_hook(
                    make_hook(layer, head))
                handles.append(handle)

        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]

        for h in handles:
            h.remove()

        rank = (logits.argsort(descending=True) == fact_id).nonzero().item() + 1
        if rank == 1:
            correct += 1
        total_rank += rank
        ranks.append(rank)

    return correct / len(facts), total_rank / len(facts), ranks

def main():
    print("[P89] The Synergy Map - Ablation x Code Mode Interaction")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

    ablation_configs = {
        'none': [],
        'L9H6': [(9, 6)],
        'L11H7': [(11, 7)],
        'L9H6+L10H7': [(9, 6), (10, 7)],
        'top3': [(9, 6), (10, 7), (11, 7)],
        'top3+L9H7': [(9, 6), (9, 7), (10, 7), (11, 7)],
    }

    results = []
    for abl_name, abl_heads in ablation_configs.items():
        for use_comment in [False, True]:
            mode = 'comment' if use_comment else 'natural'
            config_name = f"{abl_name}_{mode}"
            acc, mean_rank, ranks = eval_with_config(
                model, tok, FACTS, use_comment=use_comment, ablate_heads=abl_heads)
            results.append({
                'config': config_name,
                'ablation': abl_name,
                'mode': mode,
                'accuracy': acc,
                'mean_rank': mean_rank,
                'median_rank': float(np.median(ranks)),
            })
            print(f"  {config_name:30s}: acc={acc:.0%}, mean_rank={mean_rank:.1f}")

    # Compute synergy scores
    baseline_nat = next(r for r in results if r['config'] == 'none_natural')
    baseline_com = next(r for r in results if r['config'] == 'none_comment')

    synergy_data = []
    for abl_name in ablation_configs:
        if abl_name == 'none':
            continue
        nat = next(r for r in results if r['ablation'] == abl_name and r['mode'] == 'natural')
        com = next(r for r in results if r['ablation'] == abl_name and r['mode'] == 'comment')

        # Additive prediction: (ablation_gain + comment_gain)
        abl_gain = baseline_nat['mean_rank'] - nat['mean_rank']
        com_gain = baseline_nat['mean_rank'] - baseline_com['mean_rank']
        predicted_additive = baseline_nat['mean_rank'] - (abl_gain + com_gain)
        actual_combined = com['mean_rank']

        synergy = predicted_additive - actual_combined  # positive = super-additive
        synergy_data.append({
            'ablation': abl_name,
            'abl_gain': abl_gain,
            'comment_gain': com_gain,
            'predicted_additive': predicted_additive,
            'actual_combined': actual_combined,
            'synergy': synergy,
        })

    out = {'phase': 89, 'name': 'Synergy Map', 'results': results, 'synergy': synergy_data}
    with open(os.path.join(RESULTS_DIR, 'phase89_synergy_map.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Heatmap-style: ablation x mode
    abl_names = list(ablation_configs.keys())
    nat_accs = [next(r['accuracy'] for r in results if r['ablation'] == a and r['mode'] == 'natural') for a in abl_names]
    com_accs = [next(r['accuracy'] for r in results if r['ablation'] == a and r['mode'] == 'comment') for a in abl_names]

    x = np.arange(len(abl_names))
    w = 0.35
    axes[0].bar(x - w/2, nat_accs, w, label='Natural', color='#e74c3c', alpha=0.8)
    axes[0].bar(x + w/2, com_accs, w, label='Comment (#)', color='#2ecc71', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(abl_names, rotation=30, ha='right')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy: Ablation x Code Mode')
    axes[0].legend()

    nat_ranks = [next(r['mean_rank'] for r in results if r['ablation'] == a and r['mode'] == 'natural') for a in abl_names]
    com_ranks = [next(r['mean_rank'] for r in results if r['ablation'] == a and r['mode'] == 'comment') for a in abl_names]
    axes[1].bar(x - w/2, nat_ranks, w, label='Natural', color='#e74c3c', alpha=0.8)
    axes[1].bar(x + w/2, com_ranks, w, label='Comment (#)', color='#2ecc71', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(abl_names, rotation=30, ha='right')
    axes[1].set_ylabel('Mean Rank')
    axes[1].set_title('Mean Rank: Ablation x Code Mode')
    axes[1].legend()

    # Synergy plot
    if synergy_data:
        syn_names = [s['ablation'] for s in synergy_data]
        syn_values = [s['synergy'] for s in synergy_data]
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in syn_values]
        axes[2].barh(syn_names, syn_values, color=colors)
        axes[2].axvline(x=0, color='black', linewidth=0.5)
        axes[2].set_xlabel('Synergy Score (>0 = super-additive)')
        axes[2].set_title('Ablation + Code Mode Synergy')

    fig.suptitle('Phase 89: The Synergy Map', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase89_synergy_map.png'), dpi=150)
    plt.close()
    print("[Phase 89] Complete.")

if __name__ == '__main__':
    main()
