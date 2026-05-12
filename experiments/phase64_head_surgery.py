# -*- coding: utf-8 -*-
"""
Phase 64: Targeted Suppressor Head Surgery
P61 found L11H7 is the #1 fact suppressor (+829 rank improvement when ablated).
What if we ablate ONLY the top suppressor heads during generation?
Does fact accuracy improve while grammar is preserved?
"""
import os, json, sys
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import phase_complete

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_model():
    print("[P64] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def make_ablation_hook(head_indices, n_heads, hidden_dim):
    """Zero out specific heads in a layer."""
    head_dim = hidden_dim // n_heads
    def hook_fn(module, args, output):
        hs = output[0].clone()
        for hi in head_indices:
            start = hi * head_dim
            end = start + head_dim
            hs[:, :, start:end] = 0.0
        return (hs,) + output[1:]
    return hook_fn

def main():
    print("=" * 70)
    print("  Phase 64: Targeted Suppressor Head Surgery")
    print("  Ablate L11H7 (top suppressor) during generation")
    print("=" * 70)

    model, tok = load_model()
    n_heads = model.config.n_head  # 12
    hidden_dim = model.config.n_embd  # 768

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The Earth orbits the", [4252], "Sun"),
        ("The boiling point of water is", [1802], "100"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("Oxygen has the atomic number", [807], "8"),
    ]

    # Top suppressors from P61
    suppressor_configs = [
        ("none", {}),
        ("L11H7", {11: [7]}),
        ("L11H7+L10H7", {10: [7], 11: [7]}),
        ("L11H7+H1+H4", {11: [7, 1, 4]}),
        ("top3_both", {10: [7, 6], 11: [7, 1, 4]}),
    ]

    all_results = []
    for config_name, ablation_map in suppressor_configs:
        print(f"\n[P64] Config: {config_name}")
        config_correct = 0
        config_ranks = []
        config_details = []

        for prompt, fact_ids, expected in tests:
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

            handles = []
            for layer_idx, head_indices in ablation_map.items():
                h = model.transformer.h[layer_idx].register_forward_hook(
                    make_ablation_hook(head_indices, n_heads, hidden_dim))
                handles.append(h)

            with torch.no_grad():
                out = model(inp)
            for h in handles:
                h.remove()

            logits = out.logits[0, -1, :]
            rank = get_fact_rank(logits, fact_ids[0])
            correct = torch.argmax(logits).item() in fact_ids
            top1 = tok.decode([torch.argmax(logits).item()]).encode('ascii','replace').decode().strip()

            if correct: config_correct += 1
            config_ranks.append(rank)
            config_details.append({
                'expected': expected, 'rank': rank, 'correct': correct, 'top1': top1
            })
            tag = 'OK' if correct else f'r{rank}'
            print(f"  {expected:>12s}: [{tag:>6s}] top1={top1[:8]}")

        acc = config_correct / len(tests)
        med_rank = float(np.median(config_ranks))
        all_results.append({
            'config': config_name, 'accuracy': acc, 'median_rank': med_rank,
            'mean_rank': float(np.mean(config_ranks)), 'details': config_details,
        })
        print(f"  -> {config_name}: acc={acc:.0%} median_rank={med_rank:.0f}")

    # === Multi-token generation with surgery ===
    print("\n[P64b] Multi-token generation with L11H7 ablation...")
    gen_results = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

        # Baseline
        with torch.no_grad():
            base_out = model.generate(inp, max_new_tokens=15, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        base_text = tok.decode(base_out[0][inp.shape[1]:]).encode('ascii','replace').decode()

        # With L11H7 ablated
        handle = model.transformer.h[11].register_forward_hook(
            make_ablation_hook([7], n_heads, hidden_dim))
        with torch.no_grad():
            surg_out = model.generate(inp, max_new_tokens=15, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
        handle.remove()
        surg_text = tok.decode(surg_out[0][inp.shape[1]:]).encode('ascii','replace').decode()

        gen_results.append({
            'expected': expected, 'base': base_text[:50], 'surgery': surg_text[:50],
        })
        print(f"  {expected:>12s}:")
        print(f"    Base:    {base_text[:45]}")
        print(f"    Surgery: {surg_text[:45]}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Accuracy comparison
    configs = [r['config'] for r in all_results]
    accs = [r['accuracy']*100 for r in all_results]
    colors = ['red'] + ['green']*(len(configs)-1)
    axes[0].bar(range(len(configs)), accs, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_xticks(range(len(configs)))
    axes[0].set_xticklabels(configs, fontsize=7, rotation=30)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Fact Accuracy by Ablation Config')

    # 2. Median rank comparison
    med_ranks = [r['median_rank'] for r in all_results]
    axes[1].bar(range(len(configs)), med_ranks, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(configs)))
    axes[1].set_xticklabels(configs, fontsize=7, rotation=30)
    axes[1].set_ylabel('Median Fact Rank')
    axes[1].set_title('Median Rank (lower=better)')
    axes[1].set_yscale('log')

    # 3. Per-prompt rank comparison (none vs L11H7)
    none_ranks = [d['rank'] for d in all_results[0]['details']]
    l11h7_ranks = [d['rank'] for d in all_results[1]['details']]
    labels = [d['expected'][:6] for d in all_results[0]['details']]
    x = range(len(labels))
    axes[2].bar([i-0.2 for i in x], none_ranks, 0.4, label='No ablation', color='red', alpha=0.7)
    axes[2].bar([i+0.2 for i in x], l11h7_ranks, 0.4, label='L11H7 ablated', color='green', alpha=0.7)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[2].set_ylabel('Fact Rank')
    axes[2].set_title('L11H7 Ablation Effect')
    axes[2].legend(fontsize=8)
    axes[2].set_yscale('log')

    plt.suptitle('Phase 64: Targeted Suppressor Head Surgery', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase64_head_surgery.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 64, 'name': 'Targeted Suppressor Head Surgery',
        'ablation_results': all_results,
        'generation_results': gen_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase64_head_surgery.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 64 RESULTS")
    for r in all_results:
        print(f"  {r['config']:>20s}: acc={r['accuracy']:.0%} med_rank={r['median_rank']:.0f}")
    print("=" * 70)
    phase_complete(64)

if __name__ == '__main__':
    main()
