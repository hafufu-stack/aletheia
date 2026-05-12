# -*- coding: utf-8 -*-
"""
Phase 82: The Code Mode Switch
P80 showed ALL symbols work equally well (25% vs 0% natural).
Hypothesis: GPT-2 has a binary "mode switch" between natural
language processing and code/structured processing.
Find the NEURONS that activate differently for symbol vs natural.
These are the "code mode switch" neurons.
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
    print("[P82] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def main():
    print("=" * 70)
    print("  Phase 82: The Code Mode Switch")
    print("  Finding neurons that flip between natural vs symbol mode")
    print("=" * 70)

    model, tok = load_model()

    facts = [
        ("Japan capital", 11790, "Tokyo"),
        ("France capital", 6342, "Paris"),
        ("largest planet", 22721, "Jupiter"),
        ("Earth orbits", 4252, "Sun"),
        ("gold symbol", 7591, "Au"),
        ("water freezing point", 657, "0"),
        ("Shakespeare play", 13483, "Hamlet"),
        ("oxygen atomic number", 807, "8"),
        ("Einstein theory", 44449, "relativity"),
        ("water boiling point", 1802, "100"),
    ]

    # Collect MLP activations for each format at key layers
    layers_to_check = [8, 9, 10, 11]

    natural_activations = {l: [] for l in layers_to_check}
    symbol_activations = {l: [] for l in layers_to_check}

    for desc, fact_id, expected in facts:
        for fmt, container in [('The {desc} is', natural_activations),
                               ('# {desc}:', symbol_activations)]:
            prompt = fmt.format(desc=desc)
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

            mlp_outputs = {}
            handles = []
            for li in layers_to_check:
                def make_hook(idx):
                    def fn(m, a, o):
                        mlp_outputs[idx] = o[0, -1, :].detach().cpu().numpy()
                    return fn
                handles.append(model.transformer.h[li].mlp.register_forward_hook(make_hook(li)))

            with torch.no_grad():
                model(inp)
            for h in handles:
                h.remove()

            for li in layers_to_check:
                container[li].append(mlp_outputs[li])

    # For each layer, find neurons with largest activation difference
    switch_neurons = {}
    for li in layers_to_check:
        nat = np.array(natural_activations[li])  # (n_facts, hidden_dim*4)
        sym = np.array(symbol_activations[li])

        # Mean activation per neuron
        nat_mean = nat.mean(axis=0)
        sym_mean = sym.mean(axis=0)

        # Difference
        diff = sym_mean - nat_mean
        abs_diff = np.abs(diff)

        # Top 10 most different neurons
        top_indices = abs_diff.argsort()[-10:][::-1]
        top_diffs = [(int(i), float(diff[i]), float(nat_mean[i]), float(sym_mean[i]))
                     for i in top_indices]

        switch_neurons[li] = top_diffs

        print(f"\n  Layer {li} top switch neurons:")
        for idx, d, nat_v, sym_v in top_diffs[:5]:
            direction = 'ON' if d > 0 else 'OFF'
            print(f"    Neuron {idx:>4d}: delta={d:>+8.3f} "
                  f"(nat={nat_v:>7.3f} sym={sym_v:>7.3f}) [{direction} for code]")

    # Identify THE most mode-switching neurons globally
    all_neurons = []
    for li, neurons in switch_neurons.items():
        for idx, d, nat_v, sym_v in neurons:
            all_neurons.append((li, idx, abs(d), d, nat_v, sym_v))
    all_neurons.sort(key=lambda x: x[2], reverse=True)

    print("\n  GLOBAL top code-mode switch neurons:")
    for li, idx, abs_d, d, nat_v, sym_v in all_neurons[:10]:
        direction = 'ON' if d > 0 else 'OFF'
        print(f"    L{li}:N{idx:>4d}: |delta|={abs_d:>8.3f} [{direction} for code]")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Top neuron activation difference by layer
    for i, li in enumerate(layers_to_check):
        diffs = [abs(d) for _, d, _, _ in switch_neurons[li][:5]]
        axes[0].bar([i*6+j for j in range(5)], diffs, color=f'C{i}', alpha=0.7,
                   label=f'L{li}')
    axes[0].set_xlabel('Neuron Index')
    axes[0].set_ylabel('|Activation Difference|')
    axes[0].set_title('Top Switch Neurons by Layer')
    axes[0].legend()

    # 2. Natural vs Symbol activation for top 10 global neurons
    top10 = all_neurons[:10]
    labels = [f'L{li}:N{idx}' for li, idx, _, _, _, _ in top10]
    nat_vals = [nat_v for _, _, _, _, nat_v, _ in top10]
    sym_vals = [sym_v for _, _, _, _, _, sym_v in top10]
    x = range(len(labels))
    axes[1].bar([i-0.2 for i in x], nat_vals, 0.4, label='Natural', color='red', alpha=0.7)
    axes[1].bar([i+0.2 for i in x], sym_vals, 0.4, label='Symbol', color='green', alpha=0.7)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, fontsize=6, rotation=45)
    axes[1].set_ylabel('Mean Activation')
    axes[1].set_title('Top 10 Code-Mode Switch Neurons')
    axes[1].legend(fontsize=8)

    # 3. Distribution of differences across all neurons in L11
    nat_l11 = np.array(natural_activations[11])
    sym_l11 = np.array(symbol_activations[11])
    all_diffs_l11 = (sym_l11.mean(axis=0) - nat_l11.mean(axis=0))
    axes[2].hist(all_diffs_l11, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Activation Difference (Symbol - Natural)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('L11 MLP Neuron Shifts\n(Symbol vs Natural)')
    axes[2].axvline(x=0, color='black', linewidth=0.5)

    plt.suptitle('Phase 82: The Code Mode Switch', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase82_code_mode.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 82, 'name': 'The Code Mode Switch',
        'top_global_neurons': [(li, idx, round(d, 3)) for li, idx, _, d, _, _ in all_neurons[:20]],
        'per_layer_top': {str(li): [(idx, round(d, 3)) for idx, d, _, _ in ns[:5]]
                         for li, ns in switch_neurons.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase82_code_mode.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    top1 = all_neurons[0]
    print("\n" + "=" * 70)
    print(f"  #1 Code Mode Switch: L{top1[0]}:Neuron{top1[1]} "
          f"(delta={top1[3]:+.3f})")
    print(f"  Total neurons with |delta|>1.0: "
          f"{sum(1 for x in all_neurons if x[2] > 1.0)}")
    print("=" * 70)
    phase_complete(82)

if __name__ == '__main__':
    main()
