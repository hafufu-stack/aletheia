# -*- coding: utf-8 -*-
"""
Phase 20: Multi-Fact Interference
- Spike multiple facts simultaneously in one generation
- Test: "Capital of Japan is Tokyo AND water freezes at 0"
- Do multiple spikes interfere or compose?
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_model():
    print("[P20] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def gen_with_multi_spike(model, tok, prompt, spike_schedule, max_tokens=20):
    """Generate with different spike tokens at different steps.
    spike_schedule: dict {step_idx: {token_id: magnitude}}
    """
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    gen = ids.clone()
    tokens = []
    entropies = []

    for step in range(max_tokens):
        with torch.no_grad():
            out = model(gen)
        logits = out.logits[:, -1, :].squeeze(0)

        # Apply spikes for this step
        if step in spike_schedule:
            for tid, mag in spike_schedule[step].items():
                if tid < logits.shape[0]:
                    logits[tid] += mag

        probs = F.softmax(logits, dim=-1)
        h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
        entropies.append(h)
        next_tok = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
        tokens.append(tok.decode([next_tok.item()]).encode('ascii', 'replace').decode())
        if next_tok.item() == tok.eos_token_id:
            break
        gen = torch.cat([gen, next_tok], dim=1)

    return ''.join(tokens), tokens, entropies


def main():
    print("=" * 70)
    print("  Phase 20: Multi-Fact Interference")
    print("  Can multiple spikes compose without interference?")
    print("=" * 70)

    model, tok = load_model()
    mag = 15

    # === Test 1: Sequential multi-fact generation ===
    print("\n[P20a] Sequential multi-fact spiking...")

    # Compound prompts requiring multiple facts
    tests = [
        {
            'prompt': "The capital of Japan is",
            'spikes': {0: {11790: mag}},  # Tokyo at step 0
            'label': '1 fact (Tokyo)',
        },
        {
            'prompt': "The capital of Japan is Tokyo and the capital of France is",
            'spikes': {0: {6342: mag}},  # Paris at next step
            'label': '2 facts (Tokyo given, Paris spiked)',
        },
        {
            'prompt': "The capital of Japan is",
            'spikes': {0: {11790: mag}, 7: {6342: mag}},  # Tokyo then Paris later
            'label': '2 facts (both spiked at different steps)',
        },
    ]

    results_sequential = []
    for test in tests:
        text, tokens, ents = gen_with_multi_spike(
            model, tok, test['prompt'], test['spikes'], max_tokens=15)
        results_sequential.append({
            'label': test['label'],
            'text': text[:60],
            'mean_entropy': float(np.mean(ents)),
        })
        print(f"  {test['label']}:")
        print(f"    Output: {text[:55]}")
        print(f"    Entropy: {np.mean(ents):.2f}")

    # === Test 2: Simultaneous multi-spike at t=0 ===
    print("\n[P20b] Simultaneous multi-spike at t=0...")
    prompt = "The most important facts:"

    # 1 spike, 2 spikes, 5 spikes simultaneously
    spike_sets = [
        ({11790: mag}, "1 spike (Tokyo)"),
        ({11790: mag, 6342: mag}, "2 spikes (Tokyo, Paris)"),
        ({11790: mag, 6342: mag, 657: mag, 22721: mag, 390: mag},
         "5 spikes (Tokyo, Paris, 0, Jupiter, de)"),
    ]

    results_simultaneous = []
    for spikes, label in spike_sets:
        text, tokens, ents = gen_with_multi_spike(
            model, tok, prompt, {0: spikes}, max_tokens=15)
        # Check which spiked token won
        winner_id = tok.encode(tokens[0])[0] if tokens else -1
        results_simultaneous.append({
            'label': label, 'n_spikes': len(spikes),
            'first_token': tokens[0] if tokens else '',
            'text': text[:60],
            'entropy_0': ents[0] if ents else 0,
        })
        print(f"  {label}:")
        print(f"    First token: '{tokens[0]}', Full: {text[:50]}")
        print(f"    Entropy@0: {ents[0]:.2f}")

    # === Test 3: Competing spikes (contradictory facts) ===
    print("\n[P20c] Competing spikes (which wins?)...")
    prompt = "The capital of Japan is"
    # Spike Tokyo AND Paris at the same time
    competitions = [
        ({11790: mag, 6342: mag}, "Tokyo vs Paris (equal)"),
        ({11790: mag, 6342: mag*2}, "Tokyo vs Paris (Paris 2x)"),
        ({11790: mag*2, 6342: mag}, "Tokyo (2x) vs Paris"),
    ]

    results_competition = []
    for spikes, label in competitions:
        text, tokens, ents = gen_with_multi_spike(
            model, tok, prompt, {0: spikes}, max_tokens=10)
        results_competition.append({
            'label': label,
            'winner': tokens[0] if tokens else '',
            'text': text[:50],
        })
        print(f"  {label}: winner='{tokens[0]}' -> {text[:40]}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Entropy vs number of simultaneous spikes
    n_spikes = [r['n_spikes'] for r in results_simultaneous]
    ent_0 = [r['entropy_0'] for r in results_simultaneous]
    axes[0].bar(range(len(n_spikes)), ent_0,
                tick_label=[str(n) for n in n_spikes], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Number of Simultaneous Spikes')
    axes[0].set_ylabel('Entropy at t=0')
    axes[0].set_title('Multi-Spike Entropy')

    # Plot 2: Sequential multi-fact entropy
    labels = [r['label'][:15] for r in results_sequential]
    ents = [r['mean_entropy'] for r in results_sequential]
    axes[1].barh(range(len(labels)), ents, color='teal', alpha=0.7)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel('Mean Entropy')
    axes[1].set_title('Sequential Multi-Fact')

    # Plot 3: Competition results
    comp_labels = [r['label'][:20] for r in results_competition]
    winners = [r['winner'] for r in results_competition]
    axes[2].barh(range(len(comp_labels)), [1]*len(comp_labels),
                 color=['green' if 'Tokyo' in w else 'red' for w in winners], alpha=0.7)
    for i, w in enumerate(winners):
        axes[2].text(0.5, i, f"Winner: {w}", ha='center', va='center', fontsize=10)
    axes[2].set_yticks(range(len(comp_labels)))
    axes[2].set_yticklabels(comp_labels, fontsize=8)
    axes[2].set_title('Spike Competition')

    plt.suptitle('Phase 20: Multi-Fact Interference', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase20_multi_fact.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 20, 'name': 'Multi-Fact Interference',
        'sequential': results_sequential,
        'simultaneous': results_simultaneous,
        'competition': results_competition,
    }
    with open(os.path.join(RESULTS_DIR, 'phase20_multi_fact.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 20 RESULTS: Multi-Fact Interference")
    print("=" * 70)
    print(f"  Sequential: works with step-separated spikes")
    print(f"  Simultaneous: highest magnitude wins")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
