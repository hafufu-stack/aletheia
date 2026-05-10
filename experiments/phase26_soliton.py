# -*- coding: utf-8 -*-
"""
Phase 26: Semantic Soliton Waves - Multi-Token Fact Generation
Exponentially decaying spike for multi-token facts (e.g. "deoxyribonucleic acid").
Constant spike = "Tokyo Tokyo..." collapse; Soliton wave = perfect multi-token surfing.
"""
import os, json, sys
import numpy as np
import torch
import torch.nn.functional as F
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
    print("[P26] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def gen_with_soliton(model, tok, prompt, token_schedule, max_tokens=25):
    """Generate with soliton spike: list of (token_id, magnitude) per step."""
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    gen = ids.clone()
    tokens = []
    entropies = []
    for step in range(max_tokens):
        with torch.no_grad():
            out = model(gen)
        logits = out.logits[:, -1, :].squeeze(0)
        if step < len(token_schedule):
            tid, mag = token_schedule[step]
            if tid is not None and tid < logits.shape[0] and mag > 0:
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
    print("  Phase 26: Semantic Soliton Waves")
    print("  Decaying spike for multi-token fact generation")
    print("=" * 70)

    model, tok = load_model()

    # Multi-token fact test cases: prompt -> expected BPE tokens
    multi_token_facts = [
        {
            'prompt': "DNA stands for",
            'expected_text': "deoxyribonucleic acid",
            'bpe_tokens': tok.encode(" deoxyribonucleic acid"),
            'label': 'DNA',
        },
        {
            'prompt': "The capital of the United Kingdom is",
            'expected_text': "London",
            'bpe_tokens': tok.encode(" London"),
            'label': 'UK capital',
        },
        {
            'prompt': "Albert Einstein developed the theory of",
            'expected_text': "relativity",
            'bpe_tokens': tok.encode(" relativity"),
            'label': 'Einstein',
        },
        {
            'prompt': "The chemical formula for water is",
            'expected_text': "H2O",
            'bpe_tokens': tok.encode(" H2O"),
            'label': 'Water formula',
        },
        {
            'prompt': "The first president of the United States was",
            'expected_text': "George Washington",
            'bpe_tokens': tok.encode(" George Washington"),
            'label': 'First president',
        },
    ]

    base_mag = 15

    # === P26a: Compare strategies for multi-token generation ===
    print("\n[P26a] Comparing strategies: no spike / constant / soliton / single-t0...")
    strategies = ['none', 'single_t0', 'constant', 'soliton_fast', 'soliton_slow']
    all_results = {s: [] for s in strategies}

    for fact in multi_token_facts:
        bpe = fact['bpe_tokens']
        n_bpe = len(bpe)
        print(f"\n  {fact['label']}: '{fact['expected_text']}' ({n_bpe} BPE tokens: {bpe})")

        for strategy in strategies:
            schedule = []
            for i in range(max(n_bpe, 20)):
                if strategy == 'none':
                    schedule.append((None, 0))
                elif strategy == 'single_t0':
                    if i == 0 and i < n_bpe:
                        schedule.append((bpe[i], base_mag))
                    else:
                        schedule.append((None, 0))
                elif strategy == 'constant':
                    if i < n_bpe:
                        schedule.append((bpe[i], base_mag))
                    else:
                        schedule.append((None, 0))
                elif strategy == 'soliton_fast':
                    if i < n_bpe:
                        mag = base_mag * (0.5 ** i)  # half-life = 1 step
                        schedule.append((bpe[i], mag))
                    else:
                        schedule.append((None, 0))
                elif strategy == 'soliton_slow':
                    if i < n_bpe:
                        mag = base_mag * (0.7 ** i)  # slower decay
                        schedule.append((bpe[i], mag))
                    else:
                        schedule.append((None, 0))

            text, tokens, ents = gen_with_soliton(model, tok, fact['prompt'], schedule, 20)
            # Check how many BPE tokens matched
            gen_ids = tok.encode(text)
            matched = 0
            for j, expected_id in enumerate(bpe):
                if j < len(gen_ids) and gen_ids[j] == expected_id:
                    matched += 1
                else:
                    break
            match_rate = matched / n_bpe if n_bpe > 0 else 0
            all_results[strategy].append({
                'label': fact['label'],
                'match_rate': match_rate,
                'matched': matched, 'total': n_bpe,
                'text': text[:50],
                'mean_entropy': float(np.mean(ents[:n_bpe])) if ents else 0,
            })
            print(f"    {strategy:>14s}: [{matched}/{n_bpe}] {text[:45]}")

    # === P26b: Decay rate sweep ===
    print("\n[P26b] Soliton decay rate sweep...")
    decay_sweep = {}
    for decay in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        total_match = 0
        total_tokens = 0
        for fact in multi_token_facts:
            bpe = fact['bpe_tokens']
            schedule = []
            for i in range(20):
                if i < len(bpe):
                    schedule.append((bpe[i], base_mag * (decay ** i)))
                else:
                    schedule.append((None, 0))
            text, _, _ = gen_with_soliton(model, tok, fact['prompt'], schedule, 20)
            gen_ids = tok.encode(text)
            matched = 0
            for j, eid in enumerate(bpe):
                if j < len(gen_ids) and gen_ids[j] == eid:
                    matched += 1
                else:
                    break
            total_match += matched
            total_tokens += len(bpe)
        decay_sweep[decay] = total_match / total_tokens if total_tokens > 0 else 0
        print(f"  decay={decay:.1f}: {total_match}/{total_tokens} = {decay_sweep[decay]:.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Strategy comparison
    strat_labels = ['None', 'Single t0', 'Constant', 'Soliton\n(fast)', 'Soliton\n(slow)']
    mean_matches = [np.mean([r['match_rate'] for r in all_results[s]])*100 for s in strategies]
    colors = ['red', 'orange', 'purple', 'blue', 'green']
    axes[0].bar(strat_labels, mean_matches, color=colors, alpha=0.7)
    axes[0].set_ylabel('BPE Match Rate (%)')
    axes[0].set_title('Multi-Token Strategy Comparison')
    axes[0].set_ylim(0, 110)

    # Plot 2: Decay rate sweep
    decays = sorted(decay_sweep.keys())
    rates = [decay_sweep[d]*100 for d in decays]
    axes[1].plot(decays, rates, 'g.-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Decay Rate')
    axes[1].set_ylabel('BPE Match Rate (%)')
    axes[1].set_title('Soliton Decay Rate Optimization')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Per-fact entropy comparison
    fact_labels = [r['label'] for r in all_results['none']]
    ent_none = [r['mean_entropy'] for r in all_results['none']]
    ent_sol = [r['mean_entropy'] for r in all_results['soliton_slow']]
    x = range(len(fact_labels))
    axes[2].bar([i-0.2 for i in x], ent_none, 0.4, label='No spike', color='red', alpha=0.7)
    axes[2].bar([i+0.2 for i in x], ent_sol, 0.4, label='Soliton', color='green', alpha=0.7)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(fact_labels, fontsize=7, rotation=30)
    axes[2].set_ylabel('Mean Entropy')
    axes[2].set_title('Entropy: Baseline vs Soliton')
    axes[2].legend(fontsize=8)

    plt.suptitle('Phase 26: Semantic Soliton Waves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase26_soliton.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 26, 'name': 'Semantic Soliton Waves',
        'strategy_results': {s: all_results[s] for s in strategies},
        'decay_sweep': {str(k): v for k, v in decay_sweep.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase26_soliton.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    best_decay = max(decay_sweep, key=decay_sweep.get)
    print("\n" + "=" * 70)
    print("  PHASE 26 RESULTS: Semantic Soliton Waves")
    print("=" * 70)
    print(f"  Best decay rate: {best_decay} ({decay_sweep[best_decay]:.0%})")
    print(f"  Constant spike match: {np.mean([r['match_rate'] for r in all_results['constant']]):.0%}")
    print(f"  Soliton slow match:   {np.mean([r['match_rate'] for r in all_results['soliton_slow']]):.0%}")
    print("=" * 70)

    phase_complete(26)
    return results

if __name__ == '__main__':
    main()
