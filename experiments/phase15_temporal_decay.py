# -*- coding: utf-8 -*-
"""
Phase 15: Spike Temporal Decay
- During multi-token generation, how long does spike effect persist?
- Does one spike at token 1 influence tokens 2, 3, 10?
- Measure decay rate of factual influence
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
    print("[P15] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def generate_with_spike_at_step(model, tok, prompt, fact_ids, spike_mag,
                                spike_step=0, max_tokens=20):
    """Generate tokens, applying spike only at spike_step."""
    input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    generated = input_ids.clone()
    token_entropies = []
    token_texts = []

    for step in range(max_tokens):
        with torch.no_grad():
            out = model(generated)
        logits = out.logits[:, -1, :].squeeze(0)

        # Apply spike only at the designated step
        if step == spike_step:
            for tid in fact_ids:
                if tid < logits.shape[-1]:
                    logits[tid] += spike_mag

        probs = F.softmax(logits, dim=-1)
        h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
        token_entropies.append(h)

        next_tok = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
        token_text = tok.decode([next_tok.item()]).encode('ascii', 'replace').decode()
        token_texts.append(token_text)

        if next_tok.item() == tok.eos_token_id:
            break
        generated = torch.cat([generated, next_tok], dim=1)

    return token_texts, token_entropies


def main():
    print("=" * 70)
    print("  Phase 15: Spike Temporal Decay")
    print("  How long does a single spike influence generation?")
    print("=" * 70)

    model, tok = load_model()

    test_cases = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
    ]

    spike_mag = 20  # Strong spike
    max_tokens = 15

    # === Compare: no spike, spike at step 0, spike at every step ===
    print("\n[P15a] Comparing spike strategies...")
    all_results = []

    for prompt, fact_ids, fact_name in test_cases:
        print(f"\n  Prompt: {prompt}")

        # No spike (baseline)
        texts_none, ent_none = generate_with_spike_at_step(
            model, tok, prompt, fact_ids, 0, spike_step=-1, max_tokens=max_tokens)
        print(f"    No spike:    {''.join(texts_none[:8])}")

        # Spike at step 0 only
        texts_s0, ent_s0 = generate_with_spike_at_step(
            model, tok, prompt, fact_ids, spike_mag, spike_step=0, max_tokens=max_tokens)
        print(f"    Spike@0:     {''.join(texts_s0[:8])}")

        # Spike at every step
        input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        generated = input_ids.clone()
        texts_all = []
        ent_all = []
        for step in range(max_tokens):
            with torch.no_grad():
                out = model(generated)
            logits = out.logits[:, -1, :].squeeze(0)
            for tid in fact_ids:
                if tid < logits.shape[-1]:
                    logits[tid] += spike_mag
            probs = F.softmax(logits, dim=-1)
            h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
            ent_all.append(h)
            next_tok = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
            texts_all.append(tok.decode([next_tok.item()]).encode('ascii', 'replace').decode())
            if next_tok.item() == tok.eos_token_id:
                break
            generated = torch.cat([generated, next_tok], dim=1)
        print(f"    Spike@all:   {''.join(texts_all[:8])}")

        all_results.append({
            'prompt': prompt, 'fact': fact_name,
            'no_spike': {'text': texts_none, 'entropy': ent_none},
            'spike_0': {'text': texts_s0, 'entropy': ent_s0},
            'spike_all': {'text': texts_all, 'entropy': ent_all},
        })

    # === Spike at different positions ===
    print("\n[P15b] Spike position sweep...")
    prompt, fact_ids, _ = test_cases[0]
    position_results = {}

    for spike_pos in range(max_tokens):
        texts, ents = generate_with_spike_at_step(
            model, tok, prompt, fact_ids, spike_mag,
            spike_step=spike_pos, max_tokens=max_tokens)
        # Check if fact token appears in first token
        first_tok_id = tok.encode(texts[0])[0] if texts else -1
        position_results[spike_pos] = {
            'first_token': texts[0] if texts else '',
            'correct_first': first_tok_id in fact_ids if texts else False,
            'entropy_trajectory': ents,
        }

    print(f"  Position sweep for: {prompt}")
    for pos in range(min(10, max_tokens)):
        r = position_results[pos]
        c = 'Y' if r['correct_first'] else 'N'
        print(f"    spike@{pos}: first='{r['first_token']}' correct={c}")

    # === Decay measurement ===
    print("\n[P15c] Measuring entropy decay after spike...")
    # Entropy difference between spike@0 and no-spike
    decay_curves = []
    for res in all_results:
        ent_spike = np.array(res['spike_0']['entropy'])
        ent_base = np.array(res['no_spike']['entropy'])
        min_len = min(len(ent_spike), len(ent_base))
        diff = ent_base[:min_len] - ent_spike[:min_len]  # positive = spike reduced entropy
        decay_curves.append(diff)

    # Average decay curve
    max_len = max(len(d) for d in decay_curves)
    padded = np.zeros((len(decay_curves), max_len))
    for i, d in enumerate(decay_curves):
        padded[i, :len(d)] = d
    mean_decay = padded.mean(axis=0)

    # Fit exponential decay
    valid_steps = min(10, len(mean_decay))
    if mean_decay[0] > 0:
        log_decay = np.log(np.abs(mean_decay[:valid_steps]) + 1e-12)
        slope, intercept = np.polyfit(range(valid_steps), log_decay, 1)
        half_life = -np.log(2) / slope if slope < 0 else float('inf')
        print(f"  Decay rate: {slope:.3f} per token")
        print(f"  Half-life: {half_life:.1f} tokens")
    else:
        half_life = 0
        slope = 0

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Entropy trajectories for first test case
    r0 = all_results[0]
    axes[0].plot(r0['no_spike']['entropy'][:12], 'r.-', label='No spike', linewidth=2)
    axes[0].plot(r0['spike_0']['entropy'][:12], 'g.-', label='Spike@0', linewidth=2)
    axes[0].plot(r0['spike_all']['entropy'][:12], 'b.-', label='Spike@all', linewidth=2)
    axes[0].set_xlabel('Token Position')
    axes[0].set_ylabel('Entropy')
    axes[0].set_title(f'Entropy: {test_cases[0][0][:25]}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Mean decay curve
    axes[1].plot(mean_decay[:12], 'mo-', linewidth=2, markersize=8)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Tokens After Spike')
    axes[1].set_ylabel('Entropy Reduction')
    axes[1].set_title(f'Spike Decay (half-life={half_life:.1f} tokens)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Spike position effectiveness
    positions = sorted(position_results.keys())[:12]
    correctness = [1 if position_results[p]['correct_first'] else 0 for p in positions]
    axes[2].bar(positions, correctness, color='teal', alpha=0.7)
    axes[2].set_xlabel('Spike Position')
    axes[2].set_ylabel('Correct First Token')
    axes[2].set_title('Spike Position vs Effectiveness')

    plt.suptitle('Phase 15: Spike Temporal Decay', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase15_temporal_decay.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 15, 'name': 'Spike Temporal Decay',
        'decay_rate': float(slope),
        'half_life_tokens': float(half_life),
        'spike_magnitude': spike_mag,
        'generation_examples': [{
            'prompt': r['prompt'], 'fact': r['fact'],
            'no_spike_text': ''.join(r['no_spike']['text'][:8]),
            'spike0_text': ''.join(r['spike_0']['text'][:8]),
            'spike_all_text': ''.join(r['spike_all']['text'][:8]),
        } for r in all_results],
    }
    with open(os.path.join(RESULTS_DIR, 'phase15_temporal_decay.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 15 RESULTS: Temporal Decay")
    print("=" * 70)
    print(f"  Decay rate: {slope:.3f}/token")
    print(f"  Half-life: {half_life:.1f} tokens")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
