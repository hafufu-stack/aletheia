# -*- coding: utf-8 -*-
"""
Phase 5: Spiking-FGA - Semantic Superconductivity
- Inject fact spikes into pre-softmax logits
- Measure temperature quenching effect
- Achieve deterministic output for grounded facts
"""
import os, json, gc
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
    print("[P5] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def spike_inject(logits, fact_token_ids, spike_magnitude=50.0):
    """Inject SNN-style spikes at fact token positions.
    logits shape: (1, vocab_size)"""
    spiked = logits.clone()
    for tid in fact_token_ids:
        if tid < spiked.shape[-1]:
            spiked[..., tid] += spike_magnitude
    return spiked


def effective_temperature(probs):
    """Compute effective temperature from probability distribution."""
    h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
    # T ~ entropy / log(vocab) normalized
    vocab_size = probs.shape[-1]
    t_eff = h / np.log(vocab_size)
    return t_eff, h


def main():
    print("=" * 70)
    print("  Phase 5: Spiking-FGA - Semantic Superconductivity")
    print("  Deterministic Truth via Fact Spike Injection")
    print("=" * 70)

    model, tok = load_model()

    # Fact-grounded QA pairs
    qa_pairs = [
        ("The capital of Japan is", " Tokyo", [11522]),  # "Tokyo" token
        ("Water freezes at", " 0", [657]),
        ("The chemical formula for water is", " H", [367]),
        ("The largest planet is", " Jupiter", [22721]),
        ("DNA stands for", " de", [390]),
    ]

    # Resolve actual token IDs
    resolved_pairs = []
    for prompt, answer, _ in qa_pairs:
        answer_ids = tok.encode(answer)
        resolved_pairs.append((prompt, answer, answer_ids))
        print(f"  '{answer.strip()}' -> token IDs: {answer_ids}")

    # === Sweep spike magnitudes ===
    print("\n[P5a] Sweeping spike magnitudes...")
    magnitudes = [0, 5, 10, 20, 50, 100, 200, 500]
    results_by_mag = {}

    for mag in magnitudes:
        temps = []
        entropies = []
        correct = 0

        for prompt, answer, fact_ids in resolved_pairs:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[:, -1, :]  # (1, vocab_size)

            # Inject spikes
            spiked_logits = spike_inject(logits, fact_ids, spike_magnitude=mag)
            probs = F.softmax(spiked_logits, dim=-1).squeeze(0)

            t_eff, h = effective_temperature(probs)
            temps.append(t_eff)
            entropies.append(h)

            # Check if top-1 matches fact
            top1 = torch.argmax(probs).item()
            if top1 in fact_ids:
                correct += 1

        results_by_mag[mag] = {
            'mean_temp': float(np.mean(temps)),
            'mean_entropy': float(np.mean(entropies)),
            'accuracy': correct / len(resolved_pairs),
        }
        print(f"  mag={mag:>4d}: T_eff={np.mean(temps):.6f}, "
              f"H={np.mean(entropies):.2f}, acc={correct}/{len(resolved_pairs)}")

    # === Phase transition detection ===
    print("\n[P5b] Detecting phase transition (superconductivity onset)...")
    # Find the magnitude where accuracy jumps to 100%
    transition_mag = None
    for mag in magnitudes:
        if results_by_mag[mag]['accuracy'] >= 1.0:
            transition_mag = mag
            break

    if transition_mag:
        print(f"  PHASE TRANSITION at spike magnitude = {transition_mag}")
        print(f"  T_eff at transition: {results_by_mag[transition_mag]['mean_temp']:.8f}")
    else:
        print("  No complete phase transition detected")

    # === Multi-token grounded generation ===
    print("\n[P5c] Multi-token grounded generation...")
    test_prompt = "The capital of France is"
    fact_text = " Paris"
    fact_tokens = tok.encode(fact_text)

    inp_ids = tok(test_prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    gen_tokens = []

    for step in range(10):
        with torch.no_grad():
            out = model(inp_ids)
        logits = out.logits[:, -1, :]  # (1, vocab_size)

        if step < len(fact_tokens):
            spiked = spike_inject(logits, [fact_tokens[step]], spike_magnitude=100)
            probs = F.softmax(spiked, dim=-1).squeeze(0)
        else:
            probs = F.softmax(logits, dim=-1).squeeze(0)

        next_tok = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
        gen_tokens.append(next_tok.item())
        inp_ids = torch.cat([inp_ids, next_tok], dim=1)

    gen_text = tok.decode(gen_tokens)
    print(f"  Prompt: {test_prompt}")
    print(f"  Generated: {gen_text}")
    print(f"  Fact injected: {fact_text.strip()}")

    # === Visualization ===
    print("\n[P5] Generating figures...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    mags = sorted(results_by_mag.keys())
    temps_plot = [results_by_mag[m]['mean_temp'] for m in mags]
    accs_plot = [results_by_mag[m]['accuracy'] * 100 for m in mags]
    ents_plot = [results_by_mag[m]['mean_entropy'] for m in mags]

    # Plot 1: Effective temperature
    axes[0].semilogy(mags, temps_plot, 'b.-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Spike Magnitude')
    axes[0].set_ylabel('Effective Temperature')
    axes[0].set_title('Temperature Quenching')
    if transition_mag:
        axes[0].axvline(x=transition_mag, color='r', linestyle='--',
                        label=f'Transition @ {transition_mag}')
        axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[1].plot(mags, accs_plot, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Spike Magnitude')
    axes[1].set_ylabel('Factual Accuracy (%)')
    axes[1].set_title('Deterministic Truth Emergence')
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Entropy collapse
    axes[2].plot(mags, ents_plot, 'r.-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Spike Magnitude')
    axes[2].set_ylabel('Output Entropy')
    axes[2].set_title('Entropy Collapse -> Superconductivity')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 5: Spiking-FGA (Semantic Superconductivity)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase5_spiking_fga.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 5, 'name': 'Spiking-FGA Semantic Superconductivity',
        'spike_sweep': {str(k): v for k, v in results_by_mag.items()},
        'transition_magnitude': transition_mag,
        'grounded_generation': gen_text,
    }
    with open(os.path.join(RESULTS_DIR, 'phase5_spiking_fga.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 5 RESULTS: Spiking-FGA")
    print("=" * 70)
    print(f"  Phase transition magnitude: {transition_mag}")
    if transition_mag:
        print(f"  T_eff at transition: {results_by_mag[transition_mag]['mean_temp']:.8f}")
    print(f"  Grounded output: {gen_text[:50]}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
