# -*- coding: utf-8 -*-
"""
Phase 4: Retrocausal FSPO & Micro-Apoptosis
- Look-ahead decoding: simulate future tokens
- Detect entropy spikes (hallucination precursors)
- Propagate "pain gradient" backward to kill bad branches
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
    print("[P4] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def lookahead_entropy(model, input_ids, n_steps=5, n_branches=3):
    """Simulate n_steps into future, return entropy trajectory per branch."""
    branches = []
    for _ in range(n_branches):
        ids = input_ids.clone()
        entropies = []
        for step in range(n_steps):
            with torch.no_grad():
                out = model(ids)
            logits = out.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            h = float(-torch.sum(probs * torch.log(probs + 1e-12), dim=-1).cpu())
            entropies.append(h)
            # Sample next token
            next_tok = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_tok], dim=1)
        branches.append(entropies)
    return branches


def detect_spike(entropies, threshold=1.5):
    """Detect entropy spike (hallucination precursor)."""
    if len(entropies) < 2:
        return False, 0
    diffs = [entropies[i+1] - entropies[i] for i in range(len(entropies)-1)]
    max_spike = max(diffs) if diffs else 0
    return max_spike > threshold, max_spike


def retrocausal_decode(model, tok, prompt, n_lookahead=5, n_branches=5,
                       spike_threshold=1.5, max_tokens=30):
    """Generate with retrocausal apoptosis: kill branches that spike."""
    input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    generated_ids = input_ids.clone()
    apoptosis_count = 0
    token_entropies = []

    for t in range(max_tokens):
        with torch.no_grad():
            out = model(generated_ids)
        logits = out.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        current_h = float(-torch.sum(probs * torch.log(probs + 1e-12), dim=-1).cpu())
        token_entropies.append(current_h)

        # Get top-k candidates
        top_k = 10
        top_probs, top_ids = torch.topk(probs, top_k, dim=-1)

        # Lookahead for each candidate
        best_candidate = None
        best_score = float('inf')

        for i in range(min(top_k, 5)):
            candidate_ids = torch.cat([generated_ids, top_ids[:, i:i+1]], dim=1)
            branches = lookahead_entropy(model, candidate_ids,
                                         n_steps=n_lookahead, n_branches=2)
            # Average future entropy across branches
            mean_future_h = np.mean([np.mean(b) for b in branches])

            # Check for spikes
            has_spike = any(detect_spike(b, spike_threshold)[0] for b in branches)
            if has_spike:
                apoptosis_count += 1
                continue  # KILL this branch (apoptosis!)

            if mean_future_h < best_score:
                best_score = mean_future_h
                best_candidate = top_ids[:, i:i+1]

        if best_candidate is None:
            # All branches spiked - take argmax as fallback
            best_candidate = top_ids[:, 0:1]

        if best_candidate.item() == tok.eos_token_id:
            break
        generated_ids = torch.cat([generated_ids, best_candidate], dim=1)

    text = tok.decode(generated_ids[0], skip_special_tokens=True)
    return text, apoptosis_count, token_entropies


def main():
    print("=" * 70)
    print("  Phase 4: Retrocausal FSPO & Micro-Apoptosis")
    print("  Autonomous Neural Immunity System")
    print("=" * 70)

    model, tok = load_model()

    prompts = [
        ("The capital of Japan is", "Tokyo"),
        ("Water freezes at", "0"),
        ("The chemical formula for water is", "H2O"),
        ("Albert Einstein was born in", "1879"),
        ("The largest planet is", "Jupiter"),
    ]

    # === Standard vs Retrocausal generation ===
    print("\n[P4a] Comparing standard vs retrocausal decoding...")
    gen_results = []
    total_apoptosis = 0

    for prompt, expected in prompts:
        # Standard greedy
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        with torch.no_grad():
            out_std = model.generate(inp, max_new_tokens=20,
                                     pad_token_id=tok.eos_token_id)
        text_std = tok.decode(out_std[0], skip_special_tokens=True)

        # Retrocausal
        text_rc, apop, entropies = retrocausal_decode(
            model, tok, prompt, n_lookahead=3, n_branches=3,
            spike_threshold=1.0, max_tokens=20
        )
        total_apoptosis += apop

        fact_std = expected.lower() in text_std.lower()
        fact_rc = expected.lower() in text_rc.lower()

        gen_results.append({
            'prompt': prompt, 'expected': expected,
            'standard': text_std[:80], 'retrocausal': text_rc[:80],
            'factual_std': fact_std, 'factual_rc': fact_rc,
            'apoptosis': apop,
            'entropy_trajectory': entropies[:10],
        })

        s1 = "FACT" if fact_std else "HALL"
        s2 = "FACT" if fact_rc else "HALL"
        print(f"  [{s1}->{s2}] apop={apop} | {prompt}")
        print(f"    STD: {text_std[:60]}")
        print(f"    RC:  {text_rc[:60]}")

    # === Entropy landscape analysis ===
    print("\n[P4b] Entropy landscape analysis...")
    landscape_data = []
    test_prompt = "The theory of relativity was developed by"
    inp = tok(test_prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    branches = lookahead_entropy(model, inp, n_steps=10, n_branches=8)

    spikes_detected = 0
    for b in branches:
        has_s, mag = detect_spike(b, 1.0)
        if has_s:
            spikes_detected += 1
        landscape_data.append({'entropies': b, 'has_spike': has_s, 'spike_mag': mag})
    print(f"  Branches with entropy spikes: {spikes_detected}/{len(branches)}")

    # === Visualization ===
    print("\n[P4] Generating figures...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Entropy trajectories (branches)
    for i, b in enumerate(branches):
        color = 'red' if landscape_data[i]['has_spike'] else 'green'
        label = 'Spike (kill)' if i == 0 and landscape_data[i]['has_spike'] else \
                ('Safe' if i == 0 else None)
        axes[0].plot(b, color=color, alpha=0.5, linewidth=2, label=label)
    axes[0].set_xlabel('Lookahead Step')
    axes[0].set_ylabel('Entropy')
    axes[0].set_title('Future Entropy Branches')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Apoptosis counts
    apop_counts = [r['apoptosis'] for r in gen_results]
    axes[1].bar(range(len(apop_counts)), apop_counts, color='crimson', alpha=0.7)
    axes[1].set_xlabel('Prompt')
    axes[1].set_ylabel('Branches Killed')
    axes[1].set_title(f'Micro-Apoptosis (Total: {total_apoptosis})')

    # Plot 3: Token-level entropy for one example
    if gen_results and gen_results[0]['entropy_trajectory']:
        axes[2].plot(gen_results[0]['entropy_trajectory'], 'b.-', linewidth=2)
        axes[2].set_xlabel('Token Position')
        axes[2].set_ylabel('Entropy')
        axes[2].set_title('Retrocausal Token Entropy')
        axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 4: Retrocausal FSPO', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase4_retrocausal.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 4, 'name': 'Retrocausal FSPO & Micro-Apoptosis',
        'total_apoptosis': total_apoptosis,
        'spikes_detected': spikes_detected,
        'total_branches': len(branches),
        'generation_examples': gen_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase4_retrocausal.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 4 RESULTS: Retrocausal FSPO")
    print("=" * 70)
    print(f"  Total apoptosis events: {total_apoptosis}")
    print(f"  Spike detection rate: {spikes_detected}/{len(branches)}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
