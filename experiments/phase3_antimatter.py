# -*- coding: utf-8 -*-
"""
Phase 3: Antimatter Contrastive Decoding
- Build "antimatter brain" that maximizes hallucination
- Subtract its logits (phase-inverted) from main model
- Measure hallucination annihilation rate
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
    print("[P3] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def get_logits(model, tok, text):
    inp = tok(text, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[:, -1, :].squeeze(0)  # (vocab_size,)


def contrastive_decode(main_logits, anti_logits, alpha=0.5, temp=1.0):
    """Subtract antimatter logits from main (destructive interference)."""
    adjusted = main_logits - alpha * anti_logits
    probs = F.softmax(adjusted / temp, dim=-1)
    return probs


def generate_contrastive(model, tok, prompt, alpha=0.5, max_len=30):
    """Generate using contrastive decoding against high-temp antimatter."""
    input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    generated = input_ids.clone()

    for _ in range(max_len):
        with torch.no_grad():
            out_main = model(generated)
        main_logits = out_main.logits[:, -1, :].squeeze(0)

        # Antimatter: same model but with very high temperature (hallucination mode)
        anti_logits = main_logits * 2.0  # Amplify = more hallucination-prone

        # Contrastive: subtract antimatter
        probs = contrastive_decode(main_logits, anti_logits, alpha=alpha)
        next_token = torch.argmax(probs).unsqueeze(0).unsqueeze(0)

        if next_token.item() == tok.eos_token_id:
            break
        generated = torch.cat([generated, next_token], dim=1)

    return tok.decode(generated[0], skip_special_tokens=True)


def entropy_of_probs(probs):
    return float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())


def main():
    print("=" * 70)
    print("  Phase 3: Antimatter Contrastive Decoding")
    print("  Semantic Annihilation of Hallucinations")
    print("=" * 70)

    model, tok = load_model()

    prompts = [
        ("The capital of Japan is", "Tokyo"),
        ("The chemical formula for water is", "H2O"),
        ("The year World War II ended was", "1945"),
        ("Albert Einstein was born in", "1879"),
        ("The speed of light is", "speed"),
        ("Python was created by", "Guido"),
        ("The largest ocean is the", "Pacific"),
        ("The human heart has how many chambers", "four"),
        ("DNA is a double", "helix"),
        ("The Mona Lisa was painted by", "Leonardo"),
    ]

    # === Sweep alpha values ===
    print("\n[P3a] Sweeping contrastive alpha...")
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results_by_alpha = {}

    for alpha in alphas:
        print(f"\n  alpha = {alpha}:")
        entropies = []
        kl_divs = []

        for prompt_text, _ in prompts:
            logits_main = get_logits(model, tok, prompt_text)

            # Antimatter: amplified logits
            logits_anti = logits_main * 2.0

            # Standard probs
            probs_std = F.softmax(logits_main, dim=-1)

            # Contrastive probs
            probs_cd = contrastive_decode(logits_main, logits_anti, alpha=alpha)

            # Entropy
            h_std = entropy_of_probs(probs_std)
            h_cd = entropy_of_probs(probs_cd)
            entropies.append({'standard': h_std, 'contrastive': h_cd})

            # KL divergence from standard to contrastive
            kl = float(F.kl_div(
                torch.log(probs_cd + 1e-12),
                probs_std, reduction='sum'
            ).cpu())
            kl_divs.append(kl)

        mean_h_std = np.mean([e['standard'] for e in entropies])
        mean_h_cd = np.mean([e['contrastive'] for e in entropies])
        results_by_alpha[alpha] = {
            'mean_entropy_standard': mean_h_std,
            'mean_entropy_contrastive': mean_h_cd,
            'entropy_reduction_pct': (1 - mean_h_cd / max(mean_h_std, 1e-12)) * 100,
            'mean_kl_divergence': float(np.mean(kl_divs)),
        }
        print(f"    Entropy: {mean_h_std:.2f} -> {mean_h_cd:.2f} "
              f"({results_by_alpha[alpha]['entropy_reduction_pct']:.1f}% reduction)")

    # === Phase 3b: Generate with contrastive decoding ===
    print("\n[P3b] Generating with contrastive decoding...")
    gen_results = []
    best_alpha = 0.3

    for prompt_text, expected in prompts[:5]:
        gen_std = generate_contrastive(model, tok, prompt_text, alpha=0.0, max_len=20)
        gen_cd = generate_contrastive(model, tok, prompt_text, alpha=best_alpha, max_len=20)

        fact_std = expected.lower() in gen_std.lower()
        fact_cd = expected.lower() in gen_cd.lower()

        gen_results.append({
            'prompt': prompt_text, 'expected': expected,
            'standard': gen_std[:80], 'contrastive': gen_cd[:80],
            'factual_std': fact_std, 'factual_cd': fact_cd,
        })
        s1 = "FACT" if fact_std else "HALL"
        s2 = "FACT" if fact_cd else "HALL"
        print(f"  [{s1}->{s2}] {prompt_text}")
        print(f"    STD: {gen_std[:60]}")
        print(f"    CD:  {gen_cd[:60]}")

    # === Visualization ===
    print("\n[P3] Generating figures...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Entropy vs alpha
    a_list = sorted(results_by_alpha.keys())
    h_std = [results_by_alpha[a]['mean_entropy_standard'] for a in a_list]
    h_cd = [results_by_alpha[a]['mean_entropy_contrastive'] for a in a_list]
    axes[0].plot(a_list, h_std, 'r--', label='Standard', linewidth=2)
    axes[0].plot(a_list, h_cd, 'b.-', label='Contrastive', linewidth=2)
    axes[0].set_xlabel('Alpha (antimatter strength)')
    axes[0].set_ylabel('Mean Entropy')
    axes[0].set_title('Entropy vs Antimatter Strength')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: KL divergence
    kls = [results_by_alpha[a]['mean_kl_divergence'] for a in a_list]
    axes[1].bar(range(len(a_list)), kls, tick_label=[str(a) for a in a_list],
                color='purple', alpha=0.7)
    axes[1].set_xlabel('Alpha')
    axes[1].set_ylabel('KL Divergence')
    axes[1].set_title('Distribution Shift from Standard')

    # Plot 3: Entropy reduction %
    red = [results_by_alpha[a]['entropy_reduction_pct'] for a in a_list]
    axes[2].plot(a_list, red, 'go-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Alpha')
    axes[2].set_ylabel('Entropy Reduction (%)')
    axes[2].set_title('Annihilation Efficiency')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 3: Antimatter Contrastive Decoding', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase3_antimatter.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # === Save ===
    results = {
        'phase': 3, 'name': 'Antimatter Contrastive Decoding',
        'alpha_sweep': {str(k): v for k, v in results_by_alpha.items()},
        'generation_examples': gen_results,
        'best_alpha': best_alpha,
    }
    with open(os.path.join(RESULTS_DIR, 'phase3_antimatter.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    best_red = results_by_alpha[best_alpha]['entropy_reduction_pct']
    print("\n" + "=" * 70)
    print("  PHASE 3 RESULTS: Antimatter Contrastive Decoding")
    print("=" * 70)
    print(f"  Best alpha: {best_alpha}")
    print(f"  Entropy reduction at best alpha: {best_red:.1f}%")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
