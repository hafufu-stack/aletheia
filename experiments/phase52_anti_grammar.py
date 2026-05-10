# -*- coding: utf-8 -*-
"""
Phase 52: Anti-Grammar Contrastive Decoding
P48 failed because we subtracted layers blindly. Now we know:
L10 = facts, L12 = grammar. Penalize tokens that GAINED probability
from L10->L12 (= grammar noise injected by final layers).
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
    print("[P52] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_layer_logits(model, tok, prompt, layer_idx):
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    hidden = {}
    def hook(module, args, output):
        hidden['h'] = output[0][0, -1, :].detach()
    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    normed = model.transformer.ln_f(hidden['h'].unsqueeze(0))
    ll = model.lm_head(normed).squeeze(0)
    final = out.logits[:, -1, :].squeeze(0)
    return ll, final

def main():
    print("=" * 70)
    print("  Phase 52: Anti-Grammar Contrastive Decoding")
    print("  Penalize tokens boosted by grammar layers (L10->L12)")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("DNA stands for", [390], "de"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The speed of light is approximately", [22626], "299"),
        ("The Earth orbits the", [4252], "Sun"),
        ("The boiling point of water is", [1802], "100"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("Oxygen has the atomic number", [807], "8"),
    ]

    # === P52a: Grammar noise analysis ===
    print("\n[P52a] Grammar noise analysis: what does L11-L12 add?")
    noise_analysis = []
    for prompt, fact_ids, expected in tests:
        l10, l12 = get_layer_logits(model, tok, prompt, 10)

        # Tokens whose probability INCREASED from L10->L12
        l10_probs = F.softmax(l10, dim=-1)
        l12_probs = F.softmax(l12, dim=-1)
        grammar_boost = l12_probs - l10_probs

        # Top grammar-boosted tokens
        top_boost = torch.topk(grammar_boost, 5)
        boosted_tokens = [tok.decode([t.item()]).encode('ascii','replace').decode().strip()
                          for t in top_boost.indices]

        # Fact token delta
        fact_delta = float((grammar_boost[fact_ids[0]]).cpu())

        noise_analysis.append({
            'expected': expected,
            'grammar_boosted': boosted_tokens,
            'fact_delta': round(fact_delta, 6),
        })
        direction = 'SUPPRESSED' if fact_delta < 0 else 'boosted'
        print(f"  {expected:>12s}: fact {direction} ({fact_delta:+.6f})")
        print(f"    Grammar-boosted: {boosted_tokens}")

    # === P52b: Anti-grammar decoding sweep ===
    print("\n[P52b] Anti-grammar decoding: L12 - alpha*max(0, L12-L10)...")
    sweep_results = {}
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        correct = 0
        ranks = []
        for prompt, fact_ids, expected in tests:
            l10, l12 = get_layer_logits(model, tok, prompt, 10)

            # Anti-grammar: penalize tokens that were boosted by grammar layers
            grammar_noise = torch.clamp(l12 - l10, min=0)  # Only positive boosts
            anti_grammar = l12 - alpha * grammar_noise

            rank = int((anti_grammar.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            ranks.append(rank)
            if torch.argmax(anti_grammar).item() in fact_ids:
                correct += 1

        sweep_results[alpha] = {
            'accuracy': correct / len(tests),
            'median_rank': float(np.median(ranks)),
            'mean_rank': float(np.mean(ranks)),
        }
        print(f"  alpha={alpha:.1f}: {correct}/{len(tests)} = {correct/len(tests):.0%} "
              f"med_rank={np.median(ranks):.0f}")

    # === P52c: Alternative: L10 + beta * L12 (additive blend) ===
    print("\n[P52c] Additive blend: L10 + beta * L12...")
    blend_results = {}
    for beta in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
        correct = 0
        ranks = []
        for prompt, fact_ids, expected in tests:
            l10, l12 = get_layer_logits(model, tok, prompt, 10)
            blended = l10 + beta * l12
            rank = int((blended.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            ranks.append(rank)
            if torch.argmax(blended).item() in fact_ids:
                correct += 1
        blend_results[beta] = {
            'accuracy': correct / len(tests),
            'median_rank': float(np.median(ranks)),
        }
        print(f"  beta={beta:.1f}: {correct}/{len(tests)} = {correct/len(tests):.0%} "
              f"med_rank={np.median(ranks):.0f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Grammar noise direction for facts
    labels = [n['expected'][:6] for n in noise_analysis]
    deltas = [n['fact_delta'] for n in noise_analysis]
    colors = ['red' if d < 0 else 'green' for d in deltas]
    axes[0].bar(labels, deltas, color=colors, alpha=0.7)
    axes[0].set_ylabel('Prob delta (L12-L10)')
    axes[0].set_title('Grammar Layer Effect on Facts')
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].tick_params(axis='x', rotation=45, labelsize=7)

    # Anti-grammar sweep
    alphas = sorted(sweep_results.keys())
    s_accs = [sweep_results[a]['accuracy']*100 for a in alphas]
    s_ranks = [sweep_results[a]['median_rank'] for a in alphas]
    axes[1].plot(alphas, s_accs, 'g.-', linewidth=2, label='Accuracy')
    ax1r = axes[1].twinx()
    ax1r.plot(alphas, s_ranks, 'r.--', linewidth=2, label='Med Rank')
    axes[1].set_xlabel('Alpha')
    axes[1].set_ylabel('Accuracy (%)')
    ax1r.set_ylabel('Median Rank')
    axes[1].set_title('Anti-Grammar Decoding')
    axes[1].legend(loc='upper left', fontsize=8)

    # Blend
    betas = sorted(blend_results.keys())
    b_accs = [blend_results[b]['accuracy']*100 for b in betas]
    axes[2].plot(betas, b_accs, 'b.-', linewidth=2, markersize=10)
    axes[2].set_xlabel('Beta (L12 weight in L10+beta*L12)')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('L10+L12 Blend')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 52: Anti-Grammar Contrastive Decoding', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase52_anti_grammar.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 52, 'name': 'Anti-Grammar Contrastive Decoding',
        'anti_grammar_sweep': {str(k): v for k, v in sweep_results.items()},
        'blend_sweep': {str(k): v for k, v in blend_results.items()},
        'noise_analysis': noise_analysis,
    }
    with open(os.path.join(RESULTS_DIR, 'phase52_anti_grammar.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    best_alpha = max(sweep_results, key=lambda a: sweep_results[a]['accuracy'])
    best_beta = max(blend_results, key=lambda b: blend_results[b]['accuracy'])
    print("\n" + "=" * 70)
    print("  PHASE 52 RESULTS")
    print("=" * 70)
    print(f"  Best anti-grammar: alpha={best_alpha} -> {sweep_results[best_alpha]['accuracy']:.0%}")
    print(f"  Best blend: beta={best_beta} -> {blend_results[best_beta]['accuracy']:.0%}")
    n_suppressed = sum(1 for n in noise_analysis if n['fact_delta'] < 0)
    print(f"  Facts suppressed by grammar: {n_suppressed}/{len(tests)}")
    print("=" * 70)
    phase_complete(52)

if __name__ == '__main__':
    main()
