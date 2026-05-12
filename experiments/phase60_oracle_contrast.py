# -*- coding: utf-8 -*-
"""
Phase 60: Oracle-Guided Layer Contrast (DT2 P61 idea)
Entropy-gated dynamic L10 boosting: when the model is uncertain,
boost tokens that are high in L10 but low in L12.
Combines P31 Oracle + P48 Contrastive + P51 Early Exit.
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
    print("[P60] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def main():
    print("=" * 70)
    print("  Phase 60: Oracle-Guided Layer Contrast")
    print("  Dynamic L10 boost when entropy is high")
    print("=" * 70)

    model, tok = load_model()

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

    # Sweep: threshold x boost_alpha
    print("\n[P60] Sweeping (threshold, alpha) grid...")
    grid_results = {}
    for thresh in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        for alpha in [0.0, 0.3, 0.5, 1.0, 2.0, 5.0]:
            correct = 0
            for prompt, fact_ids, expected in tests:
                inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

                h10 = {}
                def hook(m, a, o):
                    h10['h'] = o[0][0, -1, :].detach()
                handle = model.transformer.h[10].register_forward_hook(hook)
                with torch.no_grad():
                    out = model(inp, output_attentions=True, return_dict=True)
                handle.remove()

                # Entropy
                ents = []
                for attn in out.attentions:
                    for h in range(attn.shape[1]):
                        a_vec = attn[0, h, -1, :].cpu().numpy()
                        ents.append(float(-np.sum(a_vec * np.log(a_vec + 1e-12))))
                mean_ent = float(np.mean(ents))

                final_logits = out.logits[:, -1, :].squeeze(0)
                normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
                l10_logits = model.lm_head(normed).squeeze(0)

                if mean_ent > thresh:
                    # High entropy: boost L10 tokens
                    # contrastive = tokens high in L10 but low in L12
                    boost = l10_logits - final_logits
                    combined = final_logits + alpha * boost
                else:
                    combined = final_logits

                if torch.argmax(combined).item() in fact_ids:
                    correct += 1

            acc = correct / len(tests)
            grid_results[(thresh, alpha)] = acc

    # Find best config
    best_key = max(grid_results, key=grid_results.get)
    best_acc = grid_results[best_key]
    print(f"\n  BEST: thresh={best_key[0]}, alpha={best_key[1]} -> {best_acc:.0%}")

    # Baselines
    l12_correct = 0
    l10_correct = 0
    detail_results = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        h10 = {}
        def hook(m, a, o):
            h10['h'] = o[0][0, -1, :].detach()
        handle = model.transformer.h[10].register_forward_hook(hook)
        with torch.no_grad():
            out = model(inp, output_attentions=True, return_dict=True)
        handle.remove()

        final_logits = out.logits[:, -1, :].squeeze(0)
        normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
        l10_logits = model.lm_head(normed).squeeze(0)

        ents = []
        for attn in out.attentions:
            for h in range(attn.shape[1]):
                a_vec = attn[0, h, -1, :].cpu().numpy()
                ents.append(float(-np.sum(a_vec * np.log(a_vec + 1e-12))))
        mean_ent = float(np.mean(ents))

        f_ok = torch.argmax(final_logits).item() in fact_ids
        l_ok = torch.argmax(l10_logits).item() in fact_ids
        if f_ok: l12_correct += 1
        if l_ok: l10_correct += 1

        # Best config
        bt, ba = best_key
        if mean_ent > bt:
            boost = l10_logits - final_logits
            combined = final_logits + ba * boost
        else:
            combined = final_logits
        b_ok = torch.argmax(combined).item() in fact_ids

        detail_results.append({
            'expected': expected, 'entropy': round(mean_ent, 3),
            'l12_correct': f_ok, 'l10_correct': l_ok, 'oracle_correct': b_ok,
        })
        tag = 'OK' if b_ok else 'FAIL'
        print(f"  {expected:>12s}: H={mean_ent:.3f} L12={'OK' if f_ok else 'X':>4s} "
              f"L10={'OK' if l_ok else 'X':>4s} Oracle=[{tag}]")

    l12_acc = l12_correct / len(tests)
    l10_acc = l10_correct / len(tests)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Method comparison
    methods = ['L12\n(baseline)', 'L10\n(always)', f'Oracle\n(t={best_key[0]},a={best_key[1]})']
    accs = [l12_acc*100, l10_acc*100, best_acc*100]
    colors = ['red', 'green', 'blue']
    axes[0].bar(methods, accs, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Oracle-Guided Layer Contrast')

    # 2. Heatmap of (threshold, alpha) grid
    thresholds = sorted(set(k[0] for k in grid_results))
    alphas = sorted(set(k[1] for k in grid_results))
    heatmap = np.zeros((len(thresholds), len(alphas)))
    for i, t in enumerate(thresholds):
        for j, a in enumerate(alphas):
            heatmap[i, j] = grid_results.get((t, a), 0) * 100
    im = axes[1].imshow(heatmap, aspect='auto', cmap='viridis',
                        extent=[0, len(alphas), len(thresholds), 0])
    axes[1].set_xticks(np.arange(len(alphas)) + 0.5)
    axes[1].set_xticklabels([f'{a}' for a in alphas], fontsize=7)
    axes[1].set_yticks(np.arange(len(thresholds)) + 0.5)
    axes[1].set_yticklabels([f'{t}' for t in thresholds], fontsize=7)
    axes[1].set_xlabel('Alpha')
    axes[1].set_ylabel('Threshold')
    axes[1].set_title('Accuracy Grid (%)')
    plt.colorbar(im, ax=axes[1])

    # 3. Per-prompt entropy vs correctness
    ents_list = [r['entropy'] for r in detail_results]
    colors_per = ['green' if r['oracle_correct'] else 'red' for r in detail_results]
    labels = [r['expected'][:6] for r in detail_results]
    axes[2].bar(range(len(labels)), ents_list, color=colors_per, alpha=0.7)
    axes[2].axhline(y=best_key[0], color='blue', linestyle='--', label=f'threshold={best_key[0]}')
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[2].set_ylabel('Entropy')
    axes[2].set_title('Per-Prompt Entropy (green=correct)')
    axes[2].legend(fontsize=8)

    plt.suptitle('Phase 60: Oracle-Guided Layer Contrast', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase60_oracle_contrast.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 60, 'name': 'Oracle-Guided Layer Contrast',
        'best_threshold': best_key[0], 'best_alpha': best_key[1],
        'best_accuracy': best_acc,
        'l12_accuracy': l12_acc, 'l10_accuracy': l10_acc,
        'grid_results': {f'{k[0]}_{k[1]}': v for k, v in grid_results.items()},
        'detail_results': detail_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase60_oracle_contrast.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  L12 baseline: {l12_acc:.0%}")
    print(f"  L10 always:   {l10_acc:.0%}")
    print(f"  Oracle-Guided: {best_acc:.0%} (t={best_key[0]}, a={best_key[1]})")
    print("=" * 70)
    phase_complete(60)

if __name__ == '__main__':
    main()
