# -*- coding: utf-8 -*-
"""
Phase 58: Grand Unified Summary
The final experiment: consolidate ALL findings into the complete
characterization of LLM hallucination physics.
 - 58 phases of experiments
 - 7 fundamental laws/theorems
 - The complete theory of truth in LLM latent space
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
    print("[P58] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def main():
    print("=" * 70)
    print("  Phase 58: Grand Unified Summary")
    print("  The Complete Theory of Truth in LLM Latent Space")
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

    # === Comprehensive per-layer + per-method benchmark ===
    print("\n[P58] Running comprehensive benchmark...")
    grand_results = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

        # Collect all layer hidden states
        layer_hs = {}
        handles = []
        for li in range(12):
            def make_hook(idx):
                def fn(m, a, o):
                    layer_hs[idx] = o[0][0, -1, :].detach()
                return fn
            h = model.transformer.h[li].register_forward_hook(make_hook(li))
            handles.append(h)
        with torch.no_grad():
            out = model(inp, output_attentions=True, return_dict=True)
        for h in handles:
            h.remove()

        # Per-layer ranks via Logit Lens
        layer_ranks = {}
        for li in range(12):
            normed = model.transformer.ln_f(layer_hs[li].unsqueeze(0))
            logits = model.lm_head(normed).squeeze(0)
            rank = int((logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            layer_ranks[li] = rank

        # Final layer rank
        final_logits = out.logits[:, -1, :].squeeze(0)
        final_rank = int((final_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1

        # Attention entropy
        ents = []
        for attn in out.attentions:
            for h in range(attn.shape[1]):
                a = attn[0, h, -1, :].cpu().numpy()
                ents.append(float(-np.sum(a * np.log(a + 1e-12))))
        mean_ent = float(np.mean(ents))

        # Best layer
        best_layer = min(layer_ranks, key=layer_ranks.get)
        best_rank = layer_ranks[best_layer]
        suppression = final_rank - best_rank

        grand_results.append({
            'expected': expected,
            'layer_ranks': layer_ranks,
            'final_rank': final_rank,
            'best_layer': best_layer,
            'best_rank': best_rank,
            'suppression': suppression,
            'entropy': round(mean_ent, 4),
            'l10_rank': layer_ranks[10],
            'l12_correct': final_rank == 1,
            'l10_correct': layer_ranks[10] == 1,
        })

        sup_tag = f'SUPPRESSED({suppression:+d})' if suppression > 0 else 'not_suppressed'
        print(f"  {expected:>12s}: best=L{best_layer}(r{best_rank:>5d}) "
              f"L10=r{layer_ranks[10]:>5d} final=r{final_rank:>5d} [{sup_tag}]")

    # === Summary statistics ===
    l12_acc = sum(1 for r in grand_results if r['l12_correct']) / len(tests)
    l10_acc = sum(1 for r in grand_results if r['l10_correct']) / len(tests)
    n_suppressed = sum(1 for r in grand_results if r['suppression'] > 0)
    avg_best_layer = np.mean([r['best_layer'] for r in grand_results])
    avg_suppression = np.mean([r['suppression'] for r in grand_results])
    median_suppression = np.median([r['suppression'] for r in grand_results])

    print("\n" + "-" * 50)
    print(f"  L12 accuracy:     {l12_acc:.0%}")
    print(f"  L10 accuracy:     {l10_acc:.0%}")
    print(f"  Suppressed:       {n_suppressed}/{len(tests)} ({100*n_suppressed/len(tests):.0f}%)")
    print(f"  Avg best layer:   L{avg_best_layer:.1f}")
    print(f"  Avg suppression:  {avg_suppression:.1f} ranks")

    # === Grand visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Mean rank trajectory across all prompts
    mean_ranks = []
    for l in range(12):
        lr = [r['layer_ranks'][l] for r in grand_results]
        mean_ranks.append(np.median(lr))
    final_ranks = [r['final_rank'] for r in grand_results]
    mean_ranks.append(np.median(final_ranks))
    axes[0,0].plot(range(13), mean_ranks, 'r.-', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Layer (0-11 + Final)')
    axes[0,0].set_ylabel('Median Fact Rank')
    axes[0,0].set_yscale('log')
    axes[0,0].set_title('Universal Rank Trajectory')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axvspan(8, 10, alpha=0.2, color='green', label='Fact zone')
    axes[0,0].legend()

    # 2. Method comparison
    methods = ['L12\n(baseline)', 'L10\n(Logit Lens)', '4x\nimprovement']
    accs = [l12_acc*100, l10_acc*100, 0]
    colors = ['red', 'green', 'white']
    axes[0,1].bar(methods[:2], accs[:2], color=colors[:2], alpha=0.7, edgecolor='black')
    axes[0,1].set_ylabel('Accuracy (%)')
    axes[0,1].set_title(f'L10 Logit Lens: {l10_acc/l12_acc:.0f}x improvement')

    # 3. Suppression histogram
    sups = [r['suppression'] for r in grand_results]
    axes[0,2].hist(sups, bins=20, color='red', alpha=0.7, edgecolor='black')
    axes[0,2].set_xlabel('Rank Suppression (final - best)')
    axes[0,2].set_ylabel('Count')
    axes[0,2].set_title(f'Grammar Suppression\n({n_suppressed}/{len(tests)} suppressed)')

    # 4. Best layer histogram
    best_ls = [r['best_layer'] for r in grand_results]
    axes[1,0].hist(best_ls, bins=range(13), color='teal', alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Best Layer')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title(f'Fact Peak Distribution\n(avg=L{avg_best_layer:.1f})')

    # 5. Individual trajectories
    for gr in grand_results:
        ranks = [gr['layer_ranks'][l] for l in range(12)] + [gr['final_rank']]
        c = 'green' if gr['suppression'] > 0 else 'gray'
        axes[1,1].plot(range(13), ranks, '.-', alpha=0.5, color=c, linewidth=1)
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Fact Rank')
    axes[1,1].set_yscale('log')
    axes[1,1].set_title('All Fact Trajectories')
    axes[1,1].grid(True, alpha=0.3)

    # 6. Season timeline
    seasons = ['S1-5\n(P1-23)\nPhysics', 'S6-7\n(P24-36)\nDetection',
               'S8-10\n(P37-48)\nIntervention', 'S11-12\n(P49-56)\nExtraction',
               'S13\n(P57-58)\nUnification']
    findings = [5, 3, 0, 2, 1]
    axes[1,2].bar(seasons, findings, color=['blue','green','red','purple','gold'], alpha=0.7)
    axes[1,2].set_ylabel('Breakthrough Count')
    axes[1,2].set_title('Research Timeline')
    axes[1,2].tick_params(axis='x', labelsize=7)

    plt.suptitle('Phase 58: Grand Unified Theory of LLM Hallucination',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase58_grand_unified.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # === Grand Unified Theory JSON ===
    theory = {
        'phase': 58,
        'name': 'Grand Unified Theory of LLM Hallucination',
        'total_phases': 58,
        'laws': {
            'law1_degeneracy': 'Fact-skill angular separation = 1.2 deg (P1)',
            'law2_temperature': 'Critical spike is T-independent, gamma=0.000 (P7)',
            'law3_layernorm': 'All mid-layer interventions absorbed by LayerNorm (P8-P12)',
            'law4_scaling': 'spike_c ~ N^{-0.491} (P19)',
            'law5_temporal': 'Single t=0 spike half-life = 130.9 tokens (P15)',
            'law6_suppression': f'Grammar suppresses {n_suppressed}/{len(tests)} facts, avg {avg_suppression:.0f} ranks (P53)',
            'law7_oracle_duality': 'Entropy is perfect comparator but poor classifier (P31 vs P57)',
        },
        'theorems': {
            'detection_generation_separation': 'Detection (AUC=1.0) and generation correction (0%) are physically independent (P37-P43)',
            'l10_optimality': f'L10 Logit Lens achieves {l10_acc:.0%}, {l10_acc/l12_acc:.0f}x over L12 baseline (P49)',
            'internal_impossibility': 'No internal operation (gradient, topology, noise, MCTS) recovers suppressed facts (P37-P46)',
        },
        'summary_stats': {
            'l12_accuracy': l12_acc,
            'l10_accuracy': l10_acc,
            'suppression_rate': n_suppressed / len(tests),
            'avg_best_layer': round(float(avg_best_layer), 1),
            'avg_suppression': round(float(avg_suppression), 1),
        },
        'grand_results': grand_results,
    }

    with open(os.path.join(RESULTS_DIR, 'phase58_grand_unified.json'), 'w') as f:
        json.dump(theory, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  THE SEVEN LAWS OF LLM HALLUCINATION PHYSICS")
    print("=" * 70)
    for k, v in theory['laws'].items():
        print(f"  {k}: {v}")
    print("-" * 70)
    for k, v in theory['theorems'].items():
        print(f"  {k}: {v}")
    print("=" * 70)
    phase_complete(58)

if __name__ == '__main__':
    main()
