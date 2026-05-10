# -*- coding: utf-8 -*-
"""
Phase 32: Holographic Truth Salvage
P24's revenge: Extract truth from the FULL fact clique via SVD, not a single head.
Project PC1 of the fact-clique activation matrix through LM head.
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

# Fact heads from P14 (top heads whose ablation hurts factual accuracy)
FACT_HEADS = [(2,1),(0,3),(1,5),(3,7),(4,2),(6,8),(8,3),(10,1),(11,5)]

def load_model():
    print("[P32] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def main():
    print("=" * 70)
    print("  Phase 32: Holographic Truth Salvage")
    print("  SVD of fact-clique activations -> LM head projection")
    print("=" * 70)

    model, tok = load_model()
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    lm_head = model.lm_head

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("DNA stands for", [390], "de"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The speed of light is approximately", [22626], "299"),
    ]

    # === P32a: Extract per-head hidden states ===
    print("\n[P32a] Extracting fact-clique activation matrix...")
    results_list = []

    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)

        # Collect hidden states from each layer
        layer_outputs = {}
        handles = []
        for layer_idx in range(12):
            def make_hook(li):
                def hook_fn(module, args, output):
                    hs = output[0]
                    if hs.dim() == 3:
                        layer_outputs[li] = hs[0, -1, :].detach()
                    else:
                        layer_outputs[li] = hs[-1, :].detach()
                return hook_fn
            h = model.transformer.h[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(h)

        with torch.no_grad():
            out = model(**inp)
        for h in handles:
            h.remove()

        baseline_logits = out.logits[:, -1, :].squeeze(0)
        baseline_rank = int((baseline_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1

        # Build activation matrix: rows = fact heads, cols = d_head
        act_matrix = []
        for (li, hi) in FACT_HEADS:
            if li in layer_outputs:
                h_vec = layer_outputs[li][hi * d_head : (hi + 1) * d_head]
                act_matrix.append(h_vec.cpu().numpy())
        act_matrix = np.array(act_matrix)  # (n_fact_heads, d_head)

        # SVD
        U, S, Vt = np.linalg.svd(act_matrix, full_matrices=False)

        # PC1: first right singular vector (d_head)
        pc1 = torch.tensor(Vt[0], dtype=torch.float32, device=DEVICE)

        # Project PC1 through LM head (zero-pad to full hidden dim)
        # Try each fact head position as the embedding slot
        best_rank = 99999
        best_method = ""
        for slot_head in range(n_heads):
            full_vec = torch.zeros(model.config.n_embd, device=DEVICE)
            full_vec[slot_head * d_head : (slot_head + 1) * d_head] = pc1 * S[0]
            pc1_logits = lm_head(full_vec.unsqueeze(0)).squeeze(0)
            fr = int((pc1_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            if fr < best_rank:
                best_rank = fr
                best_method = f"slot_h{slot_head}"

        # Also try: full hidden dim = sum of all fact-head contributions
        full_holo = torch.zeros(model.config.n_embd, device=DEVICE)
        for i, (li, hi) in enumerate(FACT_HEADS):
            if li in layer_outputs:
                contribution = torch.tensor(U[i, 0] * S[0] * Vt[0], dtype=torch.float32, device=DEVICE)
                full_holo[hi * d_head : (hi + 1) * d_head] += contribution
        holo_logits = lm_head(full_holo.unsqueeze(0)).squeeze(0)
        holo_rank = int((holo_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
        holo_top5 = [tok.decode([t]).encode('ascii','replace').decode().strip()
                     for t in holo_logits.argsort(descending=True)[:5]]

        # Combined: baseline + holographic boost
        for scale in [0.1, 0.5, 1.0, 5.0, 10.0]:
            combined = baseline_logits + scale * holo_logits
            combo_rank = int((combined.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            combo_correct = torch.argmax(combined).item() in fact_ids
            if combo_correct:
                best_combo_scale = scale
                break
        else:
            best_combo_scale = -1

        sv_ratio = S[0] / S[1] if len(S) > 1 else float('inf')
        results_list.append({
            'prompt': prompt[:35], 'expected': expected,
            'baseline_rank': baseline_rank,
            'holo_rank': holo_rank, 'best_slot_rank': best_rank,
            'holo_top5': holo_top5, 'sv_ratio': round(float(sv_ratio), 2),
            'best_combo_scale': best_combo_scale,
        })
        status = 'IMPROVED' if holo_rank < baseline_rank else 'same/worse'
        print(f"  {expected:>8s}: base_rank={baseline_rank:>5d}, holo_rank={holo_rank:>5d}, "
              f"slot_best={best_rank:>5d}, SV1/SV2={sv_ratio:.1f} [{status}]")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = [r['expected'] for r in results_list]
    base_ranks = [r['baseline_rank'] for r in results_list]
    holo_ranks = [min(r['holo_rank'], max(base_ranks)*3) for r in results_list]
    x = range(len(labels))
    axes[0].bar([i-0.2 for i in x], base_ranks, 0.4, label='Baseline', color='red', alpha=0.7)
    axes[0].bar([i+0.2 for i in x], holo_ranks, 0.4, label='Holographic', color='blue', alpha=0.7)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, fontsize=8, rotation=45)
    axes[0].set_ylabel('Fact Token Rank')
    axes[0].set_title('Baseline vs Holographic Rank')
    axes[0].legend(fontsize=8)

    sv_ratios = [r['sv_ratio'] for r in results_list]
    axes[1].bar(labels, sv_ratios, color='purple', alpha=0.7)
    axes[1].set_ylabel('SV1/SV2 Ratio')
    axes[1].set_title('Singular Value Dominance')
    axes[1].tick_params(axis='x', rotation=45, labelsize=8)

    improved = sum(1 for r in results_list if r['holo_rank'] < r['baseline_rank'])
    axes[2].pie([improved, len(results_list)-improved],
                labels=['Improved', 'Not improved'],
                colors=['green', 'red'], autopct='%1.0f%%', startangle=90)
    axes[2].set_title('Holographic Salvage Success')

    plt.suptitle('Phase 32: Holographic Truth Salvage', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase32_holographic.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 32, 'name': 'Holographic Truth Salvage',
        'per_case': results_list,
        'improved_count': improved, 'total': len(results_list),
    }
    with open(os.path.join(RESULTS_DIR, 'phase32_holographic.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 32 RESULTS: Holographic Truth Salvage")
    print("=" * 70)
    print(f"  Improved: {improved}/{len(results_list)}")
    print("=" * 70)
    phase_complete(32)
    return results

if __name__ == '__main__':
    main()
