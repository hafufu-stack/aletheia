# -*- coding: utf-8 -*-
"""
Phase 24: Neural Wormhole - Self-Firing Spike
Bypass LayerNorm by routing Fact Head (L2H1) output directly to logits.
The model generates its own truth spike without external knowledge.
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
    print("[P24] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def main():
    print("=" * 70)
    print("  Phase 24: Neural Wormhole - Self-Firing Spike")
    print("  Can the model's own Fact Head bypass LayerNorm?")
    print("=" * 70)

    model, tok = load_model()
    lm_head = model.lm_head  # projection: hidden -> vocab

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

    # Fact head: L2H1 (from Phase 14)
    fact_layer, fact_head = 2, 1
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    # === P24a: Extract Fact Head output and project to logits ===
    print("\n[P24a] Extracting Fact Head (L2H1) -> direct logit projection...")

    results_wormhole = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        # Hook to capture attention head output
        head_output = {}
        def hook_fn(module, args, output):
            # output[0] shape: (batch, seq, hidden)
            hs = output[0]
            if hs.dim() == 3:
                head_output['hidden'] = hs[0, -1, :].detach()
            else:
                head_output['hidden'] = hs[-1, :].detach()
        handle = model.transformer.h[fact_layer].register_forward_hook(hook_fn)

        with torch.no_grad():
            out = model(**inp)
        handle.remove()

        # Extract fact head's slice from hidden state
        h = head_output['hidden']
        head_slice = h[fact_head * d_head : (fact_head + 1) * d_head]

        # Project through LM head (zero-pad to full hidden dim)
        wormhole_vec = torch.zeros(model.config.n_embd, device=DEVICE)
        wormhole_vec[fact_head * d_head : (fact_head + 1) * d_head] = head_slice

        # Get wormhole logits
        wormhole_logits = lm_head(wormhole_vec.unsqueeze(0)).squeeze(0)

        # Get baseline logits
        baseline_logits = out.logits[:, -1, :].squeeze(0)

        # Add wormhole as bias (scaled)
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
            combined = baseline_logits + scale * wormhole_logits
            winner = torch.argmax(combined).item()
            correct = winner in fact_ids

            if scale == 1.0:
                # Check wormhole's top predictions
                wh_top5 = wormhole_logits.argsort(descending=True)[:5]
                wh_top5_tokens = [tok.decode([t]).encode('ascii','replace').decode().strip()
                                  for t in wh_top5]
                fact_rank_in_wh = (wormhole_logits.argsort(descending=True) == fact_ids[0]).nonzero()
                fr = fact_rank_in_wh.item() + 1 if fact_rank_in_wh.numel() > 0 else -1

                results_wormhole.append({
                    'prompt': prompt[:35], 'expected': expected,
                    'correct_at_scale1': correct,
                    'wormhole_top5': wh_top5_tokens,
                    'fact_rank_in_wormhole': fr,
                    'baseline_rank': int((baseline_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1,
                })

        status = 'OK' if results_wormhole[-1]['correct_at_scale1'] else 'FAIL'
        print(f"  {expected:>8s}: baseline_rank={results_wormhole[-1]['baseline_rank']:>4d}, "
              f"wh_fact_rank={results_wormhole[-1]['fact_rank_in_wormhole']:>5d}, "
              f"wh_top5={results_wormhole[-1]['wormhole_top5'][:3]} [{status}]")

    # === P24b: Scale sweep ===
    print("\n[P24b] Wormhole scale sweep...")
    scale_results = {}
    for scale in [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        correct = 0
        for prompt, fact_ids, expected in tests:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            head_output2 = {}
            def hook_fn2(module, args, output):
                hs = output[0]
                if hs.dim() == 3:
                    head_output2['hidden'] = hs[0, -1, :].detach()
                else:
                    head_output2['hidden'] = hs[-1, :].detach()
            handle = model.transformer.h[fact_layer].register_forward_hook(hook_fn2)
            with torch.no_grad():
                out = model(**inp)
            handle.remove()
            h = head_output2['hidden']
            wv = torch.zeros(model.config.n_embd, device=DEVICE)
            wv[fact_head * d_head : (fact_head + 1) * d_head] = h[fact_head * d_head : (fact_head + 1) * d_head]
            wl = lm_head(wv.unsqueeze(0)).squeeze(0)
            combined = out.logits[:, -1, :].squeeze(0) + scale * wl
            if torch.argmax(combined).item() in fact_ids:
                correct += 1
        scale_results[scale] = correct / len(tests)
        print(f"  scale={scale:>5.1f}: {correct}/{len(tests)} = {correct/len(tests):.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    scales = sorted(scale_results.keys())
    accs = [scale_results[s]*100 for s in scales]
    axes[0].plot(scales, accs, 'g.-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Wormhole Scale')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Wormhole Scale vs Accuracy')
    axes[0].grid(True, alpha=0.3)

    b_ranks = [r['baseline_rank'] for r in results_wormhole]
    w_ranks = [r['fact_rank_in_wormhole'] for r in results_wormhole]
    labels = [r['expected'] for r in results_wormhole]
    x = range(len(labels))
    axes[1].bar([i-0.2 for i in x], b_ranks, 0.4, label='Baseline', color='red', alpha=0.7)
    axes[1].bar([i+0.2 for i in x], [min(r, max(b_ranks)*2) for r in w_ranks], 0.4,
                label='Wormhole', color='blue', alpha=0.7)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, fontsize=8, rotation=45)
    axes[1].set_ylabel('Fact Token Rank')
    axes[1].set_title('Baseline vs Wormhole Rank')
    axes[1].legend(fontsize=8)

    correct_count = sum(1 for r in results_wormhole if r['correct_at_scale1'])
    axes[2].pie([correct_count, len(tests)-correct_count],
                labels=['Correct', 'Wrong'],
                colors=['green', 'red'], autopct='%1.0f%%', startangle=90)
    axes[2].set_title('Wormhole (scale=1.0)')

    plt.suptitle('Phase 24: Neural Wormhole', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase24_wormhole.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 24, 'name': 'Neural Wormhole',
        'scale_sweep': {str(k): v for k, v in scale_results.items()},
        'per_case': results_wormhole,
    }
    with open(os.path.join(RESULTS_DIR, 'phase24_wormhole.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 24 RESULTS: Neural Wormhole")
    print("=" * 70)
    print(f"  Scale=1.0 accuracy: {correct_count}/{len(tests)}")
    print(f"  Best scale: {max(scale_results, key=scale_results.get)}")
    print("=" * 70)

    phase_complete(24)
    return results

if __name__ == '__main__':
    main()
