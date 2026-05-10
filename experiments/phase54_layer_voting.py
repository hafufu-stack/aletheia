# -*- coding: utf-8 -*-
"""
Phase 54: Multi-Layer Voting (Layer Ensemble)
Instead of using only L10, aggregate evidence from L8-L10.
If a token is consistently high-ranked across multiple intermediate
layers, it's a FACT being carried through the network, not grammar noise.
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
    print("[P54] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_all_layer_logits(model, tok, prompt):
    """Extract logits from all 12 layers via Logit Lens."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    layer_hs = {}
    handles = []
    for li in range(12):
        def make_hook(idx):
            def hook_fn(module, args, output):
                layer_hs[idx] = output[0][0, -1, :].detach()
            return hook_fn
        h = model.transformer.h[li].register_forward_hook(make_hook(li))
        handles.append(h)

    with torch.no_grad():
        out = model(**inp)
    for h in handles:
        h.remove()

    layer_logits = {}
    for li in range(12):
        normed = model.transformer.ln_f(layer_hs[li].unsqueeze(0))
        layer_logits[li] = model.lm_head(normed).squeeze(0)

    final_logits = out.logits[:, -1, :].squeeze(0)
    return layer_logits, final_logits

def main():
    print("=" * 70)
    print("  Phase 54: Multi-Layer Voting (Layer Ensemble)")
    print("  Aggregate L6-L10 to find consensus facts")
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

    # === P54a: Rank-based voting across layer windows ===
    print("\n[P54a] Layer window voting (reciprocal rank fusion)...")
    window_results = {}
    for start in range(0, 11):
        for end in range(start + 1, 12):
            layers = list(range(start, end + 1))
            if len(layers) < 2 or len(layers) > 6:
                continue
            correct = 0
            for prompt, fact_ids, expected in tests:
                ll, fl = get_all_layer_logits(model, tok, prompt)
                # Reciprocal rank fusion
                rrf_scores = torch.zeros(50257, device=DEVICE)
                for li in layers:
                    ranks = ll[li].argsort(descending=True).argsort() + 1
                    rrf_scores += 1.0 / ranks.float()
                winner = torch.argmax(rrf_scores).item()
                if winner in fact_ids:
                    correct += 1
            key = f"L{start}-L{end}"
            window_results[key] = correct / len(tests)

    # Print top windows
    sorted_w = sorted(window_results.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_w[:10]:
        print(f"  {k}: {v:.0%}")

    # === P54b: Probability ensemble (average softmax) ===
    print("\n[P54b] Softmax averaging across windows...")
    prob_results = {}
    for layers_range in [(6,10), (7,10), (8,10), (9,10), (6,11), (8,11), (0,11)]:
        start, end = layers_range
        layers = list(range(start, end + 1))
        correct = 0
        for prompt, fact_ids, expected in tests:
            ll, fl = get_all_layer_logits(model, tok, prompt)
            avg_probs = torch.zeros(50257, device=DEVICE)
            for li in layers:
                avg_probs += torch.softmax(ll[li], dim=-1)
            avg_probs /= len(layers)
            winner = torch.argmax(avg_probs).item()
            if winner in fact_ids:
                correct += 1
        key = f"L{start}-L{end}"
        prob_results[key] = correct / len(tests)
        print(f"  {key}: {correct}/{len(tests)} = {correct/len(tests):.0%}")

    # === P54c: Top-1 stability score ===
    print("\n[P54c] Top-1 stability (how many layers agree on same top-1)...")
    stability_results = []
    for prompt, fact_ids, expected in tests:
        ll, fl = get_all_layer_logits(model, tok, prompt)
        top1s = {}
        for li in range(12):
            t1 = torch.argmax(ll[li]).item()
            tname = tok.decode([t1]).encode('ascii','replace').decode().strip()
            top1s[li] = (t1, tname)

        # Find most common top-1 across L6-L10
        from collections import Counter
        mid_top1s = [top1s[li][0] for li in range(6, 11)]
        counter = Counter(mid_top1s)
        consensus, count = counter.most_common(1)[0]
        consensus_name = tok.decode([consensus]).encode('ascii','replace').decode().strip()
        is_fact = consensus in fact_ids

        stability_results.append({
            'expected': expected, 'consensus': consensus_name,
            'agreement': count, 'is_fact': is_fact,
            'top1_per_layer': {str(k): v[1] for k, v in top1s.items()},
        })
        tag = 'OK' if is_fact else 'FAIL'
        print(f"  {expected:>12s}: consensus='{consensus_name}' ({count}/5 agree) [{tag}]")
        layer_str = ' '.join([f"L{l}={top1s[l][1][:4]}" for l in range(6, 12)])
        print(f"    {layer_str}")

    consensus_acc = sum(1 for s in stability_results if s['is_fact']) / len(tests)

    # === P54d: Confidence-weighted layer selection ===
    print("\n[P54d] Confidence-weighted: use layer with highest top-1 prob...")
    conf_results = []
    for prompt, fact_ids, expected in tests:
        ll, fl = get_all_layer_logits(model, tok, prompt)
        best_layer = None
        best_conf = -1
        for li in range(6, 11):
            probs = torch.softmax(ll[li], dim=-1)
            top1_conf = probs.max().item()
            if top1_conf > best_conf:
                best_conf = top1_conf
                best_layer = li

        winner = torch.argmax(ll[best_layer]).item()
        wname = tok.decode([winner]).encode('ascii','replace').decode().strip()
        is_correct = winner in fact_ids

        conf_results.append({
            'expected': expected, 'best_layer': best_layer,
            'confidence': round(best_conf, 4), 'selected': wname,
            'correct': is_correct,
        })
        tag = 'OK' if is_correct else 'FAIL'
        print(f"  {expected:>12s}: L{best_layer}(conf={best_conf:.4f}) -> '{wname}' [{tag}]")

    conf_acc = sum(1 for c in conf_results if c['correct']) / len(tests)

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Method comparison
    methods = ['L12 final', 'L10 only', 'RRF best', 'Softmax avg', 'Consensus', 'Conf-select']
    best_rrf = max(window_results.values()) if window_results else 0
    best_prob = max(prob_results.values()) if prob_results else 0
    accs = [8.3, 33.3, best_rrf*100, best_prob*100, consensus_acc*100, conf_acc*100]
    colors = ['red','green','teal','blue','purple','orange']
    axes[0].bar(methods, accs, color=colors, alpha=0.7)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Method Comparison')
    axes[0].tick_params(axis='x', rotation=30, labelsize=7)

    # Consensus agreement distribution
    agreements = [s['agreement'] for s in stability_results]
    axes[1].hist(agreements, bins=[1,2,3,4,5,6], color='purple', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Layers agreeing on top-1')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Consensus Strength (L6-L10)')

    # Confidence per layer
    labels = [c['expected'][:6] for c in conf_results]
    best_ls = [c['best_layer'] for c in conf_results]
    confs = [c['confidence'] for c in conf_results]
    cols = ['green' if c['correct'] else 'red' for c in conf_results]
    axes[2].bar(labels, confs, color=cols, alpha=0.7)
    for i, bl in enumerate(best_ls):
        axes[2].text(i, confs[i]+0.002, f'L{bl}', ha='center', fontsize=6)
    axes[2].set_ylabel('Max Confidence')
    axes[2].set_title('Confidence-Weighted Layer Selection')
    axes[2].tick_params(axis='x', rotation=45, labelsize=6)

    plt.suptitle('Phase 54: Multi-Layer Voting', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase54_layer_voting.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 54, 'name': 'Multi-Layer Voting',
        'window_results': window_results,
        'prob_ensemble': prob_results,
        'consensus_accuracy': consensus_acc,
        'confidence_accuracy': conf_acc,
        'best_rrf_window': sorted_w[0] if sorted_w else None,
        'stability': stability_results,
        'confidence_selection': conf_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase54_layer_voting.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 54 RESULTS")
    print("=" * 70)
    print(f"  L12 final:     8%")
    print(f"  L10 only:      33%")
    print(f"  Best RRF:      {sorted_w[0][0]} -> {sorted_w[0][1]:.0%}")
    print(f"  Best prob avg: {max(prob_results, key=prob_results.get)} -> {max(prob_results.values()):.0%}")
    print(f"  Consensus:     {consensus_acc:.0%}")
    print(f"  Confidence:    {conf_acc:.0%}")
    print("=" * 70)
    phase_complete(54)

if __name__ == '__main__':
    main()
