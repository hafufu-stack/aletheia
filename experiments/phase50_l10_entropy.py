# -*- coding: utf-8 -*-
"""
Phase 50: L10-Candidate Entropy Rejection
THE FINAL ANSWER: When Oracle detects hallucination, get candidates
from L10 (where facts are Rank 1), simulate each continuation,
and pick the one with lowest future entropy. Zero external knowledge.
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

ENTROPY_THRESHOLD = 1.0

def load_model():
    print("[P50] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_attn_entropy(model, input_ids):
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, return_dict=True)
    ents = []
    for attn in out.attentions:
        for h in range(attn.shape[1]):
            a = attn[0, h, -1, :].cpu().numpy()
            ents.append(float(-np.sum(a * np.log(a + 1e-12))))
    return float(np.mean(ents))

def logit_lens_topk(model, input_ids, layer_idx=10, k=5):
    """Get top-k token candidates from intermediate layer via Logit Lens."""
    hidden = {}
    def hook(module, args, output):
        hidden['h'] = output[0][0, -1, :].detach()
    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    normed = model.transformer.ln_f(hidden['h'].unsqueeze(0))
    logits = model.lm_head(normed).squeeze(0)
    topk = torch.topk(logits, k)
    return topk.indices.cpu().tolist(), topk.values.cpu().tolist()

def simulate_future_entropy(model, input_ids, candidate_id, n_future=3):
    """Simulate appending candidate and measure future entropy."""
    extended = torch.cat([input_ids,
                         torch.tensor([[candidate_id]], device=DEVICE)], dim=1)
    total_ent = 0
    ids = extended
    for step in range(n_future):
        ent = get_attn_entropy(model, ids)
        total_ent += ent
        # Generate next token greedily
        with torch.no_grad():
            out = model(ids)
        next_tok = torch.argmax(out.logits[:, -1, :]).item()
        ids = torch.cat([ids, torch.tensor([[next_tok]], device=DEVICE)], dim=1)
    return total_ent / n_future

def main():
    print("=" * 70)
    print("  Phase 50: L10-Candidate Entropy Rejection")
    print("  Oracle + L10 candidates + future entropy selection")
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

    # === P50a: L12 candidates vs L10 candidates ===
    print("\n[P50a] Comparing L12 vs L10 candidate pools...")
    comparison = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

        # L12 (final) top-5
        with torch.no_grad():
            out = model(inp)
        l12_top5 = torch.topk(out.logits[:, -1, :].squeeze(0), 5).indices.tolist()
        l12_tokens = [tok.decode([t]).encode('ascii','replace').decode().strip() for t in l12_top5]

        # L10 top-5
        l10_top5, l10_vals = logit_lens_topk(model, inp, layer_idx=10, k=5)
        l10_tokens = [tok.decode([t]).encode('ascii','replace').decode().strip() for t in l10_top5]

        fact_in_l12 = any(t in fact_ids for t in l12_top5)
        fact_in_l10 = any(t in fact_ids for t in l10_top5)

        comparison.append({
            'expected': expected, 'fact_in_l12': fact_in_l12,
            'fact_in_l10': fact_in_l10, 'l12_top5': l12_tokens,
            'l10_top5': l10_tokens,
        })
        tag_12 = 'YES' if fact_in_l12 else 'no'
        tag_10 = 'YES' if fact_in_l10 else 'no'
        print(f"  {expected:>12s}: L12=[{tag_12}]{l12_tokens}  L10=[{tag_10}]{l10_tokens}")

    l12_hit = sum(1 for c in comparison if c['fact_in_l12'])
    l10_hit = sum(1 for c in comparison if c['fact_in_l10'])
    print(f"  Fact in L12 top-5: {l12_hit}/{len(tests)}")
    print(f"  Fact in L10 top-5: {l10_hit}/{len(tests)}")

    # === P50b: Entropy-guided selection from L10 candidates ===
    print("\n[P50b] Entropy-guided selection from L10 top-K...")
    p50_results = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

        # Check Oracle
        base_ent = get_attn_entropy(model, inp)

        # Get L10 candidates
        l10_top5, _ = logit_lens_topk(model, inp, layer_idx=10, k=5)

        # Simulate each and pick lowest entropy
        best_tok = None
        best_ent = float('inf')
        candidate_ents = {}
        for cand in l10_top5:
            cand_ent = simulate_future_entropy(model, inp, cand, n_future=2)
            cand_name = tok.decode([cand]).encode('ascii','replace').decode().strip()
            candidate_ents[cand_name] = round(cand_ent, 3)
            if cand_ent < best_ent:
                best_ent = cand_ent
                best_tok = cand

        # Also check L12 greedy
        with torch.no_grad():
            out = model(inp)
        l12_greedy = torch.argmax(out.logits[:, -1, :]).item()
        l12_ent = simulate_future_entropy(model, inp, l12_greedy, n_future=2)

        is_correct = best_tok in fact_ids
        l12_correct = l12_greedy in fact_ids
        best_name = tok.decode([best_tok]).encode('ascii','replace').decode().strip()
        l12_name = tok.decode([l12_greedy]).encode('ascii','replace').decode().strip()

        p50_results.append({
            'expected': expected, 'oracle_ent': round(base_ent, 3),
            'l10_selected': best_name, 'l10_correct': is_correct,
            'l10_ent': round(best_ent, 3),
            'l12_selected': l12_name, 'l12_correct': l12_correct,
            'l12_ent': round(l12_ent, 3),
            'candidate_ents': candidate_ents,
        })

        l_tag = 'OK' if is_correct else 'FAIL'
        f_tag = 'OK' if l12_correct else 'FAIL'
        print(f"  {expected:>12s}: L10={best_name:>12s}[{l_tag}](H={best_ent:.3f}) "
              f"L12={l12_name:>8s}[{f_tag}](H={l12_ent:.3f})")

    l10_acc = sum(1 for r in p50_results if r['l10_correct']) / len(tests)
    l12_acc = sum(1 for r in p50_results if r['l12_correct']) / len(tests)

    # === P50c: Oracle-conditional system ===
    print(f"\n[P50c] Full system: L12 when confident, L10 when uncertain...")
    system_results = []
    for r in p50_results:
        if r['oracle_ent'] > ENTROPY_THRESHOLD:
            # Uncertain -> use L10 candidate
            correct = r['l10_correct']
            source = 'L10'
        else:
            # Confident -> use L12
            correct = r['l12_correct']
            source = 'L12'
        system_results.append({
            'expected': r['expected'], 'source': source, 'correct': correct,
        })
        tag = 'OK' if correct else 'FAIL'
        print(f"  {r['expected']:>12s}: [{source}] [{tag}]")

    system_acc = sum(1 for s in system_results if s['correct']) / len(tests)

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = [r['expected'][:6] for r in p50_results]
    x = range(len(labels))
    l10e = [r['l10_ent'] for r in p50_results]
    l12e = [r['l12_ent'] for r in p50_results]
    axes[0].bar([i-0.2 for i in x], l10e, 0.4, label='L10 best', color='green', alpha=0.7)
    axes[0].bar([i+0.2 for i in x], l12e, 0.4, label='L12 greedy', color='red', alpha=0.7)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, fontsize=6, rotation=45)
    axes[0].set_ylabel('Future Entropy')
    axes[0].set_title('L10 vs L12 Future Entropy')
    axes[0].legend(fontsize=8)

    methods = ['L12 greedy', 'L10 entropy', 'Hybrid']
    accs = [l12_acc*100, l10_acc*100, system_acc*100]
    colors = ['red', 'green', 'blue']
    axes[1].bar(methods, accs, color=colors, alpha=0.7)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Method Comparison')
    axes[1].set_ylim(0, 50)

    # Oracle vs L10 hit
    categories = ['Fact in L12\ntop-5', 'Fact in L10\ntop-5']
    hits = [l12_hit, l10_hit]
    axes[2].bar(categories, hits, color=['red', 'green'], alpha=0.7)
    axes[2].set_ylabel(f'Count (/{len(tests)})')
    axes[2].set_title('Candidate Pool Quality')

    plt.suptitle('Phase 50: L10-Candidate Entropy Rejection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase50_l10_entropy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 50, 'name': 'L10-Candidate Entropy Rejection',
        'l12_accuracy': l12_acc, 'l10_accuracy': l10_acc,
        'system_accuracy': system_acc,
        'l12_top5_hit': l12_hit, 'l10_top5_hit': l10_hit,
        'total': len(tests),
        'per_case': p50_results, 'system': system_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase50_l10_entropy.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 50 RESULTS")
    print("=" * 70)
    print(f"  L12 greedy:      {l12_acc:.0%}")
    print(f"  L10 entropy:     {l10_acc:.0%}")
    print(f"  Hybrid system:   {system_acc:.0%}")
    print(f"  L10 top-5 hits:  {l10_hit}/{len(tests)}")
    print("=" * 70)
    phase_complete(50)

if __name__ == '__main__':
    main()
