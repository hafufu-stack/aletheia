# -*- coding: utf-8 -*-
"""
Phase 48: Contrastive Layer Decoding
P3 used contrastive decoding between TWO MODELS.
P44 proved facts exist at intermediate layers but get suppressed deeper.
Key insight: subtract late-layer logits from early-layer logits.
Tokens that DROP in rank from L6->L11 are exactly the facts drowned by grammar.
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
    print("[P48] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def logit_lens_all(model, tok, prompt):
    """Get logits from every layer via Logit Lens."""
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
    for li, hs in layer_hs.items():
        normed = model.transformer.ln_f(hs.unsqueeze(0))
        layer_logits[li] = model.lm_head(normed).squeeze(0)

    return layer_logits, out.logits[:, -1, :].squeeze(0)

def main():
    print("=" * 70)
    print("  Phase 48: Contrastive Layer Decoding")
    print("  logits(early) - logits(late) = suppressed facts")
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
    ]

    # === P48a: Rank trajectory (which tokens decline L_early -> L_late?) ===
    print("\n[P48a] Fact token rank trajectory across layers...")
    trajectories = []
    for prompt, fact_ids, expected in tests:
        layer_logits, final_logits = logit_lens_all(model, tok, prompt)
        ranks = {}
        for li in range(12):
            r = int((layer_logits[li].argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            ranks[li] = r
        final_r = int((final_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1

        # Find the layer where fact is best ranked
        best_l = min(ranks, key=ranks.get)
        trajectories.append({
            'expected': expected, 'ranks': ranks,
            'best_layer': best_l, 'best_rank': ranks[best_l],
            'final_rank': final_r,
        })
        rank_str = ' '.join([f'{ranks[l]:>5d}' for l in range(12)])
        print(f"  {expected:>8s}: best=L{best_l}:r{ranks[best_l]:>4d} final=r{final_r:>4d}")

    # === P48b: Contrastive Layer Decoding ===
    print("\n[P48b] Contrastive decoding: early - alpha*late...")
    best_config = {'early': 0, 'late': 0, 'alpha': 0, 'acc': 0, 'median': 99999}

    for early in [3, 4, 5, 6, 7, 8]:
        for late in [10, 11]:
            if early >= late:
                continue
            for alpha in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]:
                correct = 0
                ranks = []
                for prompt, fact_ids, expected in tests:
                    layer_logits, _ = logit_lens_all(model, tok, prompt)
                    # Contrastive: amplify what's in early but not in late
                    contrastive = layer_logits[early] - alpha * layer_logits[late]
                    rank = int((contrastive.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
                    ranks.append(rank)
                    if torch.argmax(contrastive).item() in fact_ids:
                        correct += 1
                acc = correct / len(tests)
                med = float(np.median(ranks))

                if med < best_config['median'] or (med == best_config['median'] and acc > best_config['acc']):
                    best_config = {'early': early, 'late': late, 'alpha': alpha,
                                   'acc': acc, 'median': med}

                if early == 6 and late == 11:
                    print(f"  L{early}-{alpha:.1f}*L{late}: {correct}/{len(tests)} "
                          f"median_rank={med:.0f}")

    print(f"\n  BEST CONFIG: L{best_config['early']} - {best_config['alpha']}*L{best_config['late']}: "
          f"acc={best_config['acc']:.0%} median={best_config['median']:.0f}")

    # === P48c: Contrastive + final logits fusion ===
    print("\n[P48c] Contrastive + final logits fusion...")
    be = best_config['early']
    bl = best_config['late']
    ba = best_config['alpha']

    fusion_results = {}
    for beta in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        correct = 0
        ranks = []
        for prompt, fact_ids, expected in tests:
            layer_logits, final_logits = logit_lens_all(model, tok, prompt)
            contrastive = layer_logits[be] - ba * layer_logits[bl]
            fused = final_logits + beta * contrastive
            rank = int((fused.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            ranks.append(rank)
            if torch.argmax(fused).item() in fact_ids:
                correct += 1
        fusion_results[beta] = {
            'accuracy': correct / len(tests),
            'median_rank': float(np.median(ranks)),
        }
        print(f"  beta={beta:.1f}: {correct}/{len(tests)} median_rank={np.median(ranks):.0f}")

    # === P48d: Per-token analysis of what contrastive decoding surfaces ===
    print(f"\n[P48d] What does contrastive decoding surface? (L{be}-{ba}*L{bl})...")
    surfaced = []
    for prompt, fact_ids, expected in tests:
        layer_logits, final_logits = logit_lens_all(model, tok, prompt)
        contrastive = layer_logits[be] - ba * layer_logits[bl]
        top5_c = contrastive.argsort(descending=True)[:5]
        top5_f = final_logits.argsort(descending=True)[:5]

        c_tokens = [tok.decode([t.item()]).encode('ascii','replace').decode().strip() for t in top5_c]
        f_tokens = [tok.decode([t.item()]).encode('ascii','replace').decode().strip() for t in top5_f]
        fact_in_c5 = any(t.item() in fact_ids for t in top5_c)
        fact_in_f5 = any(t.item() in fact_ids for t in top5_f)

        c_rank = int((contrastive.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
        f_rank = int((final_logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1

        surfaced.append({
            'expected': expected, 'c_rank': c_rank, 'f_rank': f_rank,
            'c_top5': c_tokens, 'f_top5': f_tokens,
            'fact_in_contrastive_top5': fact_in_c5,
        })
        improved = 'IMPROVED' if c_rank < f_rank else 'worse' if c_rank > f_rank else 'same'
        print(f"  {expected:>8s}: final_r={f_rank:>5d} contrastive_r={c_rank:>5d} [{improved}]")
        print(f"    Final top5:       {f_tokens}")
        print(f"    Contrastive top5: {c_tokens}")

    improved_count = sum(1 for s in surfaced if s['c_rank'] < s['f_rank'])
    fact_in_top5 = sum(1 for s in surfaced if s['fact_in_contrastive_top5'])

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Rank trajectories
    for t in trajectories:
        color = 'green' if t['best_rank'] < t['final_rank'] else 'red'
        axes[0].plot(range(12), [t['ranks'][l] for l in range(12)],
                    '.-', alpha=0.6, color=color, label=t['expected'])
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Fact Token Rank')
    axes[0].set_title('Rank Trajectory (lower=better)')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Contrastive vs final rank
    labels = [s['expected'] for s in surfaced]
    f_ranks = [s['f_rank'] for s in surfaced]
    c_ranks = [s['c_rank'] for s in surfaced]
    x = range(len(labels))
    axes[1].bar([i-0.2 for i in x], f_ranks, 0.4, label='Final', color='red', alpha=0.7)
    axes[1].bar([i+0.2 for i in x], c_ranks, 0.4, label='Contrastive', color='green', alpha=0.7)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[1].set_ylabel('Fact Rank')
    axes[1].set_title(f'Final vs Contrastive ({improved_count}/{len(tests)} improved)')
    axes[1].legend(fontsize=8)
    axes[1].set_yscale('log')

    # Fusion sweep
    betas = sorted(fusion_results.keys())
    f_accs = [fusion_results[b]['accuracy']*100 for b in betas]
    f_meds = [fusion_results[b]['median_rank'] for b in betas]
    axes[2].plot(betas, f_meds, 'b.-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Beta (contrastive weight)')
    axes[2].set_ylabel('Median Fact Rank')
    axes[2].set_title('Fusion: final + beta*contrastive')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 48: Contrastive Layer Decoding', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase48_contrastive_layer.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 48, 'name': 'Contrastive Layer Decoding',
        'best_contrastive_config': best_config,
        'fusion_results': {str(k): v for k, v in fusion_results.items()},
        'improved_by_contrastive': improved_count,
        'fact_in_contrastive_top5': fact_in_top5,
        'total': len(tests),
        'surfaced': surfaced,
        'trajectories': trajectories,
    }
    with open(os.path.join(RESULTS_DIR, 'phase48_contrastive_layer.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 48 RESULTS: Contrastive Layer Decoding")
    print("=" * 70)
    print(f"  Best config: L{best_config['early']} - {best_config['alpha']}*L{best_config['late']}")
    print(f"  Rank improved: {improved_count}/{len(tests)}")
    print(f"  Fact in contrastive top-5: {fact_in_top5}/{len(tests)}")
    for b in betas:
        print(f"  Fusion beta={b:.1f}: acc={fusion_results[b]['accuracy']:.0%} "
              f"med_rank={fusion_results[b]['median_rank']:.0f}")
    print("=" * 70)
    phase_complete(48)

if __name__ == '__main__':
    main()
