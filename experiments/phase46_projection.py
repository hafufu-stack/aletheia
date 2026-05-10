# -*- coding: utf-8 -*-
"""
Phase 46: 11D Clique Projection (Truth Subspace Projection)
P41 failed because it ADDED a vector. This time: PROJECT hidden state
onto the truth subspace (remove noise dimensions), then decode.
From P43: facts live on lower-dim manifold. Strip away the hallu dimensions.
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
    print("[P46] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_hidden(model, tok, prompt, layer_idx=11):
    hidden = {}
    def hook(module, args, output):
        hidden['h'] = output[0][0, -1, :].detach().cpu().numpy()
    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    return hidden['h'], out.logits[:, -1, :].squeeze(0)

def main():
    print("=" * 70)
    print("  Phase 46: 11D Clique Projection")
    print("  Project hidden state onto truth subspace, strip hallu noise")
    print("=" * 70)

    model, tok = load_model()

    # Training: fact prompts for truth subspace
    train_facts = [
        "The capital of Japan is", "The capital of France is",
        "The capital of Germany is", "The capital of Italy is",
        "Albert Einstein developed the theory of",
        "The sun is a", "The Earth orbits the",
        "The boiling point of water is",
    ]
    # Test set
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

    # === P46a: Build truth subspace via PCA ===
    print("\n[P46a] Building truth subspace from training facts...")
    for target_layer in [6, 8, 10, 11]:
        train_states = [get_hidden(model, tok, p, target_layer)[0] for p in train_facts]
        train_matrix = np.array(train_states)
        train_mean = train_matrix.mean(axis=0)
        centered = train_matrix - train_mean

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # === P46b: Project test hidden states onto truth subspace ===
        for n_dims in [3, 5, 8, 11, 20]:
            # Truth subspace basis (top n_dims principal components)
            basis = Vt[:n_dims]  # (n_dims, d_model)

            correct = 0
            for prompt, fact_ids, expected in tests:
                h, base_logits = get_hidden(model, tok, prompt, target_layer)

                # Project onto truth subspace
                h_centered = h - train_mean
                coords = h_centered @ basis.T  # (n_dims,)
                h_projected = coords @ basis + train_mean  # back to d_model

                # Decode projected hidden state
                h_tensor = torch.tensor(h_projected, dtype=torch.float32, device=DEVICE)
                proj_logits = model.lm_head(model.transformer.ln_f(h_tensor.unsqueeze(0))).squeeze(0)

                if torch.argmax(proj_logits).item() in fact_ids:
                    correct += 1

            if target_layer == 11:
                print(f"  L{target_layer} dims={n_dims:>2d}: {correct}/{len(tests)} = {correct/len(tests):.0%}")

    # === P46c: Interpolation: alpha*projected + (1-alpha)*original ===
    print(f"\n[P46c] Interpolation sweep (L11, 11D)...")
    # Rebuild L11 basis
    train_states_11 = [get_hidden(model, tok, p, 11)[0] for p in train_facts]
    tm = np.array(train_states_11)
    tmean = tm.mean(axis=0)
    _, _, Vt11 = np.linalg.svd(tm - tmean, full_matrices=False)
    basis11 = Vt11[:11]

    interp_results = {}
    for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        correct = 0
        ranks = []
        for prompt, fact_ids, expected in tests:
            h, base_logits = get_hidden(model, tok, prompt, 11)

            h_centered = h - tmean
            coords = h_centered @ basis11.T
            h_proj = coords @ basis11 + tmean

            h_interp = alpha * h_proj + (1 - alpha) * h
            h_t = torch.tensor(h_interp, dtype=torch.float32, device=DEVICE)
            logits = model.lm_head(model.transformer.ln_f(h_t.unsqueeze(0))).squeeze(0)

            rank = int((logits.argsort(descending=True) == fact_ids[0]).nonzero().item()) + 1
            ranks.append(rank)
            if torch.argmax(logits).item() in fact_ids:
                correct += 1

        interp_results[alpha] = {
            'accuracy': correct / len(tests),
            'median_rank': float(np.median(ranks)),
        }
        print(f"  alpha={alpha:.1f}: {correct}/{len(tests)} = {correct/len(tests):.0%} "
              f"median_rank={np.median(ranks):.0f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Dims sweep (L11)
    dims_list = [3, 5, 8, 11, 20]
    dim_accs = []
    for nd in dims_list:
        b = Vt11[:nd]
        c = 0
        for prompt, fact_ids, _ in tests:
            h, _ = get_hidden(model, tok, prompt, 11)
            hp = (h - tmean) @ b.T @ b + tmean
            ht = torch.tensor(hp, dtype=torch.float32, device=DEVICE)
            lg = model.lm_head(model.transformer.ln_f(ht.unsqueeze(0))).squeeze(0)
            if torch.argmax(lg).item() in fact_ids:
                c += 1
        dim_accs.append(c / len(tests) * 100)
    axes[0].plot(dims_list, dim_accs, 'g.-', linewidth=2, markersize=10)
    axes[0].set_xlabel('Subspace Dimensions')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Truth Subspace Dimension')
    axes[0].grid(True, alpha=0.3)

    # Interpolation
    alphas = sorted(interp_results.keys())
    i_accs = [interp_results[a]['accuracy']*100 for a in alphas]
    axes[1].plot(alphas, i_accs, 'b.-', linewidth=2, markersize=10)
    axes[1].set_xlabel('Alpha (projection weight)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Original <-> Projected')
    axes[1].grid(True, alpha=0.3)

    # Rank comparison
    i_ranks = [interp_results[a]['median_rank'] for a in alphas]
    axes[2].bar([str(a) for a in alphas], i_ranks, color='purple', alpha=0.7)
    axes[2].set_xlabel('Alpha')
    axes[2].set_ylabel('Median Rank')
    axes[2].set_title('Fact Token Rank')

    plt.suptitle('Phase 46: 11D Clique Projection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase46_projection.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 46, 'name': '11D Clique Projection',
        'interp_results': {str(k): v for k, v in interp_results.items()},
        'subspace_dims_accuracy': dict(zip(dims_list, dim_accs)),
    }
    with open(os.path.join(RESULTS_DIR, 'phase46_projection.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 46 RESULTS: 11D Clique Projection")
    print("=" * 70)
    for a in alphas:
        print(f"  alpha={a:.1f}: {interp_results[a]['accuracy']:.0%} rank={interp_results[a]['median_rank']:.0f}")
    print("=" * 70)
    phase_complete(46)

if __name__ == '__main__':
    main()
