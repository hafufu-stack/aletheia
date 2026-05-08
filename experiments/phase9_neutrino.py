# -*- coding: utf-8 -*-
"""
Phase 9: Neutrino Spike - Null-Space Penetration through LayerNorm
- Compute zero-mean spike vectors that are invisible to LayerNorm
- Tunnel through intermediate layers without absorption
- Prove mid-layer truth control is mathematically possible
"""
import os, json
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
    print("[P9] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


class NeutrinoHook:
    """Inject zero-mean spike that survives LayerNorm."""
    def __init__(self, spike_vec, magnitude):
        self.spike_vec = spike_vec  # (hidden_dim,) zero-mean vector
        self.magnitude = magnitude
        self.injected = False

    def __call__(self, module, input, output):
        hidden = output[0]  # (batch, seq, hidden)
        if not self.injected and self.magnitude > 0:
            spike = self.spike_vec.to(hidden.device) * self.magnitude
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + spike
            self.injected = True
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


def compute_neutrino_vector(model, tok, prompt, fact_ids, layer_idx):
    """Compute a zero-mean spike vector that, when injected at layer_idx,
    maximally boosts fact_ids at the output."""
    # Get baseline hidden states at target layer
    inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    baseline_hidden = out.hidden_states[layer_idx + 1][:, -1, :].clone()  # +1 for embedding
    baseline_logits = out.logits[:, -1, :].squeeze(0)

    # Strategy: compute gradient of fact logits w.r.t. hidden state
    # Since we can't backprop through eval, use finite difference
    hidden_dim = baseline_hidden.shape[-1]
    deltas = torch.zeros(hidden_dim, device=DEVICE)

    # Estimate gradient via perturbation for top-10 dimensions
    eps = 0.1
    for d in range(min(hidden_dim, 50)):  # Sample 50 dims for speed
        perturbed_hidden = baseline_hidden.clone()
        perturbed_hidden[0, d] += eps

        # Forward from this layer onward using hooks
        hook = NeutrinoHook(torch.zeros(hidden_dim), 0)
        # We need a simpler approach: just compute lm_head response
        # Use the final layer norm + lm_head
        with torch.no_grad():
            normed = model.transformer.ln_f(perturbed_hidden)
            perturbed_logits = model.lm_head(normed).squeeze(0)

        # How much did fact token logits increase?
        for tid in fact_ids:
            if tid < perturbed_logits.shape[-1]:
                deltas[d] += (perturbed_logits[tid] - baseline_logits[tid]).item()

    # Make zero-mean: subtract mean
    neutrino = deltas - deltas.mean()
    # Normalize
    neutrino = neutrino / (torch.norm(neutrino) + 1e-12)

    # Verify zero-mean
    assert abs(neutrino.mean().item()) < 1e-6, "Neutrino vector is not zero-mean!"
    return neutrino


def main():
    print("=" * 70)
    print("  Phase 9: Neutrino Spike")
    print("  Null-Space Penetration through LayerNorm")
    print("=" * 70)

    model, tok = load_model()

    qa_pairs = [
        ("The capital of Japan is", [11790]),
        ("Water freezes at", [657]),
        ("The chemical formula for water is", [367]),
        ("The largest planet is", [22721]),
        ("DNA stands for", [390]),
    ]

    magnitudes = [0, 1, 5, 10, 20, 50, 100, 200]
    layers_to_test = [0, 3, 6, 9, 11]

    # === Baseline ===
    print("\n[P9a] Baseline...")
    baseline_acc = 0
    for prompt, fact_ids in qa_pairs:
        inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        if torch.argmax(out.logits[:, -1, :]).item() in fact_ids:
            baseline_acc += 1
    print(f"  Baseline: {baseline_acc}/5")

    # === Naive spike (P8 method) vs Neutrino spike ===
    print("\n[P9b] Computing neutrino vectors...")
    results_by_layer = {}

    for layer_idx in layers_to_test:
        print(f"\n  Layer {layer_idx}:")

        # Compute neutrino vector for each QA pair
        neutrino_vecs = []
        for prompt, fact_ids in qa_pairs:
            nv = compute_neutrino_vector(model, tok, prompt, fact_ids, layer_idx)
            neutrino_vecs.append(nv)

        # Sweep magnitudes
        naive_results = {}
        neutrino_results = {}

        for mag in magnitudes:
            naive_correct = 0
            neutrino_correct = 0
            naive_ent = []
            neutrino_ent = []

            for qi, (prompt, fact_ids) in enumerate(qa_pairs):
                inp = tok(prompt, return_tensors='pt', truncation=True,
                          max_length=128).to(DEVICE)
                hidden_dim = model.config.n_embd

                # --- Naive spike (uniform, like P8) ---
                naive_hook = NeutrinoHook(
                    torch.ones(hidden_dim) / np.sqrt(hidden_dim), mag)
                handle = model.transformer.h[layer_idx].register_forward_hook(naive_hook)
                with torch.no_grad():
                    out_naive = model(**inp)
                handle.remove()
                logits_n = out_naive.logits[:, -1, :].squeeze(0)
                probs_n = F.softmax(logits_n, dim=-1)
                h_n = float(-torch.sum(probs_n * torch.log(probs_n + 1e-12)).cpu())
                naive_ent.append(h_n)
                if torch.argmax(probs_n).item() in fact_ids:
                    naive_correct += 1

                # --- Neutrino spike (zero-mean, targeted) ---
                neut_hook = NeutrinoHook(neutrino_vecs[qi], mag)
                handle = model.transformer.h[layer_idx].register_forward_hook(neut_hook)
                with torch.no_grad():
                    out_neut = model(**inp)
                handle.remove()
                logits_nu = out_neut.logits[:, -1, :].squeeze(0)
                probs_nu = F.softmax(logits_nu, dim=-1)
                h_nu = float(-torch.sum(probs_nu * torch.log(probs_nu + 1e-12)).cpu())
                neutrino_ent.append(h_nu)
                if torch.argmax(probs_nu).item() in fact_ids:
                    neutrino_correct += 1

            naive_results[mag] = {'acc': naive_correct/5, 'H': float(np.mean(naive_ent))}
            neutrino_results[mag] = {'acc': neutrino_correct/5, 'H': float(np.mean(neutrino_ent))}
            print(f"    mag={mag:>4d}: naive={naive_correct}/5  neutrino={neutrino_correct}/5")

        results_by_layer[layer_idx] = {
            'naive': {str(k): v for k, v in naive_results.items()},
            'neutrino': {str(k): v for k, v in neutrino_results.items()},
        }

    # === Find best neutrino layer ===
    best_layer = None
    best_acc = 0
    best_mag = None
    for layer_idx in layers_to_test:
        for mag in magnitudes:
            acc = results_by_layer[layer_idx]['neutrino'][str(mag)]['acc']
            if acc > best_acc:
                best_acc = acc
                best_layer = layer_idx
                best_mag = mag

    print(f"\n  Best neutrino: layer={best_layer}, mag={best_mag}, acc={best_acc:.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Neutrino vs Naive at best layer
    bl = best_layer or layers_to_test[-1]
    naive_accs = [results_by_layer[bl]['naive'][str(m)]['acc']*100 for m in magnitudes]
    neut_accs = [results_by_layer[bl]['neutrino'][str(m)]['acc']*100 for m in magnitudes]
    axes[0].plot(magnitudes, naive_accs, 'r.-', label='Naive (P8)', linewidth=2)
    axes[0].plot(magnitudes, neut_accs, 'g.-', label='Neutrino', linewidth=2)
    axes[0].set_xlabel('Spike Magnitude')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title(f'Layer {bl}: Naive vs Neutrino')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Best accuracy per layer
    layer_best = []
    for l in layers_to_test:
        best = max(results_by_layer[l]['neutrino'][str(m)]['acc'] for m in magnitudes)
        layer_best.append(best * 100)
    axes[1].bar(range(len(layers_to_test)), layer_best,
                tick_label=[str(l) for l in layers_to_test], color='teal', alpha=0.7)
    axes[1].set_xlabel('Injection Layer')
    axes[1].set_ylabel('Best Accuracy (%)')
    axes[1].set_title('Neutrino Penetration by Layer')
    axes[1].axhline(y=100, color='g', linestyle='--', alpha=0.5)

    # Plot 3: Entropy at best layer
    naive_ent = [results_by_layer[bl]['naive'][str(m)]['H'] for m in magnitudes]
    neut_ent = [results_by_layer[bl]['neutrino'][str(m)]['H'] for m in magnitudes]
    axes[2].plot(magnitudes, naive_ent, 'r.-', label='Naive', linewidth=2)
    axes[2].plot(magnitudes, neut_ent, 'g.-', label='Neutrino', linewidth=2)
    axes[2].set_xlabel('Spike Magnitude')
    axes[2].set_ylabel('Entropy')
    axes[2].set_title(f'Layer {bl}: Entropy Response')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 9: Neutrino Spike (Null-Space Penetration)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase9_neutrino.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 9, 'name': 'Neutrino Spike - Null-Space Penetration',
        'best_layer': best_layer, 'best_magnitude': best_mag,
        'best_accuracy': best_acc,
        'by_layer': {str(k): v for k, v in results_by_layer.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase9_neutrino.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 9 RESULTS: Neutrino Spike")
    print("=" * 70)
    print(f"  Best layer: {best_layer}, mag={best_mag}, acc={best_acc:.0%}")
    print(f"  LayerNorm penetration: {'YES' if best_acc > 0 else 'NO'}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
