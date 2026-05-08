# -*- coding: utf-8 -*-
"""
Phase 8: Layer-wise Spike Propagation
- Inject spikes at each of GPT-2's 12 layers
- Measure which layer produces the strongest factual grounding
- Bridge to SNN research: spike timing and layer depth
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
    print("[P8] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


class SpikeHook:
    """Hook to inject spikes at a specific layer's hidden states."""
    def __init__(self, fact_token_ids, magnitude, vocab_size):
        self.fact_token_ids = fact_token_ids
        self.magnitude = magnitude
        self.vocab_size = vocab_size
        self.activated = False

    def __call__(self, module, input, output):
        # output is a tuple; output[0] is the hidden state (batch, seq, hidden)
        hidden = output[0]
        if self.magnitude > 0 and not self.activated:
            # Create a spike direction in hidden space
            # Use the last token's hidden state and amplify it
            spike = torch.zeros_like(hidden[:, -1:, :])
            # Add spike energy uniformly to boost signal
            spike += self.magnitude * 0.01
            hidden = hidden.clone()
            hidden[:, -1:, :] = hidden[:, -1:, :] + spike
            self.activated = True
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


def main():
    print("=" * 70)
    print("  Phase 8: Layer-wise Spike Propagation")
    print("  Which layer is the best injection point?")
    print("=" * 70)

    model, tok = load_model()

    qa_pairs = [
        ("The capital of Japan is", [11790]),
        ("Water freezes at", [657]),
        ("The chemical formula for water is", [367]),
        ("The largest planet is", [22721]),
        ("DNA stands for", [390]),
    ]

    n_layers = 12  # GPT-2 has 12 transformer blocks
    spike_mag = 10  # Use the P5 transition magnitude
    layers_to_test = list(range(n_layers))

    # === Baseline (no spike) ===
    print("\n[P8a] Baseline (no injection)...")
    baseline_acc = 0
    baseline_entropies = []
    for prompt, fact_ids in qa_pairs:
        inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1).squeeze(0)
        h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
        baseline_entropies.append(h)
        if torch.argmax(probs).item() in fact_ids:
            baseline_acc += 1
    print(f"  Baseline: acc={baseline_acc}/{len(qa_pairs)}, "
          f"H={np.mean(baseline_entropies):.2f}")

    # === Layer-wise injection ===
    print(f"\n[P8b] Injecting spikes at each of {n_layers} layers (mag={spike_mag})...")
    results_by_layer = {}

    for layer_idx in layers_to_test:
        correct = 0
        entropies = []
        hidden_norms = []

        for prompt, fact_ids in qa_pairs:
            inp = tok(prompt, return_tensors='pt', truncation=True,
                      max_length=128).to(DEVICE)

            # Register hook at this layer
            hook_obj = SpikeHook(fact_ids, spike_mag, tok.vocab_size)
            block = model.transformer.h[layer_idx]
            handle = block.register_forward_hook(hook_obj)

            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            handle.remove()

            logits = out.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1).squeeze(0)
            h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
            entropies.append(h)

            # Track hidden state norm at output
            final_h = out.hidden_states[-1][:, -1, :]
            hidden_norms.append(float(torch.norm(final_h).cpu()))

            if torch.argmax(probs).item() in fact_ids:
                correct += 1

        results_by_layer[layer_idx] = {
            'accuracy': correct / len(qa_pairs),
            'mean_entropy': float(np.mean(entropies)),
            'mean_hidden_norm': float(np.mean(hidden_norms)),
        }
        print(f"  Layer {layer_idx:>2d}: acc={correct}/5, "
              f"H={np.mean(entropies):.2f}, ||h||={np.mean(hidden_norms):.1f}")

    # === Output-layer spike comparison ===
    print(f"\n[P8c] Output-layer spike (P5 method) for comparison...")
    output_acc = 0
    output_ent = []
    for prompt, fact_ids in qa_pairs:
        inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[:, -1, :]
        spiked = logits.clone()
        for tid in fact_ids:
            if tid < spiked.shape[-1]:
                spiked[..., tid] += spike_mag
        probs = F.softmax(spiked, dim=-1).squeeze(0)
        h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
        output_ent.append(h)
        if torch.argmax(probs).item() in fact_ids:
            output_acc += 1
    print(f"  Output spike: acc={output_acc}/5, H={np.mean(output_ent):.2f}")

    # === Find optimal layer ===
    best_layer = max(results_by_layer.keys(),
                     key=lambda l: (results_by_layer[l]['accuracy'],
                                    -results_by_layer[l]['mean_entropy']))
    print(f"\n  Best injection layer: {best_layer}")
    print(f"  Best acc: {results_by_layer[best_layer]['accuracy']:.0%}")

    # === Visualization ===
    print("\n[P8] Generating figures...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    layers = sorted(results_by_layer.keys())
    accs = [results_by_layer[l]['accuracy'] * 100 for l in layers]
    ents = [results_by_layer[l]['mean_entropy'] for l in layers]
    norms = [results_by_layer[l]['mean_hidden_norm'] for l in layers]

    # Plot 1: Accuracy by layer
    axes[0].bar(layers, accs, color='steelblue', alpha=0.7)
    axes[0].axhline(y=baseline_acc/len(qa_pairs)*100, color='r', linestyle='--',
                     label=f'Baseline ({baseline_acc}/5)')
    axes[0].axhline(y=output_acc/len(qa_pairs)*100, color='g', linestyle='--',
                     label=f'Output spike ({output_acc}/5)')
    axes[0].set_xlabel('Injection Layer')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Factual Accuracy by Injection Depth')
    axes[0].legend(fontsize=8)
    axes[0].set_xticks(layers)

    # Plot 2: Entropy by layer
    axes[1].plot(layers, ents, 'ro-', linewidth=2)
    axes[1].axhline(y=np.mean(baseline_entropies), color='gray', linestyle='--',
                     label='Baseline')
    axes[1].set_xlabel('Injection Layer')
    axes[1].set_ylabel('Output Entropy')
    axes[1].set_title('Entropy vs Injection Depth')
    axes[1].legend()
    axes[1].set_xticks(layers)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Hidden norm propagation
    axes[2].plot(layers, norms, 'g.-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Injection Layer')
    axes[2].set_ylabel('Final Hidden State ||h||')
    axes[2].set_title('Spike Energy Propagation')
    axes[2].set_xticks(layers)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 8: Layer-wise Spike Propagation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase8_layer_spike.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 8, 'name': 'Layer-wise Spike Propagation',
        'spike_magnitude': spike_mag,
        'baseline_accuracy': baseline_acc / len(qa_pairs),
        'output_spike_accuracy': output_acc / len(qa_pairs),
        'best_layer': best_layer,
        'best_layer_accuracy': results_by_layer[best_layer]['accuracy'],
        'by_layer': {str(k): v for k, v in results_by_layer.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase8_layer_spike.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  PHASE 8 RESULTS: Layer-wise Spike Propagation")
    print("=" * 70)
    print(f"  Best injection layer: {best_layer}")
    print(f"  Best accuracy: {results_by_layer[best_layer]['accuracy']:.0%}")
    print(f"  Output spike accuracy: {output_acc}/{len(qa_pairs)}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
