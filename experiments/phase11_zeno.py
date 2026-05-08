# -*- coding: utf-8 -*-
"""
Phase 11: Quantum Zeno Effect - Continuous Observation Freezes Lies
- Inject micro-spikes at ALL layers simultaneously
- Test if continuous weak observation prevents state transition to hallucination
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
    print("[P11] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


class ZenoHook:
    """Micro-spike at every layer (continuous observation)."""
    def __init__(self, direction, magnitude):
        self.direction = direction
        self.magnitude = magnitude

    def __call__(self, module, input, output):
        hidden = output[0].clone()
        if self.magnitude > 0:
            spike = self.direction.to(hidden.device) * self.magnitude
            hidden[:, -1, :] = hidden[:, -1, :] + spike
        return (hidden,) + output[1:]


def main():
    print("=" * 70)
    print("  Phase 11: Quantum Zeno Effect")
    print("  Continuous Observation Freezes Hallucination")
    print("=" * 70)

    model, tok = load_model()

    # Compute fact direction
    fact_texts = ["The capital of France is", "Water boils at",
                  "DNA stands for", "The speed of light is"]
    fact_vecs = []
    for t in fact_texts:
        inp = tok(t, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        fact_vecs.append(out.hidden_states[-1].mean(dim=1).squeeze(0))
    fact_dir = F.normalize(torch.stack(fact_vecs).mean(dim=0), dim=0)

    qa_pairs = [
        ("The capital of Japan is", [11790]),
        ("Water freezes at", [657]),
        ("The chemical formula for water is", [367]),
        ("The largest planet is", [22721]),
        ("DNA stands for", [390]),
    ]

    n_layers = 12

    # === Comparison: single big spike vs distributed micro-spikes ===
    # Total energy budget = magnitude * n_injection_points
    total_budgets = [0, 1, 3, 5, 10, 20, 30, 50, 100]

    print("\n[P11a] Comparing: 1 big spike (output) vs 12 micro-spikes (Zeno)...")
    single_results = {}  # All budget at output layer
    zeno_results = {}    # Budget spread across all 12 layers

    for budget in total_budgets:
        # --- Single spike at output logits ---
        single_correct = 0
        single_ent = []
        for prompt, fact_ids in qa_pairs:
            inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[:, -1, :].clone()
            for tid in fact_ids:
                if tid < logits.shape[-1]:
                    logits[..., tid] += budget
            probs = F.softmax(logits, dim=-1).squeeze(0)
            h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
            single_ent.append(h)
            if torch.argmax(probs).item() in fact_ids:
                single_correct += 1

        single_results[budget] = {'acc': single_correct/5, 'H': float(np.mean(single_ent))}

        # --- Zeno: micro-spikes at all 12 layers ---
        micro_mag = budget / n_layers  # Distribute budget equally
        zeno_correct = 0
        zeno_ent = []

        for prompt, fact_ids in qa_pairs:
            inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            handles = []
            for li in range(n_layers):
                hook = ZenoHook(fact_dir.clone(), micro_mag)
                h = model.transformer.h[li].register_forward_hook(hook)
                handles.append(h)

            with torch.no_grad():
                out = model(**inp)

            for h in handles:
                h.remove()

            logits = out.logits[:, -1, :].squeeze(0)
            probs = F.softmax(logits, dim=-1)
            entropy = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
            zeno_ent.append(entropy)
            if torch.argmax(probs).item() in fact_ids:
                zeno_correct += 1

        zeno_results[budget] = {'acc': zeno_correct/5, 'H': float(np.mean(zeno_ent))}

        print(f"  budget={budget:>4d}: single={single_correct}/5  "
              f"zeno(12x{micro_mag:.1f})={zeno_correct}/5")

    # === Zeno depth sweep: how many layers of observation? ===
    print("\n[P11b] Zeno depth sweep (budget=50)...")
    budget = 50
    depth_results = {}
    for n_obs_layers in [1, 2, 3, 4, 6, 8, 10, 12]:
        micro = budget / n_obs_layers
        correct = 0
        for prompt, fact_ids in qa_pairs:
            inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            handles = []
            # Spread observation across first n_obs_layers
            step = max(1, n_layers // n_obs_layers)
            for li in range(0, min(n_layers, n_obs_layers * step), step):
                hook = ZenoHook(fact_dir.clone(), micro)
                h = model.transformer.h[li].register_forward_hook(hook)
                handles.append(h)
            with torch.no_grad():
                out = model(**inp)
            for h in handles:
                h.remove()
            logits = out.logits[:, -1, :].squeeze(0)
            if torch.argmax(F.softmax(logits, dim=-1)).item() in fact_ids:
                correct += 1
        depth_results[n_obs_layers] = correct / 5
        print(f"  {n_obs_layers} layers x {micro:.1f} mag = {correct}/5")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    budgets = sorted(single_results.keys())
    s_acc = [single_results[b]['acc']*100 for b in budgets]
    z_acc = [zeno_results[b]['acc']*100 for b in budgets]

    axes[0].plot(budgets, s_acc, 'r.-', label='Single (output)', linewidth=2)
    axes[0].plot(budgets, z_acc, 'g.-', label='Zeno (all layers)', linewidth=2)
    axes[0].set_xlabel('Total Energy Budget')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Single Spike vs Quantum Zeno')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    s_ent = [single_results[b]['H'] for b in budgets]
    z_ent = [zeno_results[b]['H'] for b in budgets]
    axes[1].plot(budgets, s_ent, 'r.-', label='Single', linewidth=2)
    axes[1].plot(budgets, z_ent, 'g.-', label='Zeno', linewidth=2)
    axes[1].set_xlabel('Total Energy Budget')
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Entropy: Concentrated vs Distributed')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    depths = sorted(depth_results.keys())
    d_acc = [depth_results[d]*100 for d in depths]
    axes[2].plot(depths, d_acc, 'bo-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Observation Layers')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Zeno Depth (budget=50)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 11: Quantum Zeno Fact-Fixing', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase11_zeno.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 11, 'name': 'Quantum Zeno Fact-Fixing',
        'single_vs_zeno': {str(k): {'single': single_results[k], 'zeno': zeno_results[k]}
                           for k in budgets},
        'depth_sweep': {str(k): v for k, v in depth_results.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase11_zeno.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  PHASE 11 RESULTS: Quantum Zeno Effect")
    print("=" * 70)
    print(f"  Single spike (budget=10): {single_results[10]['acc']:.0%}")
    print(f"  Zeno 12-layer (budget=10): {zeno_results[10]['acc']:.0%}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
