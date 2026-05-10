# -*- coding: utf-8 -*-
"""
Phase 41: Truth Vector (from SNN-Synthesis Aha! Vector)
Compute differential PCA of hidden states: fact-correct vs hallucination.
Inject the resulting "Truth Vector" to steer generation toward facts.
Cross-project: SNN-Synthesis Phase 4 -> Aletheia.
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
    print("[P41] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_hidden_states(model, tok, prompt, layer_idx=10):
    """Extract hidden state at specified layer."""
    hidden = {}
    def hook(module, args, output):
        hidden['h'] = output[0][0, -1, :].detach().cpu().numpy()
    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    logits = out.logits[:, -1, :].squeeze(0)
    return hidden['h'], logits

def main():
    print("=" * 70)
    print("  Phase 41: Truth Vector (Aha! Vector for Facts)")
    print("  Differential PCA: fact vs hallucination hidden states")
    print("=" * 70)

    model, tok = load_model()

    # Training set: known fact prompts for computing truth vector
    train_facts = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("The sun is a", [3491], "star"),
    ]
    train_hallu = [
        ("The capital of the underwater nation Atlantis is", None, "?"),
        ("The inventor of the quantum flux capacitor was", None, "?"),
        ("The winner of the 2089 Nobel Prize in Physics was", None, "?"),
        ("The color of the 5th quark flavor is", None, "?"),
    ]

    # Test set: unseen prompts
    test_prompts = [
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("DNA stands for", [390], "de"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The speed of light is approximately", [22626], "299"),
    ]

    # === P41a: Compute truth vector per layer ===
    print("\n[P41a] Computing truth vectors per layer...")
    best_layer = -1
    best_improvement = -1
    layer_results = {}

    for layer_idx in [6, 8, 10, 11]:
        # Collect hidden states
        fact_states = []
        hallu_states = []
        for prompt, _, _ in train_facts:
            h, _ = get_hidden_states(model, tok, prompt, layer_idx)
            fact_states.append(h)
        for prompt, _, _ in train_hallu:
            h, _ = get_hidden_states(model, tok, prompt, layer_idx)
            hallu_states.append(h)

        fact_mean = np.mean(fact_states, axis=0)
        hallu_mean = np.mean(hallu_states, axis=0)

        # Truth vector = fact_mean - hallu_mean (direction toward truth)
        truth_vec = fact_mean - hallu_mean
        truth_vec = truth_vec / (np.linalg.norm(truth_vec) + 1e-12)

        # Test: inject truth vector at different magnitudes
        for alpha in [0, 1, 3, 5, 10, 20]:
            correct = 0
            for prompt, fact_ids, _ in test_prompts:
                h, logits = get_hidden_states(model, tok, prompt, layer_idx)
                # Inject truth vector into hidden state and project through LM head
                h_mod = h + alpha * truth_vec
                h_tensor = torch.tensor(h_mod, dtype=torch.float32, device=DEVICE)

                # We need to re-run from this layer forward
                # Simpler: add scaled truth vector direction to logits
                tv_tensor = torch.tensor(truth_vec, dtype=torch.float32, device=DEVICE)
                # Project truth vector through LM head
                full_tv = torch.zeros(model.config.n_embd, device=DEVICE)
                full_tv[:len(tv_tensor)] = tv_tensor
                tv_logits = model.lm_head(model.transformer.ln_f(full_tv.unsqueeze(0))).squeeze(0)

                # Combine
                combined = logits + alpha * tv_logits
                if torch.argmax(combined).item() in fact_ids:
                    correct += 1

            acc = correct / len(test_prompts)
            if alpha == 5:
                if acc > best_improvement:
                    best_improvement = acc
                    best_layer = layer_idx

            if layer_idx == 10:
                print(f"  L{layer_idx} alpha={alpha:>3d}: {correct}/{len(test_prompts)} = {acc:.0%}")

        layer_results[layer_idx] = {
            'truth_vec_norm': float(np.linalg.norm(fact_mean - hallu_mean)),
            'cos_fact_hallu': float(np.dot(fact_mean, hallu_mean) /
                                   (np.linalg.norm(fact_mean) * np.linalg.norm(hallu_mean) + 1e-12)),
        }

    # === P41b: Anti-truth vector (SNN-Synthesis's Anti-Aha!) ===
    print(f"\n[P41b] Anti-truth vector (inverted direction)...")
    h_states_l10 = []
    for prompt, _, _ in train_facts:
        h, _ = get_hidden_states(model, tok, prompt, 10)
        h_states_l10.append(h)
    for prompt, _, _ in train_hallu:
        h, _ = get_hidden_states(model, tok, prompt, 10)
        h_states_l10.append(h)

    fact_mean = np.mean(h_states_l10[:len(train_facts)], axis=0)
    hallu_mean = np.mean(h_states_l10[len(train_facts):], axis=0)
    truth_vec = (fact_mean - hallu_mean)
    truth_vec = truth_vec / (np.linalg.norm(truth_vec) + 1e-12)

    anti_results = {}
    for direction, label in [(1, 'truth'), (-1, 'anti-truth')]:
        for alpha in [0, 3, 5, 10]:
            correct = 0
            for prompt, fact_ids, _ in test_prompts:
                _, logits = get_hidden_states(model, tok, prompt, 10)
                tv_t = torch.tensor(truth_vec * direction, dtype=torch.float32, device=DEVICE)
                full_tv = torch.zeros(model.config.n_embd, device=DEVICE)
                full_tv[:len(tv_t)] = tv_t
                tv_logits = model.lm_head(model.transformer.ln_f(full_tv.unsqueeze(0))).squeeze(0)
                combined = logits + alpha * tv_logits
                if torch.argmax(combined).item() in fact_ids:
                    correct += 1
            anti_results[f"{label}_a{alpha}"] = correct / len(test_prompts)
            print(f"  {label:>11s} alpha={alpha:>3d}: {correct}/{len(test_prompts)}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    alphas = [0, 1, 3, 5, 10, 20]
    for li in [6, 8, 10, 11]:
        accs = []
        for a in alphas:
            c = 0
            for prompt, fact_ids, _ in test_prompts:
                _, logits = get_hidden_states(model, tok, prompt, li)
                tv_t = torch.tensor(truth_vec, dtype=torch.float32, device=DEVICE)
                full_tv = torch.zeros(model.config.n_embd, device=DEVICE)
                full_tv[:len(tv_t)] = tv_t
                tv_logits = model.lm_head(model.transformer.ln_f(full_tv.unsqueeze(0))).squeeze(0)
                if torch.argmax(logits + a * tv_logits).item() in fact_ids:
                    c += 1
            accs.append(c / len(test_prompts) * 100)
        axes[0].plot(alphas, accs, '.-', label=f'L{li}', linewidth=2)
    axes[0].set_xlabel('Alpha (injection strength)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Truth Vector by Layer')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Truth vs Anti-truth
    dirs = ['truth', 'anti-truth']
    for d in dirs:
        vals = [anti_results.get(f"{d}_a{a}", 0)*100 for a in [0, 3, 5, 10]]
        color = 'green' if d == 'truth' else 'red'
        axes[1].plot([0, 3, 5, 10], vals, '.-', color=color, label=d, linewidth=2)
    axes[1].set_xlabel('Alpha')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Truth vs Anti-Truth Vector')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Cosine similarity
    layers = sorted(layer_results.keys())
    cos_vals = [layer_results[l]['cos_fact_hallu'] for l in layers]
    axes[2].bar([f'L{l}' for l in layers], cos_vals, color='purple', alpha=0.7)
    axes[2].set_ylabel('Cosine(fact, hallu)')
    axes[2].set_title('Fact-Hallu Similarity per Layer')

    plt.suptitle('Phase 41: Truth Vector (Aha! Vector for Facts)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase41_truth_vector.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 41, 'name': 'Truth Vector',
        'inspiration': 'SNN-Synthesis Aha! Vector (Phase 4)',
        'best_layer': best_layer,
        'layer_results': layer_results,
        'anti_results': anti_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase41_truth_vector.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 41 RESULTS: Truth Vector")
    print("=" * 70)
    print(f"  Best layer: L{best_layer}")
    for k, v in anti_results.items():
        print(f"  {k}: {v:.0%}")
    print("=" * 70)
    phase_complete(41)

if __name__ == '__main__':
    main()
