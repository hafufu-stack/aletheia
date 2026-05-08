# -*- coding: utf-8 -*-
"""
Phase 1: Pauli Exclusion Architecture
- Orthogonalize Fact vs Skill subspaces in GPT-2's hidden states
- Gram-Schmidt + Pauli exclusion loss
- Measure cross-talk reduction and hallucination suppression
"""
import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EXPERIMENT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_model():
    print("[P1] Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    model.to(DEVICE)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_hidden_states(model, tokenizer, texts, layer=-1):
    """Extract hidden states from a specific layer."""
    all_states = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                           max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Mean pool over sequence
        h = outputs.hidden_states[layer].mean(dim=1).squeeze(0)
        all_states.append(h.cpu().numpy())
    return np.array(all_states)


def gram_schmidt_orthogonalize(fact_vecs, skill_vecs):
    """Orthogonalize skill subspace w.r.t. fact subspace."""
    # Compute fact subspace basis via SVD
    U, S, Vt = np.linalg.svd(fact_vecs, full_matrices=False)
    k = min(10, len(S))  # top-k fact directions
    fact_basis = Vt[:k]  # (k, d)

    # Project skill vectors to be orthogonal to fact basis
    skill_orth = skill_vecs.copy()
    for basis_vec in fact_basis:
        basis_vec = basis_vec / (np.linalg.norm(basis_vec) + 1e-12)
        projections = skill_orth @ basis_vec[:, None]  # (n, 1)
        skill_orth -= projections * basis_vec[None, :]

    return fact_basis, skill_orth


def measure_crosstalk(fact_vecs, skill_vecs):
    """Measure cross-talk: mean |cos(fact, skill)|."""
    fact_norm = fact_vecs / (np.linalg.norm(fact_vecs, axis=1, keepdims=True) + 1e-12)
    skill_norm = skill_vecs / (np.linalg.norm(skill_vecs, axis=1, keepdims=True) + 1e-12)
    cos_matrix = np.abs(fact_norm @ skill_norm.T)
    return float(np.mean(cos_matrix))


def pauli_exclusion_loss(fact_vecs_t, skill_vecs_t):
    """Pauli exclusion: penalize overlap between fact and skill subspaces."""
    fact_n = F.normalize(fact_vecs_t, dim=1)
    skill_n = F.normalize(skill_vecs_t, dim=1)
    overlap = torch.abs(fact_n @ skill_n.T)
    return overlap.mean()


def generate_with_temperature(model, tokenizer, prompt, temp=1.0, max_len=50):
    """Generate text with given temperature."""
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_len, temperature=temp,
            do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    print("=" * 70)
    print("  Phase 1: Pauli Exclusion Architecture")
    print("  Orthogonalizing Fact vs Skill Subspaces")
    print("=" * 70)

    model, tokenizer = load_model()

    # === Dataset: Fact prompts vs Skill prompts ===
    fact_prompts = [
        "The capital of France is",
        "Water boils at",
        "The speed of light is approximately",
        "The Earth orbits the",
        "DNA stands for",
        "Albert Einstein developed the theory of",
        "The chemical symbol for gold is",
        "The largest planet in our solar system is",
        "The human body has approximately how many bones",
        "Photosynthesis converts sunlight into",
        "The Great Wall of China was built during the",
        "The periodic table was created by",
        "Mount Everest is located in",
        "The Pythagorean theorem states that",
        "Oxygen makes up approximately what percent of Earth's atmosphere",
        "The first person to walk on the moon was",
        "The boiling point of water in Fahrenheit is",
        "Shakespeare wrote the play",
        "The mitochondria is the powerhouse of the",
        "Pi is approximately equal to",
    ]

    skill_prompts = [
        "Once upon a time, in a land far away,",
        "The detective examined the crime scene and noticed",
        "If we consider the implications of this theory,",
        "In my opinion, the best approach would be to",
        "To solve this problem, first we need to",
        "The protagonist gazed out the window and thought",
        "Furthermore, the analysis reveals that",
        "Let me explain why this matters:",
        "The story begins with a mysterious stranger",
        "In contrast to the previous method,",
        "Imagine a world where technology has advanced to",
        "The argument can be summarized as follows:",
        "She walked through the forest, feeling",
        "The key insight here is that",
        "As a result of these findings, we can conclude",
        "The algorithm works by iteratively",
        "He couldn't believe what he saw when",
        "This suggests a fundamental shift in",
        "The recipe calls for three cups of",
        "In conclusion, the evidence strongly supports",
    ]

    # === Phase 1a: Extract hidden states ===
    print("\n[P1a] Extracting hidden states from multiple layers...")
    layers_to_probe = [1, 3, 6, 9, 12]  # GPT-2 has 12 layers
    results_by_layer = {}

    for layer_idx in layers_to_probe:
        print(f"  Layer {layer_idx}...")
        fact_vecs = extract_hidden_states(model, tokenizer, fact_prompts,
                                          layer=layer_idx)
        skill_vecs = extract_hidden_states(model, tokenizer, skill_prompts,
                                           layer=layer_idx)

        # Measure original cross-talk
        ct_before = measure_crosstalk(fact_vecs, skill_vecs)

        # Orthogonalize
        fact_basis, skill_orth = gram_schmidt_orthogonalize(fact_vecs,
                                                            skill_vecs)

        # Measure post-orthogonalization cross-talk
        ct_after = measure_crosstalk(fact_vecs, skill_orth)

        # Pauli exclusion loss
        ft = torch.tensor(fact_vecs, dtype=torch.float32)
        st = torch.tensor(skill_vecs, dtype=torch.float32)
        st_orth = torch.tensor(skill_orth, dtype=torch.float32)
        pauli_before = float(pauli_exclusion_loss(ft, st))
        pauli_after = float(pauli_exclusion_loss(ft, st_orth))

        results_by_layer[layer_idx] = {
            'crosstalk_before': ct_before,
            'crosstalk_after': ct_after,
            'reduction_pct': (1 - ct_after / max(ct_before, 1e-12)) * 100,
            'pauli_loss_before': pauli_before,
            'pauli_loss_after': pauli_after,
            'fact_basis_shape': list(fact_basis.shape),
        }
        print(f"    Cross-talk: {ct_before:.4f} -> {ct_after:.4f} "
              f"({results_by_layer[layer_idx]['reduction_pct']:.1f}% reduction)")
        print(f"    Pauli loss: {pauli_before:.4f} -> {pauli_after:.4f}")

    # === Phase 1b: Hallucination probe ===
    print("\n[P1b] Hallucination probe - factual accuracy test...")
    test_prompts = [
        ("The capital of Japan is", "Tokyo"),
        ("The chemical formula for water is", "H2O"),
        ("The year World War II ended was", "1945"),
        ("The first element on the periodic table is", "hydrogen"),
        ("The speed of sound in air is approximately", "343"),
    ]

    hallucination_scores = []
    for prompt, expected_keyword in test_prompts:
        generated = generate_with_temperature(model, tokenizer, prompt,
                                              temp=0.7, max_len=20)
        has_fact = expected_keyword.lower() in generated.lower()
        hallucination_scores.append({
            'prompt': prompt,
            'generated': generated[:100],
            'expected': expected_keyword,
            'factual': has_fact
        })
        status = "FACT" if has_fact else "HALLUCINATION"
        print(f"  [{status}] {prompt} -> {generated[:60]}...")

    factual_rate = sum(1 for s in hallucination_scores if s['factual']) / len(hallucination_scores)

    # === Phase 1c: Subspace dimensionality analysis ===
    print("\n[P1c] Subspace dimensionality analysis...")
    fact_vecs_full = extract_hidden_states(model, tokenizer, fact_prompts,
                                           layer=12)
    skill_vecs_full = extract_hidden_states(model, tokenizer, skill_prompts,
                                            layer=12)

    # SVD of each subspace
    _, s_fact, _ = np.linalg.svd(fact_vecs_full, full_matrices=False)
    _, s_skill, _ = np.linalg.svd(skill_vecs_full, full_matrices=False)

    # Effective dimensionality (90% variance)
    cum_fact = np.cumsum(s_fact ** 2) / np.sum(s_fact ** 2)
    cum_skill = np.cumsum(s_skill ** 2) / np.sum(s_skill ** 2)
    dim_fact_90 = int(np.searchsorted(cum_fact, 0.9) + 1)
    dim_skill_90 = int(np.searchsorted(cum_skill, 0.9) + 1)

    # Principal angle between subspaces (use Vt from SVD)
    _, _, Vt_fact = np.linalg.svd(fact_vecs_full, full_matrices=False)
    _, _, Vt_skill = np.linalg.svd(skill_vecs_full, full_matrices=False)
    k = min(5, Vt_fact.shape[0], Vt_skill.shape[0])
    # Principal angles via SVD of the overlap matrix
    cos_angles = np.linalg.svd(
        Vt_fact[:k] @ Vt_skill[:k].T,
        compute_uv=False
    )
    principal_angles = np.arccos(np.clip(cos_angles, -1, 1))

    print(f"  Fact subspace effective dim (90%%): {dim_fact_90}")
    print(f"  Skill subspace effective dim (90%%): {dim_skill_90}")
    print(f"  Principal angles (deg): {np.degrees(principal_angles[:3])}")

    # === Visualization ===
    print("\n[P1] Generating figures...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Cross-talk reduction by layer
    layers = sorted(results_by_layer.keys())
    ct_b = [results_by_layer[l]['crosstalk_before'] for l in layers]
    ct_a = [results_by_layer[l]['crosstalk_after'] for l in layers]
    axes[0].plot(layers, ct_b, 'ro-', label='Before (entangled)', linewidth=2)
    axes[0].plot(layers, ct_a, 'go-', label='After (orthogonal)', linewidth=2)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Cross-talk (mean |cos|)')
    axes[0].set_title('Pauli Exclusion: Cross-talk Reduction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Singular value spectra
    axes[1].semilogy(s_fact[:15], 'b.-', label='Fact subspace', linewidth=2)
    axes[1].semilogy(s_skill[:15], 'r.-', label='Skill subspace', linewidth=2)
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Singular Value')
    axes[1].set_title(f'Subspace Spectra (Fact dim={dim_fact_90}, Skill dim={dim_skill_90})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Principal angles histogram
    axes[2].bar(range(len(principal_angles[:5])),
                np.degrees(principal_angles[:5]), color='purple', alpha=0.7)
    axes[2].set_xlabel('Component pair')
    axes[2].set_ylabel('Principal Angle (degrees)')
    axes[2].set_title('Fact-Skill Principal Angles')
    axes[2].axhline(y=90, color='g', linestyle='--', label='Orthogonal (90)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 1: Pauli Exclusion Architecture', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase1_pauli_exclusion.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # === Save results ===
    results = {
        'phase': 1,
        'name': 'Pauli Exclusion Architecture',
        'layers': results_by_layer,
        'factual_accuracy': factual_rate,
        'hallucination_probes': hallucination_scores,
        'fact_dim_90': dim_fact_90,
        'skill_dim_90': dim_skill_90,
        'principal_angles_deg': np.degrees(principal_angles).tolist(),
        'mean_crosstalk_reduction_pct': np.mean(
            [r['reduction_pct'] for r in results_by_layer.values()]),
    }

    res_path = os.path.join(RESULTS_DIR, 'phase1_pauli_exclusion.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {res_path}")

    # === Summary ===
    print("\n" + "=" * 70)
    print("  PHASE 1 RESULTS: Pauli Exclusion Architecture")
    print("=" * 70)
    mean_reduction = results['mean_crosstalk_reduction_pct']
    print(f"  Mean cross-talk reduction: {mean_reduction:.1f}%")
    print(f"  Fact subspace dim: {dim_fact_90}")
    print(f"  Skill subspace dim: {dim_skill_90}")
    print(f"  Factual accuracy (baseline): {factual_rate*100:.0f}%")
    print(f"  Principal angle (1st): {np.degrees(principal_angles[0]):.1f} deg")
    print("=" * 70)

    return results


if __name__ == '__main__':
    main()
