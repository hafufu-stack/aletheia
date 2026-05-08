# -*- coding: utf-8 -*-
"""
Phase 2: Epistemic Event Horizon
- Create "I don't know" attractor in latent space void regions
- Measure gravitational pull toward abstention for unknown prompts
- Compare known vs unknown prompt trajectories
"""
import os, json, gc
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
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
    print("[P2] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def get_hidden(model, tok, text, layer=12):
    inp = tok(text, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.hidden_states[layer].mean(dim=1).squeeze(0).cpu().numpy()


def get_entropy(model, tok, text):
    """Get output entropy (uncertainty measure)."""
    inp = tok(text, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    return float(entropy.cpu())


def main():
    print("=" * 70)
    print("  Phase 2: The Epistemic Event Horizon")
    print("  Gravitational Collapse toward 'I Don't Know'")
    print("=" * 70)

    model, tok = load_model()

    # Known facts (model should know these)
    known = [
        "The capital of France is",
        "Water boils at 100 degrees",
        "The sun is a star",
        "Python is a programming language",
        "Shakespeare wrote Romeo and Juliet",
        "The Earth revolves around the sun",
        "Gravity pulls objects downward",
        "DNA contains genetic information",
        "The speed of light is very fast",
        "Mathematics deals with numbers",
    ]

    # Unknown/nonsensical (model should NOT know)
    unknown = [
        "The capital of Zorbistan is",
        "The melting point of glorbium is",
        "Professor Xylophone discovered the law of",
        "The population of Narnia according to the 2025 census is",
        "The chemical formula for unobtanium is",
        "The president of the underwater kingdom declared",
        "The third moon of Kepler-442b is named",
        "In the year 3000, the dominant species will be",
        "The quantum flavor of dark chocolate is",
        "The GDP of Wakanda in 2024 was",
    ]

    # "I don't know" anchors
    idk_texts = [
        "I don't know the answer to that question",
        "I'm not sure about this",
        "This information is unknown to me",
        "I cannot verify this claim",
        "I don't have enough information",
    ]

    # === Extract representations ===
    print("\n[P2a] Extracting hidden states...")
    known_vecs = np.array([get_hidden(model, tok, t) for t in known])
    unknown_vecs = np.array([get_hidden(model, tok, t) for t in unknown])
    idk_vecs = np.array([get_hidden(model, tok, t) for t in idk_texts])

    # IDK centroid = the "black hole" center
    idk_centroid = idk_vecs.mean(axis=0)

    # === Measure distances to IDK black hole ===
    print("\n[P2b] Measuring distances to epistemic black hole...")
    known_dists = np.linalg.norm(known_vecs - idk_centroid, axis=1)
    unknown_dists = np.linalg.norm(unknown_vecs - idk_centroid, axis=1)

    # Cosine similarity to IDK
    known_cos = cosine_similarity(known_vecs, idk_centroid.reshape(1, -1)).flatten()
    unknown_cos = cosine_similarity(unknown_vecs, idk_centroid.reshape(1, -1)).flatten()

    # === Entropy analysis ===
    print("\n[P2c] Measuring output entropy (uncertainty)...")
    known_entropy = [get_entropy(model, tok, t) for t in known]
    unknown_entropy = [get_entropy(model, tok, t) for t in unknown]

    print(f"  Known prompts:   mean entropy = {np.mean(known_entropy):.2f}")
    print(f"  Unknown prompts: mean entropy = {np.mean(unknown_entropy):.2f}")
    print(f"  Entropy ratio (unknown/known): {np.mean(unknown_entropy)/np.mean(known_entropy):.2f}x")

    # === Gravitational potential ===
    print("\n[P2d] Computing gravitational potential field...")
    # G * M / r model: how strongly does the IDK attractor pull?
    idk_mass = np.linalg.norm(idk_centroid)  # "mass" of IDK attractor
    known_potential = idk_mass / (known_dists + 1e-12)
    unknown_potential = idk_mass / (unknown_dists + 1e-12)

    event_horizon_radius = np.mean(unknown_dists) * 0.5
    captured_known = np.sum(known_dists < event_horizon_radius)
    captured_unknown = np.sum(unknown_dists < event_horizon_radius)

    print(f"  IDK attractor mass: {idk_mass:.2f}")
    print(f"  Event horizon radius: {event_horizon_radius:.2f}")
    print(f"  Known captured: {captured_known}/{len(known)}")
    print(f"  Unknown captured: {captured_unknown}/{len(unknown)}")

    # === Visualization ===
    print("\n[P2] Generating figures...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Distance to IDK
    axes[0].bar(['Known', 'Unknown'], [np.mean(known_dists), np.mean(unknown_dists)],
                color=['green', 'red'], alpha=0.7, yerr=[np.std(known_dists), np.std(unknown_dists)])
    axes[0].set_ylabel('Distance to IDK Attractor')
    axes[0].set_title('Epistemic Distance')
    axes[0].axhline(y=event_horizon_radius, color='orange', linestyle='--', label='Event Horizon')
    axes[0].legend()

    # Plot 2: Entropy comparison
    axes[1].boxplot([known_entropy, unknown_entropy], labels=['Known', 'Unknown'])
    axes[1].set_ylabel('Output Entropy (nats)')
    axes[1].set_title('Uncertainty: Known vs Unknown')

    # Plot 3: Gravitational potential
    x = range(len(known))
    axes[2].scatter(x, known_potential, c='green', label='Known', s=60, alpha=0.7)
    axes[2].scatter(x, unknown_potential[:len(x)], c='red', label='Unknown', s=60, alpha=0.7)
    axes[2].set_xlabel('Prompt index')
    axes[2].set_ylabel('Gravitational Potential (M/r)')
    axes[2].set_title('IDK Gravitational Pull')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 2: The Epistemic Event Horizon', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase2_event_horizon.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # === Results ===
    separation = float(np.mean(unknown_cos) - np.mean(known_cos))
    results = {
        'phase': 2, 'name': 'Epistemic Event Horizon',
        'known_mean_dist': float(np.mean(known_dists)),
        'unknown_mean_dist': float(np.mean(unknown_dists)),
        'known_mean_cos_idk': float(np.mean(known_cos)),
        'unknown_mean_cos_idk': float(np.mean(unknown_cos)),
        'separation': separation,
        'known_entropy': float(np.mean(known_entropy)),
        'unknown_entropy': float(np.mean(unknown_entropy)),
        'entropy_ratio': float(np.mean(unknown_entropy) / np.mean(known_entropy)),
        'event_horizon_radius': float(event_horizon_radius),
        'captured_known': int(captured_known),
        'captured_unknown': int(captured_unknown),
        'idk_mass': float(idk_mass),
    }

    with open(os.path.join(RESULTS_DIR, 'phase2_event_horizon.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  PHASE 2 RESULTS: Epistemic Event Horizon")
    print("=" * 70)
    print(f"  Known dist to IDK:   {np.mean(known_dists):.2f}")
    print(f"  Unknown dist to IDK: {np.mean(unknown_dists):.2f}")
    print(f"  Entropy ratio: {results['entropy_ratio']:.2f}x")
    print(f"  Captured by event horizon: known={captured_known}, unknown={captured_unknown}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
