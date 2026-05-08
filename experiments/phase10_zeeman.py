# -*- coding: utf-8 -*-
"""
Phase 10: Semantic Zeeman Effect
- Apply rotation matrices to force Fact/Skill separation at each layer
- Like a magnetic field splitting degenerate energy levels
- Measure angle expansion from 1.2 deg toward orthogonality
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
    print("[P10] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def get_hidden(model, tok, text, layer=12):
    inp = tok(text, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.hidden_states[layer].mean(dim=1).squeeze(0)


class ZeemanHook:
    """Apply rotation in the Fact-Skill plane to separate them."""
    def __init__(self, fact_dir, skill_dir, angle_rad):
        self.fact_dir = F.normalize(fact_dir, dim=0)
        self.skill_dir = skill_dir
        # Orthogonalize skill w.r.t. fact to get the 2D plane
        proj = torch.dot(self.skill_dir, self.fact_dir) * self.fact_dir
        self.orth_dir = F.normalize(self.skill_dir - proj, dim=0)
        self.angle = angle_rad

    def __call__(self, module, input, output):
        hidden = output[0].clone()
        h = hidden[:, -1, :]  # (1, d)
        # Project onto Fact-Skill plane
        c_fact = torch.sum(h * self.fact_dir)
        c_orth = torch.sum(h * self.orth_dir)
        # Rotate by angle in this plane
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        new_fact = c_fact * cos_a - c_orth * sin_a
        new_orth = c_fact * sin_a + c_orth * cos_a
        # Apply rotation
        delta = (new_fact - c_fact) * self.fact_dir + (new_orth - c_orth) * self.orth_dir
        hidden[:, -1, :] = h + delta
        return (hidden,) + output[1:]


def main():
    print("=" * 70)
    print("  Phase 10: Semantic Zeeman Effect")
    print("  Splitting Fact-Skill Degeneracy with Rotation Fields")
    print("=" * 70)

    model, tok = load_model()

    # Compute Fact and Skill directions
    fact_texts = ["The capital of France is", "Water boils at",
                  "DNA stands for", "The speed of light is",
                  "The Earth orbits the"]
    skill_texts = ["Once upon a time,", "In my opinion,",
                   "The story begins with", "Let me explain why",
                   "Imagine a world where"]

    fact_vecs = torch.stack([get_hidden(model, tok, t) for t in fact_texts])
    skill_vecs = torch.stack([get_hidden(model, tok, t) for t in skill_texts])
    fact_dir = fact_vecs.mean(dim=0)
    skill_dir = skill_vecs.mean(dim=0)

    cos_orig = float(F.cosine_similarity(fact_dir.unsqueeze(0), skill_dir.unsqueeze(0)))
    angle_orig = float(np.degrees(np.arccos(np.clip(cos_orig, -1, 1))))
    print(f"  Original Fact-Skill angle: {angle_orig:.2f} deg")

    qa_pairs = [
        ("The capital of Japan is", [11790]),
        ("Water freezes at", [657]),
        ("The chemical formula for water is", [367]),
        ("The largest planet is", [22721]),
        ("DNA stands for", [390]),
    ]

    # === Sweep rotation angles ===
    rotation_angles_deg = [0, 5, 10, 15, 20, 30, 45, 60, 90]
    layers_to_apply = [0, 3, 6, 9, 11]  # Apply Zeeman field at multiple layers

    print(f"\n[P10a] Sweeping rotation angles (field applied at layers {layers_to_apply})...")
    results_by_angle = {}

    for rot_deg in rotation_angles_deg:
        rot_rad = np.radians(rot_deg)
        correct = 0
        entropies = []
        measured_angles = []

        for prompt, fact_ids in qa_pairs:
            inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            handles = []
            for li in layers_to_apply:
                hook = ZeemanHook(fact_dir.clone(), skill_dir.clone(), rot_rad)
                h = model.transformer.h[li].register_forward_hook(hook)
                handles.append(h)

            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            for h in handles:
                h.remove()

            logits = out.logits[:, -1, :].squeeze(0)
            probs = F.softmax(logits, dim=-1)
            entropy = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
            entropies.append(entropy)

            if torch.argmax(probs).item() in fact_ids:
                correct += 1

            # Measure resulting angle
            h_fact = out.hidden_states[-1].mean(dim=1).squeeze(0)
            # Re-extract skill direction after rotation
            inp_skill = tok(skill_texts[0], return_tensors='pt',
                            truncation=True, max_length=128).to(DEVICE)
            handles2 = []
            for li in layers_to_apply:
                hook2 = ZeemanHook(fact_dir.clone(), skill_dir.clone(), rot_rad)
                h2 = model.transformer.h[li].register_forward_hook(hook2)
                handles2.append(h2)
            with torch.no_grad():
                out_s = model(**inp_skill, output_hidden_states=True)
            for h2 in handles2:
                h2.remove()
            h_skill = out_s.hidden_states[-1].mean(dim=1).squeeze(0)

            cos_new = float(F.cosine_similarity(h_fact.unsqueeze(0), h_skill.unsqueeze(0)))
            measured_angles.append(float(np.degrees(np.arccos(np.clip(cos_new, -1, 1)))))

        results_by_angle[rot_deg] = {
            'accuracy': correct / len(qa_pairs),
            'mean_entropy': float(np.mean(entropies)),
            'mean_measured_angle': float(np.mean(measured_angles)),
        }
        print(f"  rot={rot_deg:>3d} deg: acc={correct}/5, "
              f"H={np.mean(entropies):.2f}, "
              f"measured angle={np.mean(measured_angles):.1f} deg")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rots = sorted(results_by_angle.keys())
    accs = [results_by_angle[r]['accuracy']*100 for r in rots]
    ents = [results_by_angle[r]['mean_entropy'] for r in rots]
    angs = [results_by_angle[r]['mean_measured_angle'] for r in rots]

    axes[0].plot(rots, accs, 'go-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Applied Rotation (degrees)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Zeeman Splitting: Accuracy')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rots, angs, 'bo-', linewidth=2, markersize=8)
    axes[1].axhline(y=angle_orig, color='r', linestyle='--',
                     label=f'Original ({angle_orig:.1f} deg)')
    axes[1].axhline(y=90, color='g', linestyle='--', alpha=0.5, label='Orthogonal')
    axes[1].set_xlabel('Applied Rotation (degrees)')
    axes[1].set_ylabel('Measured F-S Angle (degrees)')
    axes[1].set_title('Degeneracy Breaking')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(rots, ents, 'r.-', linewidth=2)
    axes[2].set_xlabel('Applied Rotation (degrees)')
    axes[2].set_ylabel('Output Entropy')
    axes[2].set_title('Entropy Response to Field')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 10: Semantic Zeeman Effect', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase10_zeeman.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 10, 'name': 'Semantic Zeeman Effect',
        'original_angle_deg': angle_orig,
        'by_rotation': {str(k): v for k, v in results_by_angle.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase10_zeeman.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  PHASE 10 RESULTS: Semantic Zeeman Effect")
    print("=" * 70)
    print(f"  Original angle: {angle_orig:.2f} deg")
    for r in [0, 30, 60, 90]:
        if r in results_by_angle:
            a = results_by_angle[r]
            print(f"  rot={r} deg: measured={a['mean_measured_angle']:.1f} deg, acc={a['accuracy']:.0%}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
