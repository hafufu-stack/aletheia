# -*- coding: utf-8 -*-
"""
Phase 6: Orthogonalization x Spike Fusion
- Does Gram-Schmidt orthogonalization LOWER the spike threshold for phase transition?
- Compare: vanilla spike vs orthogonalized-then-spike
"""
import os, json, gc
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
    print("[P6] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def get_hidden_and_logits(model, tok, text):
    inp = tok(text, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hidden = out.hidden_states[-1].mean(dim=1).squeeze(0)  # (768,)
    logits = out.logits[:, -1, :]  # (1, vocab)
    return hidden, logits


def orthogonalize_logits(logits, fact_direction, skill_direction):
    """Remove skill-direction component from logits to suppress confabulation."""
    # Project logits onto skill direction and subtract
    skill_norm = skill_direction / (torch.norm(skill_direction) + 1e-12)
    projection = torch.sum(logits * skill_norm) * skill_norm
    return logits - projection


def spike_inject(logits, fact_ids, magnitude):
    spiked = logits.clone()
    for tid in fact_ids:
        if tid < spiked.shape[-1]:
            spiked[..., tid] += magnitude
    return spiked


def main():
    print("=" * 70)
    print("  Phase 6: Orthogonalization x Spike Fusion")
    print("  Does separating Fact/Skill LOWER the spike threshold?")
    print("=" * 70)

    model, tok = load_model()

    # Build fact vs skill direction vectors
    fact_prompts = [
        "The capital of France is", "Water boils at",
        "The Earth orbits the", "DNA stands for",
        "The speed of light is approximately",
    ]
    skill_prompts = [
        "Once upon a time,", "In my opinion,",
        "The story begins with", "Let me explain why",
        "Imagine a world where",
    ]

    print("\n[P6a] Computing Fact/Skill direction vectors...")
    fact_hiddens = []
    skill_hiddens = []
    for t in fact_prompts:
        h, _ = get_hidden_and_logits(model, tok, t)
        fact_hiddens.append(h)
    for t in skill_prompts:
        h, _ = get_hidden_and_logits(model, tok, t)
        skill_hiddens.append(h)

    fact_dir = torch.stack(fact_hiddens).mean(dim=0)
    skill_dir = torch.stack(skill_hiddens).mean(dim=0)

    # Measure original angle
    cos_angle = float(F.cosine_similarity(fact_dir.unsqueeze(0),
                                           skill_dir.unsqueeze(0)))
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
    print(f"  Fact-Skill angle: {angle_deg:.1f} deg (cos={cos_angle:.4f})")

    # QA pairs
    qa_pairs = [
        ("The capital of Japan is", [11790]),
        ("Water freezes at", [657]),
        ("The chemical formula for water is", [367]),
        ("The largest planet is", [22721]),
        ("DNA stands for", [390]),
    ]

    # === Sweep: vanilla spike vs ortho+spike ===
    print("\n[P6b] Comparing vanilla spike vs ortho+spike...")
    magnitudes = [0, 1, 2, 3, 5, 7, 10, 15, 20, 50]
    vanilla_results = {}
    ortho_results = {}

    for mag in magnitudes:
        v_correct = 0
        o_correct = 0
        v_entropies = []
        o_entropies = []

        for prompt, fact_ids in qa_pairs:
            _, logits = get_hidden_and_logits(model, tok, prompt)

            # Vanilla: just spike
            spiked_v = spike_inject(logits, fact_ids, mag)
            probs_v = F.softmax(spiked_v, dim=-1).squeeze(0)
            h_v = float(-torch.sum(probs_v * torch.log(probs_v + 1e-12)).cpu())
            v_entropies.append(h_v)
            if torch.argmax(probs_v).item() in fact_ids:
                v_correct += 1

            # Ortho+spike: project out skill direction in logit space
            # Map skill_dir from hidden(768) to logit(50257) via lm_head
            with torch.no_grad():
                skill_logit = model.lm_head(skill_dir.unsqueeze(0)).squeeze(0)
            skill_n = skill_logit / (torch.norm(skill_logit) + 1e-12)
            logits_flat = logits.squeeze(0)
            proj = torch.dot(logits_flat, skill_n) * skill_n
            logits_orth = logits_flat - 0.5 * proj
            logits_orth = logits_orth.unsqueeze(0)

            spiked_o = spike_inject(logits_orth, fact_ids, mag)
            probs_o = F.softmax(spiked_o, dim=-1).squeeze(0)
            h_o = float(-torch.sum(probs_o * torch.log(probs_o + 1e-12)).cpu())
            o_entropies.append(h_o)
            if torch.argmax(probs_o).item() in fact_ids:
                o_correct += 1

        vanilla_results[mag] = {
            'accuracy': v_correct / len(qa_pairs),
            'entropy': float(np.mean(v_entropies)),
        }
        ortho_results[mag] = {
            'accuracy': o_correct / len(qa_pairs),
            'entropy': float(np.mean(o_entropies)),
        }
        print(f"  mag={mag:>3d}: vanilla={v_correct}/5 (H={np.mean(v_entropies):.2f})  "
              f"ortho+spike={o_correct}/5 (H={np.mean(o_entropies):.2f})")

    # Find transition thresholds
    v_threshold = None
    o_threshold = None
    for mag in magnitudes:
        if vanilla_results[mag]['accuracy'] >= 1.0 and v_threshold is None:
            v_threshold = mag
        if ortho_results[mag]['accuracy'] >= 1.0 and o_threshold is None:
            o_threshold = mag

    print(f"\n  Vanilla phase transition: spike={v_threshold}")
    print(f"  Ortho+spike phase transition: spike={o_threshold}")
    if v_threshold and o_threshold:
        reduction = (1 - o_threshold / v_threshold) * 100
        print(f"  Threshold reduction: {reduction:.0f}%")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    mags = sorted(vanilla_results.keys())
    v_acc = [vanilla_results[m]['accuracy'] * 100 for m in mags]
    o_acc = [ortho_results[m]['accuracy'] * 100 for m in mags]

    axes[0].plot(mags, v_acc, 'r.-', label='Vanilla Spike', linewidth=2)
    axes[0].plot(mags, o_acc, 'g.-', label='Ortho + Spike', linewidth=2)
    axes[0].set_xlabel('Spike Magnitude')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Phase Transition Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    v_ent = [vanilla_results[m]['entropy'] for m in mags]
    o_ent = [ortho_results[m]['entropy'] for m in mags]
    axes[1].plot(mags, v_ent, 'r.-', label='Vanilla', linewidth=2)
    axes[1].plot(mags, o_ent, 'g.-', label='Ortho+Spike', linewidth=2)
    axes[1].set_xlabel('Spike Magnitude')
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Entropy Collapse')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Threshold comparison bar
    thresholds = [v_threshold or 0, o_threshold or 0]
    axes[2].bar(['Vanilla', 'Ortho+Spike'], thresholds,
                color=['red', 'green'], alpha=0.7)
    axes[2].set_ylabel('Phase Transition Threshold')
    axes[2].set_title('Spike Threshold Reduction')

    plt.suptitle('Phase 6: Orthogonalization x Spike Fusion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase6_ortho_spike.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 6, 'name': 'Orthogonalization x Spike Fusion',
        'fact_skill_angle_deg': angle_deg,
        'vanilla_threshold': v_threshold,
        'ortho_threshold': o_threshold,
        'vanilla_sweep': {str(k): v for k, v in vanilla_results.items()},
        'ortho_sweep': {str(k): v for k, v in ortho_results.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'phase6_ortho_spike.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  PHASE 6 RESULTS")
    print("=" * 70)
    print(f"  Fact-Skill angle: {angle_deg:.1f} deg")
    print(f"  Vanilla threshold: {v_threshold}")
    print(f"  Ortho+Spike threshold: {o_threshold}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
