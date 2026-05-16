# -*- coding: utf-8 -*-
"""
Phase 142: Thermodynamic Persistence
Does continued training reverse Embedding Surgery?

After surgery, number embeddings have low cosine (~0.05).
If we continue next-token prediction training on generic text,
does the model's loss function "heal the wound" and push
embeddings back to their original clustered state?

Uses public domain text (simple English sentences, no copyrighted material).

Model: Qwen2.5-0.5B (GPU, float32)
"""
import torch, json, os, gc, numpy as np, time
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_TOKENS = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]

# Generic training text - simple factual sentences (no copyrighted material)
TRAINING_TEXTS = [
    "The sun rises in the east and sets in the west every day.",
    "Mountains are formed by the movement of tectonic plates over millions of years.",
    "Rivers flow from higher elevations to lower elevations due to gravity.",
    "Trees produce oxygen through the process of photosynthesis.",
    "The moon orbits the earth approximately once every month.",
    "Sound travels faster through water than through air.",
    "Ice is less dense than liquid water, which is why it floats.",
    "Lightning is caused by the buildup of electrical charge in clouds.",
    "Earthquakes occur along fault lines where tectonic plates meet.",
    "Rain forms when water vapor in clouds condenses into droplets.",
    "The ocean covers about seventy percent of the surface of the earth.",
    "Volcanoes release molten rock from deep within the planet.",
    "Stars produce energy through nuclear fusion of hydrogen atoms.",
    "Coral reefs are built by tiny marine organisms over many years.",
    "Glaciers are massive bodies of ice that move slowly across land.",
    "The atmosphere protects life on earth from harmful radiation.",
    "Fossils are preserved remains of ancient living organisms.",
    "Hurricanes form over warm ocean waters near the equator.",
    "Metals conduct electricity because their electrons move freely.",
    "Gravity is the force that attracts objects toward each other.",
]


def disperse_embeddings(model, tok, strength=1.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366"]
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in num_tokens]
    vecs = embed[ids].clone()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()


def measure_geometry(model, tok):
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in NUM_TOKENS]
    vecs = embed[ids].float()
    # Cosine similarity
    norms = vecs.norm(dim=-1, keepdim=True)
    cos_mat = (vecs @ vecs.T) / (norms @ norms.T + 1e-8)
    mask = ~torch.eye(len(ids), dtype=bool, device=cos_mat.device)
    cos_val = cos_mat[mask].mean().item()
    # L2 distance
    dists = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0)).squeeze(0)
    l2 = dists[mask].mean().item()
    # Mean norm
    mean_norm = norms.mean().item()
    return cos_val, l2, mean_norm


def train_next_token(model, tok, texts, steps):
    """Continue pre-training with next-token prediction."""
    model.train()
    # Only train embedding + first few layers (where surgery happened)
    for name, p in model.named_parameters():
        p.requires_grad = False
        if 'embed_tokens' in name:
            p.requires_grad = True
        # Also allow first 4 layers to adapt
        for i in range(4):
            if f"layers.{i}." in name:
                p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-5)

    for step in range(steps):
        text = texts[step % len(texts)]
        inp = tok(text, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        ids = inp['input_ids']
        outputs = model(**inp, labels=ids)
        loss = outputs.loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()


def main():
    print("[P142] Thermodynamic Persistence")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Step counts to measure at
    step_counts = [0, 10, 25, 50, 100, 200, 500]
    results = []

    # Measure original baseline (pre-surgery)
    model_orig = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    cos_orig, l2_orig, norm_orig = measure_geometry(model_orig, tok)
    print(f"  Original (pre-surgery): cos={cos_orig:.4f}, L2={l2_orig:.4f}")
    del model_orig; gc.collect(); torch.cuda.empty_cache()

    for steps in step_counts:
        print(f"\n  === steps = {steps} ===")
        # Fresh model + surgery each time (for independent measurements)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
        disperse_embeddings(model, tok, strength=1.0)

        if steps > 0:
            train_next_token(model, tok, TRAINING_TEXTS, steps)

        cos, l2, norm = measure_geometry(model, tok)
        # How much has it reverted toward original?
        reversion = (cos - 0.05) / (cos_orig - 0.05) if (cos_orig - 0.05) > 0.01 else 0
        print(f"    cos={cos:.4f}, L2={l2:.4f}, norm={norm:.4f}, reversion={reversion:.1%}")
        results.append({
            'steps': steps, 'cos': cos, 'l2': l2, 'norm': norm,
            'reversion': reversion
        })
        del model; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase142_persistence.json'), 'w') as f:
        json.dump({
            'phase': '142', 'name': 'Thermodynamic Persistence',
            'original_cos': cos_orig, 'original_l2': l2_orig,
            'results': results
        }, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    steps_vals = [r['steps'] for r in results]
    cos_vals = [r['cos'] for r in results]
    l2_vals = [r['l2'] for r in results]
    rev_vals = [r['reversion'] for r in results]

    # Left: Cosine similarity over training steps
    ax = axes[0]
    ax.plot(steps_vals, cos_vals, 'r-o', lw=2.5, markersize=8, label='After training')
    ax.axhline(y=cos_orig, color='orange', ls='--', lw=2, alpha=0.7,
              label=f'Original (pre-surgery): {cos_orig:.3f}')
    ax.axhline(y=cos_vals[0], color='green', ls=':', lw=2, alpha=0.7,
              label=f'Post-surgery: {cos_vals[0]:.3f}')
    ax.set_xlabel('Continued Training Steps', fontsize=12)
    ax.set_ylabel('Mean Pairwise Cosine', fontsize=12)
    ax.set_title('Embedding Cosine Recovery', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Center: L2 distance over training steps
    ax = axes[1]
    ax.plot(steps_vals, l2_vals, 'b-o', lw=2.5, markersize=8)
    ax.axhline(y=l2_orig, color='orange', ls='--', lw=2, alpha=0.7,
              label=f'Original: {l2_orig:.3f}')
    ax.axhline(y=1.25, color='red', ls=':', lw=2, alpha=0.7, label='L2* = 1.25')
    ax.set_xlabel('Continued Training Steps', fontsize=12)
    ax.set_ylabel('Mean Pairwise L2 Distance', fontsize=12)
    ax.set_title('L2 Distance Decay', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Reversion percentage
    ax = axes[2]
    ax.plot(steps_vals, [r * 100 for r in rev_vals], 'purple', lw=2.5, marker='s', markersize=8)
    ax.fill_between(steps_vals, [r * 100 for r in rev_vals], alpha=0.15, color='purple')
    ax.set_xlabel('Continued Training Steps', fontsize=12)
    ax.set_ylabel('Reversion to Original (%)', fontsize=12)
    ax.set_title('Self-Healing Rate', fontsize=13, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 142: Thermodynamic Persistence\n'
                'Does continued pre-training reverse Embedding Surgery?',
                fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase142_persistence.png'), dpi=150,
               bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    if rev_vals[-1] > 0.5:
        print(f"  -> SELF-HEALING CONFIRMED: {rev_vals[-1]:.0%} reversion after {steps_vals[-1]} steps")
        print("     The loss function actively 'heals' the surgery wound!")
    else:
        print(f"  -> Surgery is PERSISTENT: only {rev_vals[-1]:.0%} reversion after {steps_vals[-1]} steps")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 142] Complete.")

if __name__ == '__main__':
    main()
