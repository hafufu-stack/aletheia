# -*- coding: utf-8 -*-
"""
Phase 171: Multi-Signal Epistemic Oracle
P169 showed entropy alone gives only 1.29x separation (weak).

Combine MULTIPLE uncertainty signals:
  1. Entropy (P169 baseline)
  2. L2 norm of last hidden state (P118: SwiGLU Oracle, AUC=0.83)
  3. Top-k logit gap (difference between #1 and #2 prediction)
  4. Number of "competitive" tokens (logits within 90% of max)

Goal: Find a composite signal that PERFECTLY separates known from unknown.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Known facts
KNOWN = [
    "# The capital of Japan is",
    "# The capital of France is",
    "# The capital of Germany is",
    "# The largest planet is",
    "# Water freezes at",
    "# The boiling point of water is",
    "# A year has",
    "# The number of continents is",
    "# The speed of light is approximately",
    "# Pi is approximately",
    "# The chemical symbol for water is",
    "# The Earth orbits the",
]

# Unknown/impossible facts
UNKNOWN = [
    "# The capital of Xylandia is",
    "# The population of Mars in 2025 was",
    "# The 937th digit of pi is",
    "# The winner of the 2030 World Cup is",
    "# The secret code of the universe is",
    "# The phone number of the president is",
    "# The name of the first alien is",
    "# The GDP of Atlantis is",
    "# The color of dark matter is",
    "# The taste of quantum foam is",
    "# The email of God is",
    "# The height of a dream in meters is",
]


def extract_signals(model, tok, prompts, label):
    """Extract multiple uncertainty signals from each prompt."""
    n_layers = model.config.num_hidden_layers
    records = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)

        # Hook to capture last hidden state
        hidden_state = {}
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_state['h'] = output[0][:, -1, :].detach().float()
            else:
                hidden_state['h'] = output[:, -1, :].detach().float()
        handle = model.model.layers[-1].register_forward_hook(hook_fn)

        with torch.no_grad():
            outputs = model(**inp)
            logits = outputs.logits[0, -1, :].float()

        handle.remove()

        # Signal 1: Entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        # Signal 2: L2 norm of last hidden state
        l2_norm = hidden_state['h'].norm().item() if 'h' in hidden_state else 0

        # Signal 3: Top-k gap (logit #1 - logit #2)
        top2 = torch.topk(logits, 2)
        top_gap = (top2.values[0] - top2.values[1]).item()

        # Signal 4: Number of competitive tokens (within 90% of max logit)
        max_logit = logits.max().item()
        threshold = max_logit * 0.9 if max_logit > 0 else max_logit * 1.1
        n_competitive = (logits > threshold).sum().item()

        # Signal 5: Max probability (confidence)
        max_prob = probs.max().item()

        records.append({
            'prompt': prompt[:40], 'label': label,
            'entropy': entropy, 'l2_norm': l2_norm,
            'top_gap': top_gap, 'n_competitive': n_competitive,
            'max_prob': max_prob
        })

    return records


def main():
    print("[P171] Multi-Signal Epistemic Oracle")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)

    # Extract signals
    print("\n  Extracting signals...")
    known_records = extract_signals(model, tok, KNOWN, 'known')
    unknown_records = extract_signals(model, tok, UNKNOWN, 'unknown')
    all_records = known_records + unknown_records

    # Compute AUC for each signal
    labels = [1 if r['label'] == 'unknown' else 0 for r in all_records]
    signal_names = ['entropy', 'l2_norm', 'top_gap', 'n_competitive', 'max_prob']
    aucs = {}

    print("\n  === Individual Signal AUCs ===")
    for sig in signal_names:
        values = [r[sig] for r in all_records]
        # For top_gap and max_prob, LOWER = more uncertain, so negate
        if sig in ['top_gap', 'max_prob']:
            neg_values = [-v for v in values]
            try:
                auc = roc_auc_score(labels, neg_values)
            except ValueError:
                auc = 0.5
        else:
            try:
                auc = roc_auc_score(labels, values)
            except ValueError:
                auc = 0.5
        aucs[sig] = auc

        known_vals = [r[sig] for r in known_records]
        unknown_vals = [r[sig] for r in unknown_records]
        print(f"    {sig:16s}: AUC={auc:.3f}  "
              f"known={np.mean(known_vals):.3f}+/-{np.std(known_vals):.3f}  "
              f"unknown={np.mean(unknown_vals):.3f}+/-{np.std(unknown_vals):.3f}")

    # Try composite signals
    print("\n  === Composite Signals ===")
    composites = {}

    # Composite 1: entropy + l2_norm (normalized)
    for w_ent in [0.3, 0.5, 0.7]:
        w_l2 = 1 - w_ent
        values_e = np.array([r['entropy'] for r in all_records])
        values_l = np.array([r['l2_norm'] for r in all_records])
        # Normalize
        values_e = (values_e - values_e.mean()) / max(values_e.std(), 1e-8)
        values_l = (values_l - values_l.mean()) / max(values_l.std(), 1e-8)
        composite = w_ent * values_e + w_l2 * values_l
        try:
            auc = roc_auc_score(labels, composite)
        except ValueError:
            auc = 0.5
        key = f'ent{w_ent:.1f}_l2{w_l2:.1f}'
        composites[key] = auc
        print(f"    {key}: AUC={auc:.3f}")

    # Composite 2: entropy - top_gap (high entropy + low gap = uncertain)
    values_e = np.array([r['entropy'] for r in all_records])
    values_g = np.array([r['top_gap'] for r in all_records])
    values_e = (values_e - values_e.mean()) / max(values_e.std(), 1e-8)
    values_g = (values_g - values_g.mean()) / max(values_g.std(), 1e-8)
    composite2 = values_e - values_g
    try:
        auc2 = roc_auc_score(labels, composite2)
    except ValueError:
        auc2 = 0.5
    composites['ent_minus_gap'] = auc2
    print(f"    ent_minus_gap: AUC={auc2:.3f}")

    # Composite 3: all signals combined
    all_sigs = np.column_stack([
        (np.array([r['entropy'] for r in all_records]) - np.mean([r['entropy'] for r in all_records])) / max(np.std([r['entropy'] for r in all_records]), 1e-8),
        (np.array([r['l2_norm'] for r in all_records]) - np.mean([r['l2_norm'] for r in all_records])) / max(np.std([r['l2_norm'] for r in all_records]), 1e-8),
        -(np.array([r['top_gap'] for r in all_records]) - np.mean([r['top_gap'] for r in all_records])) / max(np.std([r['top_gap'] for r in all_records]), 1e-8),
        (np.array([r['n_competitive'] for r in all_records]) - np.mean([r['n_competitive'] for r in all_records])) / max(np.std([r['n_competitive'] for r in all_records]), 1e-8),
    ])
    composite_all = all_sigs.sum(axis=1)
    try:
        auc_all = roc_auc_score(labels, composite_all)
    except ValueError:
        auc_all = 0.5
    composites['all_combined'] = auc_all
    print(f"    all_combined: AUC={auc_all:.3f}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase171_oracle.json'), 'w') as f:
        json.dump({'phase': '171', 'name': 'Multi-Signal Epistemic Oracle',
                   'individual_aucs': aucs, 'composite_aucs': composites,
                   'records': all_records}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: Individual AUCs
    ax = axes[0]
    names = list(aucs.keys())
    vals = [aucs[n] for n in names]
    colors = ['#e74c3c' if v > 0.7 else '#f39c12' if v > 0.6 else '#3498db' for v in vals]
    bars = ax.bar(names, vals, color=colors, alpha=0.8)
    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5, label='Random')
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Individual Signal AUCs', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Middle: Signal distributions
    ax = axes[1]
    best_sig = max(aucs, key=aucs.get)
    known_vals = [r[best_sig] for r in known_records]
    unknown_vals = [r[best_sig] for r in unknown_records]
    ax.hist(known_vals, bins=8, alpha=0.7, color='#2ecc71', label='Known', density=True)
    ax.hist(unknown_vals, bins=8, alpha=0.7, color='#e74c3c', label='Unknown', density=True)
    ax.set_xlabel(f'Best signal: {best_sig}', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_title(f'Best Signal Distribution (AUC={aucs[best_sig]:.3f})',
                 fontsize=13, fontweight='bold')

    # Right: Composite AUCs
    ax = axes[2]
    comp_names = list(composites.keys())
    comp_vals = [composites[n] for n in comp_names]
    colors2 = ['#e74c3c' if v > 0.7 else '#f39c12' if v > 0.6 else '#3498db' for v in comp_vals]
    ax.barh(comp_names, comp_vals, color=colors2, alpha=0.8)
    ax.axvline(x=0.5, color='gray', ls='--', alpha=0.5)
    ax.axvline(x=aucs[best_sig], color='green', ls=':', alpha=0.7,
               label=f'Best individual ({aucs[best_sig]:.3f})')
    for i, v in enumerate(comp_vals):
        ax.text(v+0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('AUC', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_title('Composite Signal AUCs', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.1)

    plt.suptitle('Phase 171: Multi-Signal Epistemic Oracle\n'
                 'Can combining uncertainty signals beat entropy alone?',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase171_oracle.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    best_ind = max(aucs, key=aucs.get)
    best_comp = max(composites, key=composites.get)
    print(f"  -> Best individual: {best_ind} (AUC={aucs[best_ind]:.3f})")
    print(f"  -> Best composite:  {best_comp} (AUC={composites[best_comp]:.3f})")
    if composites[best_comp] > aucs[best_ind] + 0.03:
        print(f"  -> COMPOSITE WINS by {composites[best_comp]-aucs[best_ind]:.3f}!")
    else:
        print(f"  -> Individual signal is sufficient")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 171] Complete.")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
