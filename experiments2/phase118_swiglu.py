# -*- coding: utf-8 -*-
"""
Phase 118: The SwiGLU / Residual Oracle
Hypothesis: Qwen's epistemic uncertainty is encoded in MLP gate activation
and residual stream norm, NOT in attention entropy.

Measures at L_0.94:
  1. MLP gate_proj L1 norm (SwiGLU gate sparsity)
  2. Residual stream L2 norm
  3. Attention entropy (P116b baseline for comparison)
Then computes ROC/AUC for Known vs Unknown fact separation.

Also scans ALL layers to show WHERE the uncertainty signal emerges.

Models: Qwen2.5-0.5B, Qwen2.5-1.5B, GPT-2 XL (GPU)
"""
import torch, json, os, gc, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

KNOWN_FACTS = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("The chemical symbol for gold is", " Au"),
    ("The author of Romeo and Juliet is", " William"),
    ("The first president of the United States was", " George"),
    ("The chemical formula for water is", " H"),
    ("The boiling point of water is", " 100"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("The atomic number of carbon is", " 6"),
    ("The currency of the United Kingdom is the", " pound"),
]

UNKNOWN_FACTS = [
    ("The 47th prime number is", " 211"),
    ("The population of Funabashi in 2025 was approximately", " 640"),
    ("The melting point of hafnium in Kelvin is", " 2506"),
    ("The shortest-serving US president by days is", " William"),
    ("The capital of Nauru is", " Yar"),
    ("The deepest point of Lake Baikal in meters is", " 1642"),
    ("The atomic mass of Lutetium is approximately", " 175"),
    ("The ISO country code for Bhutan is", " BT"),
    ("The year the Treaty of Westphalia was signed is", " 1648"),
    ("The founder of the Mughal Empire was", " Bab"),
    ("The chemical formula for aluminum oxide is", " Al"),
    ("The speed of sound in steel in m/s is approximately", " 5960"),
    ("The orbital period of Neptune in Earth years is", " 165"),
    ("The wavelength of green light in nanometers is approximately", " 550"),
    ("The melting point of tungsten in Celsius is", " 3422"),
]


def measure_layer_signals(model, tok, prompt, layer_idx, model_type):
    """Measure gate L1 norm, residual L2 norm, and attention entropy at a layer."""
    signals = {}

    if model_type == 'qwen':
        # Hook 1: MLP gate activation
        gate_data = {}
        def mlp_hook(module, inp, out):
            x = inp[0][:, -1, :].detach().float()
            # Compute gate activation: SiLU(gate_proj(x))
            gate_out = torch.nn.functional.silu(module.gate_proj(x.to(next(module.gate_proj.parameters()).dtype)))
            gate_data['gate_l1'] = gate_out.float().abs().mean().item()
            # Also measure sparsity (fraction of near-zero activations)
            gate_data['gate_sparsity'] = (gate_out.float().abs() < 0.01).float().mean().item()

        # Hook 2: Residual stream (layer output)
        residual_data = {}
        def layer_hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
            else:
                h = out
            if h.dim() == 3:
                h = h[:, -1, :]
            elif h.dim() == 2:
                h = h[-1, :]
            h = h.detach().float()
            residual_data['residual_l2'] = h.norm(2).item()
            residual_data['residual_mean'] = h.abs().mean().item()

        mlp_handle = model.model.layers[layer_idx].mlp.register_forward_hook(mlp_hook)
        layer_handle = model.model.layers[layer_idx].register_forward_hook(layer_hook)

        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)

        mlp_handle.remove()
        layer_handle.remove()
        signals.update(gate_data)
        signals.update(residual_data)

        # Attention entropy at this layer
        hidden = {}
        def h_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.dim() == 3:
                hidden['h'] = h[:, -1, :].detach()
            else:
                hidden['h'] = h[-1, :].detach().unsqueeze(0)
        handle = model.model.layers[layer_idx].register_forward_hook(h_hook)
        with torch.no_grad():
            model(**inp)
        handle.remove()

        h = model.model.norm(hidden['h'].float())
        logits = model.lm_head(h.to(next(model.lm_head.parameters()).dtype)).squeeze()
        probs = torch.softmax(logits.float(), dim=-1)
        signals['attn_entropy'] = -(probs * torch.log(probs + 1e-10)).sum().item()

    elif model_type == 'gpt2':
        # GPT-2: no SwiGLU, but measure MLP output norm and residual
        residual_data = {}
        def layer_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.dim() == 3:
                h = h[:, -1, :]
            elif h.dim() == 2:
                h = h[-1, :]
            h = h.detach().float()
            residual_data['residual_l2'] = h.norm(2).item()
            residual_data['residual_mean'] = h.abs().mean().item()

        # MLP activation (GPT-2 uses GELU, not SwiGLU)
        mlp_data = {}
        def mlp_hook(module, inp, out):
            mlp_data['gate_l1'] = out[:, -1, :].detach().float().abs().mean().item()
            mlp_data['gate_sparsity'] = (out[:, -1, :].detach().float().abs() < 0.01).float().mean().item()

        layer_handle = model.transformer.h[layer_idx].register_forward_hook(layer_hook)
        mlp_handle = model.transformer.h[layer_idx].mlp.register_forward_hook(mlp_hook)

        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        layer_handle.remove()
        mlp_handle.remove()

        signals.update(residual_data)
        signals.update(mlp_data)

        # Attention entropy
        hidden = {}
        def h_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.dim() == 3:
                hidden['h'] = h[:, -1, :].detach()
            else:
                hidden['h'] = h[-1, :].detach().unsqueeze(0)
        handle = model.transformer.h[layer_idx].register_forward_hook(h_hook)
        with torch.no_grad():
            model(**inp)
        handle.remove()

        h = model.transformer.ln_f(hidden['h'].float())
        logits = model.lm_head(h).squeeze()
        probs = torch.softmax(logits.float(), dim=-1)
        signals['attn_entropy'] = -(probs * torch.log(probs + 1e-10)).sum().item()

    return signals


def compute_auc(known_vals, unknown_vals):
    """Compute AUC for separating known (low) vs unknown (high)."""
    all_vals = list(known_vals) + list(unknown_vals)
    if not all_vals or max(all_vals) == min(all_vals):
        return 0.5
    thresholds = np.linspace(min(all_vals) - 1, max(all_vals) + 1, 50)
    fpr_list, tpr_list = [], []
    for thr in thresholds:
        tp = sum(1 for v in unknown_vals if v > thr)
        fp = sum(1 for v in known_vals if v > thr)
        fn = sum(1 for v in unknown_vals if v <= thr)
        tn = sum(1 for v in known_vals if v <= thr)
        tpr_list.append(tp / max(1, tp + fn))
        fpr_list.append(fp / max(1, fp + tn))
    # Sort by fpr
    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_s = [p[0] for p in pairs]
    tpr_s = [p[1] for p in pairs]
    return float(np.trapz(tpr_s, fpr_s)) if len(fpr_s) > 1 else 0.5


def main():
    print("[P118] The SwiGLU / Residual Oracle")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    MODEL_CONFIGS = [
        ('Qwen/Qwen2.5-0.5B', 'Qwen2.5-0.5B', 24, 'qwen'),
        ('Qwen/Qwen2.5-1.5B', 'Qwen2.5-1.5B', 28, 'qwen'),
        ('gpt2-xl',            'GPT2-XL',       48, 'gpt2'),
    ]

    all_results = {}

    for model_id, label, n_layers, model_type in MODEL_CONFIGS:
        print(f"\n  === {label} ===")
        try:
            if model_type == 'gpt2':
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                model = GPT2LMHeadModel.from_pretrained(
                    model_id, local_files_only=True).eval().to(DEVICE)
                tok = GPT2Tokenizer.from_pretrained(model_id, local_files_only=True)
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, local_files_only=True,
                    torch_dtype=torch.float16
                ).eval().to(DEVICE)
                tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        best_layer = int(n_layers * 0.94)

        # === Part 1: Signals at L_0.94 ===
        print(f"    L_best (0.94): L{best_layer}")

        known_signals = []
        for prompt, answer in KNOWN_FACTS:
            sig = measure_layer_signals(model, tok, prompt, best_layer, model_type)
            known_signals.append(sig)

        unknown_signals = []
        for prompt, answer in UNKNOWN_FACTS:
            sig = measure_layer_signals(model, tok, prompt, best_layer, model_type)
            unknown_signals.append(sig)

        # Compute AUC for each metric
        metrics = ['gate_l1', 'gate_sparsity', 'residual_l2', 'residual_mean', 'attn_entropy']
        auc_results = {}
        for metric in metrics:
            k_vals = [s.get(metric, 0) for s in known_signals]
            u_vals = [s.get(metric, 0) for s in unknown_signals]
            # Try both directions (known > unknown or known < unknown)
            auc_high = compute_auc(k_vals, u_vals)  # unknown = higher
            auc_low = compute_auc(u_vals, k_vals)    # unknown = lower (inverted)
            best_auc = max(auc_high, auc_low)
            direction = 'unknown_higher' if auc_high >= auc_low else 'unknown_lower'
            auc_results[metric] = {
                'auc': best_auc,
                'direction': direction,
                'known_mean': float(np.mean(k_vals)),
                'unknown_mean': float(np.mean(u_vals)),
                'known_std': float(np.std(k_vals)),
                'unknown_std': float(np.std(u_vals)),
            }
            print(f"    {metric}: AUC={best_auc:.3f} ({direction})")
            print(f"      Known: {np.mean(k_vals):.3f} +/- {np.std(k_vals):.3f}")
            print(f"      Unknown: {np.mean(u_vals):.3f} +/- {np.std(u_vals):.3f}")

        # === Part 2: Full-depth scan (where does the signal emerge?) ===
        print(f"    Scanning all {n_layers} layers...")
        scan_layers = list(range(0, n_layers, max(1, n_layers // 12)))  # ~12 samples
        if best_layer not in scan_layers:
            scan_layers.append(best_layer)
        scan_layers.sort()

        depth_scan = {}
        for li in scan_layers:
            k_sigs = [measure_layer_signals(model, tok, p, li, model_type)
                      for p, _ in KNOWN_FACTS[:5]]  # Subset for speed
            u_sigs = [measure_layer_signals(model, tok, p, li, model_type)
                      for p, _ in UNKNOWN_FACTS[:5]]
            layer_aucs = {}
            for metric in metrics:
                kv = [s.get(metric, 0) for s in k_sigs]
                uv = [s.get(metric, 0) for s in u_sigs]
                layer_aucs[metric] = max(compute_auc(kv, uv), compute_auc(uv, kv))
            depth_scan[li] = layer_aucs

        all_results[label] = {
            'model': model_id,
            'n_layers': n_layers,
            'best_layer': best_layer,
            'auc_at_best': auc_results,
            'depth_scan': depth_scan,
            'best_metric': max(auc_results.items(), key=lambda x: x[1]['auc'])[0],
            'best_auc': max(auc_results.items(), key=lambda x: x[1]['auc'])[1]['auc'],
        }

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    out = {'phase': '118', 'name': 'SwiGLU / Residual Oracle', 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase118_swiglu_oracle.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # === Plot ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {'gate_l1': '#e74c3c', 'gate_sparsity': '#e67e22',
              'residual_l2': '#3498db', 'residual_mean': '#2ecc71',
              'attn_entropy': '#9b59b6'}

    for idx, (label, data) in enumerate(all_results.items()):
        if idx >= 3:
            break
        ax = axes[idx]
        aucs = data['auc_at_best']
        metric_names = list(aucs.keys())
        auc_vals = [aucs[m]['auc'] for m in metric_names]
        bars = ax.bar(range(len(metric_names)), auc_vals,
                     color=[colors.get(m, '#95a5a6') for m in metric_names])
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels([m.replace('_', '\n') for m in metric_names],
                          fontsize=8, rotation=0)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{label}\nBest: {data["best_metric"]} (AUC={data["best_auc"]:.2f})',
                    fontweight='bold')
        ax.set_ylabel('AUC (Known vs Unknown)')
        # Add value labels
        for bar, val in zip(bars, auc_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('Phase 118: SwiGLU/Residual Oracle - Which Signal Detects Uncertainty?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase118_swiglu_oracle.png'), dpi=150)
    plt.close()

    # Depth scan plot
    n_models = len(all_results)
    if n_models > 0:
        fig, axes = plt.subplots(1, min(3, n_models), figsize=(6*min(3, n_models), 5))
        if n_models == 1:
            axes = [axes]
        for idx, (label, data) in enumerate(all_results.items()):
            if idx >= 3:
                break
            ax = axes[idx]
            scan = data['depth_scan']
            layers = sorted(scan.keys())
            for metric in ['gate_l1', 'residual_l2', 'attn_entropy']:
                vals = [scan[l].get(metric, 0.5) for l in layers]
                ax.plot(layers, vals, 'o-', label=metric, color=colors.get(metric, 'gray'))
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=data['best_layer'], color='red', linestyle=':', alpha=0.5,
                      label=f'L_0.94={data["best_layer"]}')
            ax.set_xlabel('Layer')
            ax.set_ylabel('AUC')
            ax.set_title(f'{label}: Depth Scan')
            ax.legend(fontsize=7)
            ax.set_ylim(0, 1)
        fig.suptitle('Phase 118: Where Does the Uncertainty Signal Emerge?',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'phase118_depth_scan.png'), dpi=150)
        plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 118] Complete.")


if __name__ == '__main__':
    main()
