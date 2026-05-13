# -*- coding: utf-8 -*-
"""
Phase 88: Cosmological GSF Scaling
Test whether GSF laws hold in GPT-2 Medium (345M, 24 layers).
Map the suppression landscape in a larger model.
"""
import torch, json, os, sys, numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

FACTS = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("The chemical symbol for gold is", " Au"),
    ("The speed of light is approximately", " 299"),
    ("Albert Einstein was born in", " Ul"),
    ("The tallest mountain in the world is", " Mount"),
    ("The currency of the United Kingdom is the", " pound"),
    ("The author of Romeo and Juliet is", " William"),
    ("The first president of the United States was", " George"),
    ("The chemical formula for water is", " H"),
    ("The boiling point of water is", " 100"),
    ("The atomic number of carbon is", " 6"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("The speed of sound is approximately", " 343"),
    ("Photosynthesis converts sunlight into", " chemical"),
]

def logit_lens_layer(model, tok, prompt, layer_idx, n_layers):
    """Extract logits from a specific layer using Logit Lens."""
    inp = tok(prompt, return_tensors='pt')
    hidden = {}
    def hook_fn(module, input, output):
        hidden['h'] = output[0][:, -1, :].detach()
    handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inp)
    handle.remove()
    # Apply final LayerNorm + LM head
    h = model.transformer.ln_f(hidden['h'])
    logits = model.lm_head(h).squeeze()
    return logits

def main():
    print("[P88] Cosmological GSF Scaling - GPT-2 Medium")

    # Try to load GPT-2 Medium
    try:
        model_med = GPT2LMHeadModel.from_pretrained('gpt2-medium', local_files_only=True).eval()
        tok_med = GPT2Tokenizer.from_pretrained('gpt2-medium', local_files_only=True)
        has_medium = True
        n_layers_med = 24
        n_heads_med = 16
        print("  GPT-2 Medium loaded successfully (345M, 24 layers, 16 heads)")
    except Exception as e:
        print(f"  GPT-2 Medium not available locally: {e}")
        print("  Falling back to GPT-2 Small (124M) with layer-ablation scaling")
        has_medium = False

    # Load GPT-2 Small for comparison
    model_sm = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
    tok_sm = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    n_layers_sm = 12

    results = {'small': {}, 'medium': {}}

    # ===== GPT-2 Small: per-layer Logit Lens =====
    print("\n  === GPT-2 Small (124M, 12 layers) ===")
    sm_layer_ranks = {l: [] for l in range(n_layers_sm)}
    for prompt, answer in FACTS:
        fact_id = tok_sm.encode(answer)[0]
        for layer in range(n_layers_sm):
            logits = logit_lens_layer(model_sm, tok_sm, prompt, layer, n_layers_sm)
            rank = (logits.argsort(descending=True) == fact_id).nonzero().item() + 1
            sm_layer_ranks[layer].append(rank)

    sm_summary = {}
    for l in range(n_layers_sm):
        median = float(np.median(sm_layer_ranks[l]))
        acc = sum(1 for r in sm_layer_ranks[l] if r == 1) / len(FACTS)
        sm_summary[f'L{l}'] = {'median_rank': median, 'accuracy': acc}
        if l >= 8:
            print(f"    L{l:2d}: median_rank={median:.0f}, acc={acc:.0%}")

    best_layer_sm = min(sm_summary, key=lambda k: sm_summary[k]['median_rank'])
    results['small'] = {
        'layers': n_layers_sm,
        'best_layer': best_layer_sm,
        'best_accuracy': sm_summary[best_layer_sm]['accuracy'],
        'layer_summary': sm_summary,
    }
    print(f"    Best layer: {best_layer_sm} (acc={sm_summary[best_layer_sm]['accuracy']:.0%})")

    # ===== GPT-2 Medium: per-layer Logit Lens =====
    if has_medium:
        print(f"\n  === GPT-2 Medium (345M, {n_layers_med} layers) ===")
        med_layer_ranks = {l: [] for l in range(n_layers_med)}
        for prompt, answer in FACTS:
            fact_id = tok_med.encode(answer)[0]
            for layer in range(n_layers_med):
                logits = logit_lens_layer(model_med, tok_med, prompt, layer, n_layers_med)
                rank = (logits.argsort(descending=True) == fact_id).nonzero().item() + 1
                med_layer_ranks[layer].append(rank)

        med_summary = {}
        for l in range(n_layers_med):
            median = float(np.median(med_layer_ranks[l]))
            acc = sum(1 for r in med_layer_ranks[l] if r == 1) / len(FACTS)
            med_summary[f'L{l}'] = {'median_rank': median, 'accuracy': acc}
            if l >= 18:
                print(f"    L{l:2d}: median_rank={median:.0f}, acc={acc:.0%}")

        best_layer_med = min(med_summary, key=lambda k: med_summary[k]['median_rank'])
        results['medium'] = {
            'layers': n_layers_med,
            'best_layer': best_layer_med,
            'best_accuracy': med_summary[best_layer_med]['accuracy'],
            'layer_summary': med_summary,
        }
        print(f"    Best layer: {best_layer_med} (acc={med_summary[best_layer_med]['accuracy']:.0%})")

        # Test Code Mode Switch on Medium
        print("\n  === Code Mode Switch on GPT-2 Medium ===")
        code_results = {}
        for tmpl_name, tmpl in [('natural', '{p}'), ('comment', '# {p}')]:
            correct = 0
            for prompt, answer in FACTS:
                p = tmpl.format(p=prompt)
                inp = tok_med(p, return_tensors='pt')
                fact_id = tok_med.encode(answer)[0]
                with torch.no_grad():
                    logits = model_med(**inp).logits[0, -1, :]
                rank = (logits.argsort(descending=True) == fact_id).nonzero().item() + 1
                if rank == 1:
                    correct += 1
            acc = correct / len(FACTS)
            code_results[tmpl_name] = acc
            print(f"    {tmpl_name}: acc={acc:.0%}")
        results['medium']['code_mode_switch'] = code_results

        del model_med
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out = {'phase': 88, 'name': 'Cosmological GSF Scaling', 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase88_cosmological_scaling.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Small model trajectory
    sm_layers = list(range(n_layers_sm))
    sm_medians = [float(np.median(sm_layer_ranks[l])) for l in sm_layers]
    sm_accs = [sum(1 for r in sm_layer_ranks[l] if r == 1) / len(FACTS) for l in sm_layers]

    axes[0].plot(sm_layers, sm_medians, 'o-', color='#3498db', label='GPT-2 Small (124M)', linewidth=2)
    if has_medium:
        # Normalize x-axis to [0,1] for comparison
        med_layers = list(range(n_layers_med))
        med_medians = [float(np.median(med_layer_ranks[l])) for l in med_layers]
        # Plot on same axis with scaled x
        med_x_scaled = [l * (n_layers_sm - 1) / (n_layers_med - 1) for l in med_layers]
        axes[0].plot(med_x_scaled, med_medians, 's-', color='#e74c3c', label='GPT-2 Medium (345M)', linewidth=2)
    axes[0].set_xlabel('Layer (normalized)')
    axes[0].set_ylabel('Median Fact Rank')
    axes[0].set_title('GSF Trajectory: Small vs Medium')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sm_layers, sm_accs, 'o-', color='#3498db', label='GPT-2 Small', linewidth=2)
    if has_medium:
        med_accs = [sum(1 for r in med_layer_ranks[l] if r == 1) / len(FACTS) for l in med_layers]
        axes[1].plot(med_x_scaled, med_accs, 's-', color='#e74c3c', label='GPT-2 Medium', linewidth=2)
    axes[1].set_xlabel('Layer (normalized)')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Per-Layer Accuracy: Small vs Medium')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Phase 88: Cosmological GSF Scaling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase88_cosmological_scaling.png'), dpi=150)
    plt.close()
    print("[Phase 88] Complete.")

if __name__ == '__main__':
    main()
