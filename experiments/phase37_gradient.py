# -*- coding: utf-8 -*-
"""
Phase 37: Latent Entropy Gradient Descent
Instead of rejecting output tokens, optimize the HIDDEN STATE itself
to minimize attention entropy. LLM "System 2" thinking.
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
    print("[P37] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def entropy_gradient_descent(model, tok, prompt, n_steps=10, lr=0.5):
    """Optimize hidden state at last layer to minimize attention entropy."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    # Get baseline hidden state from final layer
    hidden_states = {}
    def capture_hook(module, args, output):
        hidden_states['last'] = output[0].detach().clone()
    handle = model.transformer.h[-1].register_forward_hook(capture_hook)
    with torch.no_grad():
        out = model(**inp, output_attentions=True, return_dict=True)
    handle.remove()

    baseline_logits = out.logits[:, -1, :].squeeze(0).detach()
    baseline_top = torch.argmax(baseline_logits).item()

    # Compute baseline entropy
    base_ents = []
    for attn in out.attentions:
        for h in range(attn.shape[1]):
            a = attn[0, h, -1, :].cpu().numpy()
            base_ents.append(float(-np.sum(a * np.log(a + 1e-12))))
    base_mean_ent = float(np.mean(base_ents))

    # Create optimizable perturbation on the hidden state
    h_orig = hidden_states['last'][:, -1, :].clone()  # (1, d_model)
    delta = torch.zeros_like(h_orig, requires_grad=True)

    entropy_trace = [base_mean_ent]
    rank_trace = []

    for step in range(n_steps):
        h_mod = h_orig + delta

        # Project through LM head to get logits
        logits = model.lm_head(model.transformer.ln_f(h_mod)).squeeze(0)

        # Compute softmax entropy of output as proxy for attention entropy
        probs = F.softmax(logits, dim=-1)
        output_entropy = -torch.sum(probs * torch.log(probs + 1e-12))

        # Also add L2 regularization to keep perturbation small
        loss = output_entropy + 0.1 * torch.sum(delta ** 2)

        loss.backward()
        with torch.no_grad():
            delta -= lr * delta.grad
            delta.grad.zero_()

        entropy_trace.append(float(output_entropy.item()))

    # Final prediction
    with torch.no_grad():
        final_logits = model.lm_head(model.transformer.ln_f(h_orig + delta)).squeeze(0)
    final_top = torch.argmax(final_logits).item()

    return {
        'baseline_token': tok.decode([baseline_top]).encode('ascii','replace').decode().strip(),
        'baseline_id': baseline_top,
        'optimized_token': tok.decode([final_top]).encode('ascii','replace').decode().strip(),
        'optimized_id': final_top,
        'baseline_entropy': base_mean_ent,
        'final_entropy': entropy_trace[-1],
        'entropy_reduction': base_mean_ent - entropy_trace[-1],
        'entropy_trace': entropy_trace,
        'token_changed': baseline_top != final_top,
    }

def main():
    print("=" * 70)
    print("  Phase 37: Latent Entropy Gradient Descent")
    print("  Optimize hidden state to minimize entropy (System 2)")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("DNA stands for", [390], "de"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The speed of light is approximately", [22626], "299"),
    ]

    # === P37a: Gradient descent optimization ===
    print("\n[P37a] Entropy gradient descent...")
    results_list = []

    for prompt, fact_ids, expected in tests:
        for n_steps in [5, 10, 20, 50]:
            res = entropy_gradient_descent(model, tok, prompt, n_steps=n_steps, lr=0.3)
            res['prompt'] = prompt[:35]
            res['expected'] = expected
            res['n_steps'] = n_steps
            res['baseline_correct'] = res['baseline_id'] in fact_ids
            res['optimized_correct'] = res['optimized_id'] in fact_ids

            if n_steps == 20:  # Main result
                b = 'OK' if res['baseline_correct'] else 'FAIL'
                o = 'OK' if res['optimized_correct'] else 'FAIL'
                print(f"  {expected:>8s}: [{b}]{res['baseline_token']:>10s} -> "
                      f"[{o}]{res['optimized_token']:>10s} "
                      f"H: {res['baseline_entropy']:.3f}->{res['final_entropy']:.3f} "
                      f"({res['entropy_reduction']:+.3f})")
            results_list.append(res)

    # Aggregate by n_steps
    step_analysis = {}
    for n in [5, 10, 20, 50]:
        subset = [r for r in results_list if r['n_steps'] == n]
        changed = sum(1 for r in subset if r['token_changed'])
        improved = sum(1 for r in subset if r['optimized_correct'] and not r['baseline_correct'])
        degraded = sum(1 for r in subset if r['baseline_correct'] and not r['optimized_correct'])
        step_analysis[n] = {
            'changed': changed, 'improved': improved, 'degraded': degraded,
            'mean_ent_reduction': float(np.mean([r['entropy_reduction'] for r in subset])),
        }
        print(f"\n  n_steps={n:>3d}: changed={changed}/{len(tests)}, "
              f"improved={improved}, degraded={degraded}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Entropy traces (n_steps=20)
    for r in [r for r in results_list if r['n_steps'] == 20]:
        color = 'green' if r['optimized_correct'] else 'red'
        axes[0].plot(r['entropy_trace'], alpha=0.6, color=color, label=r['expected'])
    axes[0].set_xlabel('Optimization Step')
    axes[0].set_ylabel('Output Entropy')
    axes[0].set_title('Entropy During Gradient Descent')
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Token changes by n_steps
    ns = sorted(step_analysis.keys())
    changed_vals = [step_analysis[n]['changed'] for n in ns]
    improved_vals = [step_analysis[n]['improved'] for n in ns]
    axes[1].bar([str(n) for n in ns], changed_vals, color='blue', alpha=0.5, label='Changed')
    axes[1].bar([str(n) for n in ns], improved_vals, color='green', alpha=0.7, label='Improved')
    axes[1].set_xlabel('Optimization Steps')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Token Changes vs Steps')
    axes[1].legend()

    # Plot 3: Entropy reduction
    main_res = [r for r in results_list if r['n_steps'] == 20]
    labels = [r['expected'] for r in main_res]
    reductions = [r['entropy_reduction'] for r in main_res]
    colors = ['green' if r > 0 else 'red' for r in reductions]
    axes[2].bar(labels, reductions, color=colors, alpha=0.7)
    axes[2].set_ylabel('Entropy Reduction')
    axes[2].set_title('Per-Prompt Entropy Change')
    axes[2].tick_params(axis='x', rotation=45, labelsize=7)
    axes[2].axhline(y=0, color='black', linewidth=0.5)

    plt.suptitle('Phase 37: Latent Entropy Gradient Descent', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase37_gradient.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 37, 'name': 'Latent Entropy Gradient Descent',
        'step_analysis': step_analysis,
        'per_case': [r for r in results_list if r['n_steps'] == 20],
    }
    with open(os.path.join(RESULTS_DIR, 'phase37_gradient.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: str(x) if isinstance(x, np.floating) else x)

    print("\n" + "=" * 70)
    print("  PHASE 37 RESULTS")
    print("=" * 70)
    for n in ns:
        sa = step_analysis[n]
        print(f"  steps={n}: changed={sa['changed']}, improved={sa['improved']}, "
              f"H_reduction={sa['mean_ent_reduction']:+.3f}")
    print("=" * 70)
    phase_complete(37)

if __name__ == '__main__':
    main()
