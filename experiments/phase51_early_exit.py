# -*- coding: utf-8 -*-
"""
Phase 51: Entropy-Triggered Early Exit
Normal: decode from L12 (full model). When entropy spikes: skip L11-L12,
decode from L10 directly. Preserves grammar when confident, extracts
raw facts when uncertain.
"""
import os, json, sys
import numpy as np
import torch
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
    print("[P51] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def early_exit_generate(model, tok, prompt, exit_layer=10,
                        entropy_threshold=1.0, max_tokens=15):
    """Generate tokens. Use L12 normally, switch to early exit when H > threshold."""
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    gen = ids.clone()
    tokens = []
    actions = []

    for step in range(max_tokens):
        # Collect hidden state at exit_layer and attentions
        hidden_at_exit = {}
        def hook(module, args, output):
            hidden_at_exit['h'] = output[0][0, -1, :].detach()
        handle = model.transformer.h[exit_layer].register_forward_hook(hook)

        with torch.no_grad():
            out = model(gen, output_attentions=True, return_dict=True)
        handle.remove()

        # Compute attention entropy
        ents = []
        for attn in out.attentions:
            for h in range(attn.shape[1]):
                a = attn[0, h, -1, :].cpu().numpy()
                ents.append(float(-np.sum(a * np.log(a + 1e-12))))
        mean_ent = float(np.mean(ents))

        if mean_ent > entropy_threshold:
            # HIGH ENTROPY -> Early exit from L10
            normed = model.transformer.ln_f(hidden_at_exit['h'].unsqueeze(0))
            logits = model.lm_head(normed).squeeze(0)
            actions.append(f'EXIT_L{exit_layer}(H={mean_ent:.2f})')
        else:
            # LOW ENTROPY -> Use full model
            logits = out.logits[:, -1, :].squeeze(0)
            actions.append(f'FULL(H={mean_ent:.2f})')

        next_tok = torch.argmax(logits).item()
        tokens.append(tok.decode([next_tok]).encode('ascii','replace').decode())
        if next_tok == tok.eos_token_id:
            break
        gen = torch.cat([gen, torch.tensor([[next_tok]], device=DEVICE)], dim=1)

    return ''.join(tokens), tokens, actions

def main():
    print("=" * 70)
    print("  Phase 51: Entropy-Triggered Early Exit")
    print("  Full model when confident, L10 when uncertain")
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
        ("The Earth orbits the", [4252], "Sun"),
        ("The boiling point of water is", [1802], "100"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("Oxygen has the atomic number", [807], "8"),
    ]

    # === P51a: Single-token early exit (first token only) ===
    print("\n[P51a] Single-token early exit analysis...")
    single_results = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

        # Full model
        hidden_at_10 = {}
        def hook(module, args, output):
            hidden_at_10['h'] = output[0][0, -1, :].detach()
        handle = model.transformer.h[10].register_forward_hook(hook)
        with torch.no_grad():
            out = model(inp, output_attentions=True, return_dict=True)
        handle.remove()

        full_logits = out.logits[:, -1, :].squeeze(0)
        normed = model.transformer.ln_f(hidden_at_10['h'].unsqueeze(0))
        l10_logits = model.lm_head(normed).squeeze(0)

        ents = []
        for attn in out.attentions:
            for h in range(attn.shape[1]):
                a = attn[0, h, -1, :].cpu().numpy()
                ents.append(float(-np.sum(a * np.log(a + 1e-12))))
        mean_ent = float(np.mean(ents))

        full_tok = torch.argmax(full_logits).item()
        l10_tok = torch.argmax(l10_logits).item()

        full_correct = full_tok in fact_ids
        l10_correct = l10_tok in fact_ids

        # Dynamic decision
        if mean_ent > 1.0:
            selected = l10_tok
            source = f'L10(H={mean_ent:.2f})'
        else:
            selected = full_tok
            source = f'FULL(H={mean_ent:.2f})'
        dyn_correct = selected in fact_ids

        single_results.append({
            'expected': expected, 'entropy': round(mean_ent, 3),
            'full_correct': full_correct, 'l10_correct': l10_correct,
            'dynamic_correct': dyn_correct, 'source': source,
        })

        fc = 'OK' if full_correct else 'FAIL'
        lc = 'OK' if l10_correct else 'FAIL'
        dc = 'OK' if dyn_correct else 'FAIL'
        print(f"  {expected:>12s}: H={mean_ent:.3f} full=[{fc}] L10=[{lc}] dyn=[{dc}] {source}")

    full_acc = sum(1 for r in single_results if r['full_correct']) / len(tests)
    l10_acc = sum(1 for r in single_results if r['l10_correct']) / len(tests)
    dyn_acc = sum(1 for r in single_results if r['dynamic_correct']) / len(tests)

    # === P51b: Multi-token generation with early exit ===
    print("\n[P51b] Multi-token early exit generation...")
    gen_results = []
    for prompt, fact_ids, expected in tests:
        # Baseline
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        with torch.no_grad():
            base_out = model.generate(inp, max_new_tokens=10, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        base_text = tok.decode(base_out[0][inp.shape[1]:]).encode('ascii','replace').decode()

        # Early exit
        ee_text, ee_tokens, ee_actions = early_exit_generate(
            model, tok, prompt, exit_layer=10, entropy_threshold=1.0, max_tokens=10)

        n_exits = sum(1 for a in ee_actions if 'EXIT' in a)
        gen_results.append({
            'expected': expected, 'base': base_text[:40],
            'early_exit': ee_text[:40], 'n_exits': n_exits,
            'actions': ee_actions[:5],
        })
        print(f"  {expected:>12s}: exits={n_exits}")
        print(f"    Base: {base_text[:35]}")
        print(f"    EE:   {ee_text[:35]}")

    # === P51c: Threshold sweep ===
    print("\n[P51c] Threshold sweep for early exit...")
    threshold_results = {}
    for thresh in [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]:
        correct = 0
        for prompt, fact_ids, expected in tests:
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
            hidden_10 = {}
            def hook(m, a, o):
                hidden_10['h'] = o[0][0, -1, :].detach()
            handle = model.transformer.h[10].register_forward_hook(hook)
            with torch.no_grad():
                out = model(inp, output_attentions=True, return_dict=True)
            handle.remove()

            ents = []
            for attn in out.attentions:
                for h in range(attn.shape[1]):
                    a = attn[0, h, -1, :].cpu().numpy()
                    ents.append(float(-np.sum(a * np.log(a + 1e-12))))
            ent = float(np.mean(ents))

            if ent > thresh:
                normed = model.transformer.ln_f(hidden_10['h'].unsqueeze(0))
                logits = model.lm_head(normed).squeeze(0)
            else:
                logits = out.logits[:, -1, :].squeeze(0)

            if torch.argmax(logits).item() in fact_ids:
                correct += 1

        threshold_results[thresh] = correct / len(tests)
        print(f"  thresh={thresh:.1f}: {correct}/{len(tests)} = {correct/len(tests):.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = ['Full (L12)', 'Always L10', 'Dynamic']
    method_accs = [full_acc*100, l10_acc*100, dyn_acc*100]
    axes[0].bar(methods, method_accs, color=['red', 'green', 'blue'], alpha=0.7)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Single-Token Accuracy')

    thresholds = sorted(threshold_results.keys())
    t_accs = [threshold_results[t]*100 for t in thresholds]
    axes[1].plot(thresholds, t_accs, 'g.-', linewidth=2, markersize=10)
    axes[1].set_xlabel('Entropy Threshold')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Early Exit Threshold Sweep')
    axes[1].grid(True, alpha=0.3)

    labels = [r['expected'][:6] for r in single_results]
    ents = [r['entropy'] for r in single_results]
    colors = ['green' if r['dynamic_correct'] else 'red' for r in single_results]
    axes[2].bar(labels, ents, color=colors, alpha=0.7)
    axes[2].axhline(y=1.0, color='black', linestyle='--', label='threshold')
    axes[2].set_ylabel('Attention Entropy')
    axes[2].set_title('Oracle + Dynamic Exit')
    axes[2].tick_params(axis='x', rotation=45, labelsize=7)

    plt.suptitle('Phase 51: Entropy-Triggered Early Exit', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase51_early_exit.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 51, 'name': 'Entropy-Triggered Early Exit',
        'full_accuracy': full_acc, 'l10_accuracy': l10_acc,
        'dynamic_accuracy': dyn_acc,
        'threshold_sweep': {str(k): v for k, v in threshold_results.items()},
        'single_results': single_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase51_early_exit.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 51 RESULTS")
    print("=" * 70)
    print(f"  Full (L12):  {full_acc:.0%}")
    print(f"  Always L10:  {l10_acc:.0%}")
    print(f"  Dynamic:     {dyn_acc:.0%}")
    best_t = max(threshold_results, key=threshold_results.get)
    print(f"  Best thresh: {best_t} -> {threshold_results[best_t]:.0%}")
    print("=" * 70)
    phase_complete(51)

if __name__ == '__main__':
    main()
