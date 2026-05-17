# -*- coding: utf-8 -*-
"""
Phase 176: The Code Mode Spectrum (Opus Addition)
P170 discovered: "# " prefix protects arithmetic under Surgery.
But WHY "# " specifically? Is it the comment character, or any code context?

Test multiple prefixes:
  "# "     - Python comment (P170 winner)
  ">>> "   - Python REPL prompt
  "def f:" - Python function context
  "// "    - C/Java comment
  "$ "     - Shell prompt
  "Q: "    - Question prefix
  ""       - No prefix (control)

If ONLY "#" works, it's specific to comment processing.
If all code prefixes work, it's a general "code mode" activation.

Model: Qwen2.5-0.5B (GPU)
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

ARITH_TEST = [
    ("1 + 1 =", " 2"), ("3 + 4 =", " 7"), ("5 + 5 =", " 10"),
    ("2 + 7 =", " 9"), ("8 + 1 =", " 9"), ("6 + 3 =", " 9"),
    ("4 + 4 =", " 8"), ("9 + 0 =", " 9"), ("7 + 2 =", " 9"),
    ("2 + 6 =", " 8"),
]

FACT_TEST = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("Water freezes at", " 0"),
    ("A year has", " 365"),
    ("The number of continents is", " 7"),
]

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"]

PREFIXES = [
    ("none", ""),
    ("hash", "# "),
    ("repl", ">>> "),
    ("def", "def f(): return "),
    ("slash", "// "),
    ("shell", "$ echo "),
    ("question", "Q: "),
    ("answer", "A: "),
    ("print", "print("),
]


def apply_surgery(model, tok, strength=2.0):
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)


def evaluate(model, tok, test_data, prefix="", fga_gain=5):
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - max(1, n_layers // 4)
    correct = 0
    for prompt, expected in test_data:
        text = f"{prefix}{prompt}"
        exp_id = tok.encode(expected)[-1]
        # FGA hook
        unembed = model.lm_head.weight.data[exp_id].float()
        direction = unembed / (unembed.norm() + 1e-8)
        hook_handle = None
        if fga_gain > 0:
            def make_hook(d, g):
                def fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].float()
                        if h.dim() == 3: h[:, -1, :] += g * d.to(h.device)
                        return (h.to(output[0].dtype),) + output[1:]
                    h = output.float()
                    if h.dim() == 3: h[:, -1, :] += g * d.to(h.device)
                    return h.to(output.dtype)
                return fn
            hook_handle = model.model.layers[fga_layer].register_forward_hook(
                make_hook(direction, fga_gain))
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if hook_handle: hook_handle.remove()
        pred_id = logits.argmax().item()
        if pred_id == exp_id or tok.decode([pred_id]).strip() == expected.strip():
            correct += 1
    return correct / len(test_data)


def main():
    print("[P176] The Code Mode Spectrum")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    results = {}
    print("\n  === Surgery + FGA + Various Prefixes ===")
    for name, prefix in PREFIXES:
        arith = evaluate(model, tok, ARITH_TEST, prefix=prefix, fga_gain=5)
        fact = evaluate(model, tok, FACT_TEST, prefix=prefix, fga_gain=5)
        print(f"    {name:12s} ('{prefix[:8]:8s}'): arith={arith:.0%} fact={fact:.0%}")
        results[name] = {'prefix': prefix, 'arith': arith, 'fact': fact}

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase176_spectrum.json'), 'w') as f:
        json.dump({'phase': '176', 'name': 'Code Mode Spectrum',
                   'results': results}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    names_list = [n for n, _ in PREFIXES]
    arith_vals = [results[n]['arith'] for n in names_list]
    fact_vals = [results[n]['fact'] for n in names_list]
    x = np.arange(len(names_list))
    w = 0.35
    ax.bar(x-w/2, arith_vals, w, label='Arithmetic', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, fact_vals, w, label='Factual', color='#e74c3c', alpha=0.8)
    for i in range(len(names_list)):
        ax.text(x[i]-w/2, arith_vals[i]+0.02, f'{arith_vals[i]:.0%}',
                ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i]+w/2, fact_vals[i]+0.02, f'{fact_vals[i]:.0%}',
                ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    labels = [f'{n}\n"{p[:6]}"' for n, p in PREFIXES]
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11); ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase 176: The Code Mode Spectrum\n'
                 'Which prefixes protect arithmetic under Surgery+FGA?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase176_spectrum.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    code_like = [n for n in names_list if results[n]['arith'] > 0.3]
    print(f"  -> Prefixes that protect arithmetic (>30%): {code_like}")
    if len(code_like) == 1 and 'hash' in code_like:
        print("  -> '#' IS SPECIAL! Only comment character activates protection.")
    elif len(code_like) > 2:
        print("  -> GENERAL CODE CONTEXT activates protection!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 176] Complete.")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
