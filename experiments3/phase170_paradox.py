# -*- coding: utf-8 -*-
"""
Phase 170: The Arithmetic Paradox Resolution
P166 showed Surgery+FGA PRESERVES 75% arithmetic accuracy.
P147 showed Surgery DESTROYS arithmetic.

WHY? The key difference: P166 used Code Mode prefix ("# ").
Hypothesis: Code Mode protects arithmetic from Surgery damage.

Systematic test:
  A) Base (no surgery, no prefix)
  B) Surgery only (no prefix, no FGA)
  C) Surgery + Code Mode (no FGA)
  D) Surgery + FGA (no Code Mode)
  E) Surgery + Code Mode + FGA (full Shield&Sword)

If C >> B, then Code Mode is the protector.
If D >> B, then FGA is the protector.
If E >> C and E >> D, they're synergistic.

Model: Qwen2.5-0.5B (GPU)
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

# Arithmetic test (single-digit addition, clear answers)
ARITH_TEST = [
    ("1 + 1 =", " 2"),
    ("3 + 4 =", " 7"),
    ("5 + 5 =", " 10"),
    ("2 + 7 =", " 9"),
    ("8 + 1 =", " 9"),
    ("6 + 3 =", " 9"),
    ("4 + 4 =", " 8"),
    ("9 + 0 =", " 9"),
    ("7 + 2 =", " 9"),
    ("2 + 6 =", " 8"),
    ("3 + 3 =", " 6"),
    ("5 + 2 =", " 7"),
]

# Factual test (for comparison)
FACT_TEST = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("Water freezes at", " 0"),
    ("A year has", " 365"),
    ("The number of continents is", " 7"),
]

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"]


def apply_surgery(model, tok, strength=2.0):
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)


class FGAHook:
    def __init__(self, direction, gain):
        self.gain = gain
        self.direction = direction
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        h = output.float()
        if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
        elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
        return h.to(output.dtype)

    def register(self, layer):
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def evaluate(model, tok, test_data, code_mode=False, fga_gain=0):
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - max(1, n_layers // 3)
    correct = 0
    for prompt, expected in test_data:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[-1]
        hook = None
        if fga_gain > 0:
            unembed = model.lm_head.weight.data[exp_id].float()
            direction = unembed / (unembed.norm() + 1e-8)
            hook = FGAHook(direction, fga_gain)
            hook.register(model.model.layers[fga_layer])
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if hook: hook.remove()

        # For arithmetic, also accept if model predicts space then digit
        pred_id = logits.argmax().item()
        if pred_id == exp_id:
            correct += 1
        elif tok.decode([pred_id]).strip() == expected.strip():
            correct += 1
    return correct / len(test_data)


def main():
    print("[P170] The Arithmetic Paradox Resolution")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    configs = {}
    conditions = [
        ('A_base',          False, False, 0),   # No surgery, no prefix, no FGA
        ('B_surgery_only',  True,  False, 0),   # Surgery, no prefix, no FGA
        ('C_surgery_code',  True,  True,  0),   # Surgery + Code Mode, no FGA
        ('D_surgery_fga',   True,  False, 5),   # Surgery + FGA, no Code Mode
        ('E_full_sas',      True,  True,  5),   # Surgery + Code Mode + FGA
        ('F_code_only',     False, True,  0),   # Code Mode only (no surgery)
        ('G_fga_only',      False, False, 5),   # FGA only (no surgery)
        ('H_code_fga',      False, True,  5),   # Code Mode + FGA (no surgery)
    ]

    for name, do_surgery, code_mode, fga_gain in conditions:
        print(f"\n  === {name} (surgery={do_surgery}, code={code_mode}, fga={fga_gain}) ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
        if do_surgery:
            apply_surgery(model, tok, strength=2.0)

        arith_acc = evaluate(model, tok, ARITH_TEST, code_mode=code_mode, fga_gain=fga_gain)
        fact_acc = evaluate(model, tok, FACT_TEST, code_mode=code_mode, fga_gain=fga_gain)
        print(f"    Arith: {arith_acc:.0%}, Fact: {fact_acc:.0%}")
        configs[name] = {'arith': arith_acc, 'fact': fact_acc,
                         'surgery': do_surgery, 'code_mode': code_mode, 'fga_gain': fga_gain}
        del model; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase170_paradox.json'), 'w') as f:
        json.dump({'phase': '170', 'name': 'Arithmetic Paradox Resolution',
                   'configs': configs}, f, indent=2, default=str)

    # Plot: 2x4 grouped bar chart
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    names = [n for n, _, _, _ in conditions]
    labels = ['Base', 'Surgery\nonly', 'Surgery\n+Code', 'Surgery\n+FGA',
              'Surgery\n+Code+FGA', 'Code\nonly', 'FGA\nonly', 'Code\n+FGA']
    arith_vals = [configs[n]['arith'] for n in names]
    fact_vals = [configs[n]['fact'] for n in names]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x-w/2, arith_vals, w, label='Arithmetic', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x+w/2, fact_vals, w, label='Factual', color='#e74c3c', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w/2, arith_vals[i]+0.02, f'{arith_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i]+w/2, fact_vals[i]+0.02, f'{fact_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')

    # Highlight surgery conditions
    for i in range(1, 5):
        ax.axvspan(i-0.45, i+0.45, alpha=0.05, color='orange')

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=12); ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase 170: The Arithmetic Paradox Resolution\n'
                 'Does Code Mode protect arithmetic from Surgery damage?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase170_paradox.png'), dpi=150)
    plt.close()

    # Verdict
    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    b = configs['B_surgery_only']['arith']
    c = configs['C_surgery_code']['arith']
    d = configs['D_surgery_fga']['arith']
    e = configs['E_full_sas']['arith']
    print(f"  Surgery only:     arith = {b:.0%}")
    print(f"  Surgery + Code:   arith = {c:.0%}  (Code effect: {c-b:+.0%})")
    print(f"  Surgery + FGA:    arith = {d:.0%}  (FGA effect: {d-b:+.0%})")
    print(f"  Surgery + C + F:  arith = {e:.0%}  (Combined: {e-b:+.0%})")
    if c > b + 0.15:
        print("  -> CODE MODE IS THE PROTECTOR!")
    elif d > b + 0.15:
        print("  -> FGA IS THE PROTECTOR!")
    elif e > b + 0.15:
        print("  -> SYNERGISTIC PROTECTION!")
    else:
        print("  -> No clear protector found")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 170] Complete.")


if __name__ == '__main__':
    main()
