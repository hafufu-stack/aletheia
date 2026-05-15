# -*- coding: utf-8 -*-
"""
Phase 136: The Shield and Sword Protocol (FGA x Code Mode)
Deep Think's "Shield and Sword" hypothesis:
- FGA at L18 = "Sword" (inject truth into attention)
- Code Mode (#) = "Shield" (blind the grammar police)
Combining both should allow numerical tokens to pass through
the suppressor layers with zero friction.

No DPO, no surgery - pure inference-time intervention.

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

# Test prompts with expected answers
PROMPTS = [
    # Word facts
    ("The capital of Japan is", " Tokyo", "word"),
    ("The capital of France is", " Paris", "word"),
    ("The capital of Germany is", " Berlin", "word"),
    ("The largest planet is", " Jupiter", "word"),
    ("The chemical symbol for gold is", " Au", "word"),
    # Number facts
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("A year has", " 365", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
    ("A decade has", " 10", "number"),
]

# Code mode templates
CODE_TEMPLATES = [
    "# {prompt}",           # hash prefix
    "// {prompt}",          # C-style comment
    ">>> {prompt}",         # Python REPL
    "def answer(): return {prompt}",  # function context
]


class FGAHook:
    """Inject grounding signal into hidden states at target layer."""
    def __init__(self, model, tok, target_layer, target_token_id, gain):
        self.model = model
        self.tok = tok
        self.target_layer = target_layer
        self.target_token_id = target_token_id
        self.gain = gain
        self.handle = None
        # Get unembedding direction for target token
        unembed = model.lm_head.weight.data[target_token_id].float()
        self.direction = unembed / (unembed.norm() + 1e-8)

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0].float()
            if hidden.dim() == 3:
                hidden[:, -1, :] += self.gain * self.direction.to(hidden.device)
            elif hidden.dim() == 2:
                hidden[-1, :] += self.gain * self.direction.to(hidden.device)
            return (hidden.to(output[0].dtype),) + output[1:]
        else:
            hidden = output.float()
            if hidden.dim() == 3:
                hidden[:, -1, :] += self.gain * self.direction.to(hidden.device)
            elif hidden.dim() == 2:
                hidden[-1, :] += self.gain * self.direction.to(hidden.device)
            return hidden.to(output.dtype)

    def register(self):
        layer = self.model.model.layers[self.target_layer]
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def evaluate_config(model, tok, prompts, template=None, fga_layer=None, fga_gain=0):
    """Evaluate model with optional Code Mode and FGA."""
    results = []
    for prompt, expected, cat in prompts:
        # Apply code mode template
        if template:
            text = template.format(prompt=prompt)
        else:
            text = prompt

        exp_id = tok.encode(expected)[-1]

        # Setup FGA hook if needed
        hook = None
        if fga_layer is not None and fga_gain > 0:
            hook = FGAHook(model, tok, fga_layer, exp_id, fga_gain)
            hook.register()

        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()

        if hook: hook.remove()

        pred_id = logits.argmax().item()
        pred_tok = tok.decode([pred_id])
        is_correct = (pred_id == exp_id)
        results.append({
            'prompt': prompt[:40], 'cat': cat, 'correct': is_correct,
            'pred': pred_tok.encode('ascii', 'replace').decode().strip(),
        })
    return results


def main():
    print("[P136] Shield and Sword Protocol: FGA x Code Mode")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

    configs = {}
    gains = [0, 5, 10, 20]

    # A: Baseline (no intervention)
    print("\n  === A: Baseline ===")
    r = evaluate_config(model, tok, PROMPTS)
    w_acc = sum(x['correct'] for x in r if x['cat']=='word') / max(1, sum(1 for x in r if x['cat']=='word'))
    n_acc = sum(x['correct'] for x in r if x['cat']=='number') / max(1, sum(1 for x in r if x['cat']=='number'))
    print(f"    Word: {w_acc:.0%}, Num: {n_acc:.0%}")
    configs['baseline'] = {'word': w_acc, 'num': n_acc, 'details': r}

    # B: Code Mode only (each template)
    for i, tpl in enumerate(CODE_TEMPLATES):
        name = f"code_{i}"
        print(f"\n  === B: Code Mode '{tpl[:15]}...' ===")
        r = evaluate_config(model, tok, PROMPTS, template=tpl)
        w_acc = sum(x['correct'] for x in r if x['cat']=='word') / max(1, sum(1 for x in r if x['cat']=='word'))
        n_acc = sum(x['correct'] for x in r if x['cat']=='number') / max(1, sum(1 for x in r if x['cat']=='number'))
        print(f"    Word: {w_acc:.0%}, Num: {n_acc:.0%}")
        configs[name] = {'word': w_acc, 'num': n_acc, 'template': tpl}

    # C: FGA only at L18 (various gains)
    for g in gains:
        if g == 0: continue
        name = f"fga_g{g}"
        print(f"\n  === C: FGA L18 g={g} ===")
        r = evaluate_config(model, tok, PROMPTS, fga_layer=18, fga_gain=g)
        w_acc = sum(x['correct'] for x in r if x['cat']=='word') / max(1, sum(1 for x in r if x['cat']=='word'))
        n_acc = sum(x['correct'] for x in r if x['cat']=='number') / max(1, sum(1 for x in r if x['cat']=='number'))
        print(f"    Word: {w_acc:.0%}, Num: {n_acc:.0%}")
        configs[name] = {'word': w_acc, 'num': n_acc}

    # D: Shield + Sword (Code Mode + FGA combined)
    best_code = 0  # use # prefix
    best_tpl = CODE_TEMPLATES[best_code]
    for g in gains:
        if g == 0: continue
        name = f"shield_sword_g{g}"
        print(f"\n  === D: Shield+Sword g={g} ===")
        r = evaluate_config(model, tok, PROMPTS, template=best_tpl,
                          fga_layer=18, fga_gain=g)
        w_acc = sum(x['correct'] for x in r if x['cat']=='word') / max(1, sum(1 for x in r if x['cat']=='word'))
        n_acc = sum(x['correct'] for x in r if x['cat']=='number') / max(1, sum(1 for x in r if x['cat']=='number'))
        print(f"    Word: {w_acc:.0%}, Num: {n_acc:.0%}")
        configs[name] = {'word': w_acc, 'num': n_acc}

    # Save
    out = {'phase': '136', 'name': 'Shield and Sword Protocol', 'configs': {}}
    for k, v in configs.items():
        out['configs'][k] = {kk: vv for kk, vv in v.items() if kk != 'details'}
    with open(os.path.join(RESULTS_DIR, 'phase136_shield_sword.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: All configs comparison
    ax = axes[0]
    names = list(configs.keys())
    word_accs = [configs[n]['word'] for n in names]
    num_accs = [configs[n]['num'] for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x-w/2, word_accs, w, label='Word', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, num_accs, w, label='Number', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.legend(); ax.set_ylabel('Accuracy')
    ax.set_title('All Configurations', fontweight='bold')
    ax.set_ylim(0, 1.2)

    # Panel 2: FGA gain curve with/without Code Mode
    ax = axes[1]
    fga_gains_plot = [0] + [g for g in gains if g > 0]
    fga_only = [configs.get('baseline', {}).get('num', 0)]
    shield_sword = [configs.get(f'code_{best_code}', {}).get('num', 0)]
    for g in gains:
        if g == 0: continue
        fga_only.append(configs.get(f'fga_g{g}', {}).get('num', 0))
        shield_sword.append(configs.get(f'shield_sword_g{g}', {}).get('num', 0))
    ax.plot(fga_gains_plot, fga_only, 'r-o', label='FGA Only (Sword)', lw=2)
    ax.plot(fga_gains_plot, shield_sword, 'g-s', label='Code+FGA (Shield+Sword)', lw=2)
    ax.set_xlabel('FGA Gain'); ax.set_ylabel('Number Accuracy')
    ax.legend(); ax.set_title('Shield+Sword Synergy', fontweight='bold')

    fig.suptitle('Phase 136: Shield and Sword Protocol (FGA x Code Mode)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase136_shield_sword.png'), dpi=150)
    plt.close()

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 136] Complete.")

if __name__ == '__main__':
    main()
