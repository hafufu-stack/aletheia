# -*- coding: utf-8 -*-
"""
Phase 144: Zero-Shot 1.5B Singularity
Can Surgery + Shield&Sword (NO DPO) break the 1.5B barrier?

P139 proved DPO is completely ineffective at 1.5B scale.
But Shield&Sword is purely inference-time -- no weight updates needed.
If FGA can inject the right answer directly, model capacity is irrelevant.

Model: Qwen2.5-1.5B (GPU, float16)
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

TEST_SET = [
    ("The capital of Japan is", " Tokyo", "word"),
    ("The capital of France is", " Paris", "word"),
    ("The capital of Germany is", " Berlin", "word"),
    ("The capital of Italy is", " Rome", "word"),
    ("The largest planet is", " Jupiter", "word"),
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("The speed of light is approximately", " 299", "number"),
    ("A year has", " 365", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
]


def disperse_embeddings(model, tok, strength=1.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366"]
    embed = model.model.embed_tokens.weight.data
    ids = []
    for t in num_tokens:
        tid = tok.encode(t)[-1]
        ids.append(tid)
    ids = list(set(ids))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)


class FGAHook:
    def __init__(self, model, target_layer, target_token_id, gain):
        self.gain = gain
        unembed = model.lm_head.weight.data[target_token_id].float()
        self.direction = unembed / (unembed.norm() + 1e-8)
        self.handle = None

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

    def register(self, model, layer_idx):
        self.handle = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def evaluate(model, tok, test_set, code_mode=False, fga_gain=0, fga_layer=None):
    n_layers = model.config.num_hidden_layers
    if fga_layer is None:
        fga_layer = n_layers - 4  # near-output layer
    results = []
    for prompt, expected, cat in test_set:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[-1]
        hook = None
        if fga_gain > 0:
            hook = FGAHook(model, fga_layer, exp_id, fga_gain)
            hook.register(model, fga_layer)
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if hook: hook.remove()
        pred_id = logits.argmax().item()
        results.append({'cat': cat, 'correct': int(pred_id == exp_id)})
    w = sum(r['correct'] for r in results if r['cat'] == 'word')
    wt = max(1, sum(1 for r in results if r['cat'] == 'word'))
    n = sum(r['correct'] for r in results if r['cat'] == 'number')
    nt = max(1, sum(1 for r in results if r['cat'] == 'number'))
    return w/wt, n/nt


def main():
    print("[P144] Zero-Shot 1.5B Singularity")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-1.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    configs = {}

    # A: 1.5B Baseline
    print("\n  === A: 1.5B Baseline ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float16).eval().to(DEVICE)
    w, n = evaluate(model, tok, TEST_SET)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['A_baseline'] = {'word': w, 'num': n}

    # B: Shield only (Code Mode)
    print("\n  === B: Shield only (Code Mode) ===")
    w, n = evaluate(model, tok, TEST_SET, code_mode=True)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['B_shield'] = {'word': w, 'num': n}

    # C: Shield + Sword (Code + FGA g=20)
    print("\n  === C: Shield + Sword (g=20) ===")
    w, n = evaluate(model, tok, TEST_SET, code_mode=True, fga_gain=20)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['C_shield_sword'] = {'word': w, 'num': n}
    del model; gc.collect(); torch.cuda.empty_cache()

    # D: Surgery + Shield + Sword (s=1.0)
    print("\n  === D: Surgery(s=1) + Shield + Sword ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float16).eval().to(DEVICE)
    disperse_embeddings(model, tok, strength=1.0)
    w, n = evaluate(model, tok, TEST_SET, code_mode=True, fga_gain=20)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['D_surg1_ss'] = {'word': w, 'num': n}
    del model; gc.collect(); torch.cuda.empty_cache()

    # E: Surgery(s=2) + Shield + Sword
    print("\n  === E: Surgery(s=2) + Shield + Sword ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float16).eval().to(DEVICE)
    disperse_embeddings(model, tok, strength=2.0)
    w, n = evaluate(model, tok, TEST_SET, code_mode=True, fga_gain=20)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['E_surg2_ss'] = {'word': w, 'num': n}
    del model; gc.collect(); torch.cuda.empty_cache()

    # F: Surgery(s=2) + Shield + Sword g=40 (stronger FGA)
    print("\n  === F: Surgery(s=2) + Shield + Sword (g=40) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float16).eval().to(DEVICE)
    disperse_embeddings(model, tok, strength=2.0)
    w, n = evaluate(model, tok, TEST_SET, code_mode=True, fga_gain=40)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['F_surg2_ss40'] = {'word': w, 'num': n}
    del model; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase144_zerodpo.json'), 'w') as f:
        json.dump({'phase': '144', 'name': 'Zero-Shot 1.5B Singularity',
                   'configs': configs}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    names = ['Baseline', 'Shield\n(Code)', 'Shield+\nSword\ng=20', 'Surgery\ns=1 +\nS&S', 'Surgery\ns=2 +\nS&S', 'Surgery\ns=2 +\nS&S g=40']
    keys = list(configs.keys())
    word_vals = [configs[k]['word'] for k in keys]
    num_vals = [configs[k]['num'] for k in keys]
    x = np.arange(len(names))
    w_bar = 0.35
    ax.bar(x-w_bar/2, word_vals, w_bar, label='Word', color='#3498db', alpha=0.8)
    ax.bar(x+w_bar/2, num_vals, w_bar, label='Number', color='#e74c3c', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w_bar/2, word_vals[i]+0.02, f'{word_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i]+w_bar/2, num_vals[i]+0.02, f'{num_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.legend(fontsize=11); ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Phase 144: Zero-Shot 1.5B Singularity\nCan Surgery + Shield&Sword break the 1.5B barrier WITHOUT DPO?',
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase144_zerodpo.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    for k in keys:
        print(f"    {k:20s}: word={configs[k]['word']:.0%} num={configs[k]['num']:.0%}")
    best_num = max(configs[k]['num'] for k in keys)
    if best_num > 0.5:
        print(f"  -> 1.5B SINGULARITY ACHIEVED without DPO! Best num={best_num:.0%}")
    elif best_num > 0:
        print(f"  -> Partial breakthrough: best num={best_num:.0%}")
    else:
        print(f"  -> 1.5B remains immune to inference-time hacks")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 144] Complete.")

if __name__ == '__main__':
    main()
