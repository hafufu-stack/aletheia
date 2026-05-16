# -*- coding: utf-8 -*-
"""
Phase 155: Dual Surgery - The 14B Breakthrough
P154 revealed: 14B's lm_head cos = 0.74 (severely clustered).
FGA can't focus because the OUTPUT projection vectors are entangled.

Solution: Disperse BOTH embed_tokens AND lm_head independently.

Model: Qwen2.5-14B (GPU, 4-bit)
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
    ("The largest planet is", " Jupiter", "word"),
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("The speed of light is approximately", " 299", "number"),
    ("A year has", " 365", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
]


def disperse_embeddings(model, tok, strength=2.0):
    """Disperse input embeddings (embed_tokens)."""
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366"]
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in num_tokens))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)
    return ids


def disperse_lm_head(model, tok, strength=2.0):
    """Disperse output projection (lm_head) for number tokens."""
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366"]
    lm = model.lm_head.weight.data
    ids = list(set(tok.encode(t)[-1] for t in num_tokens))
    vecs = lm[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        lm[idx] += (strength * direction * lm[idx].float().norm()).to(lm.dtype)
    # Verify
    post_vecs = lm[ids].float()
    cos = F.cosine_similarity(post_vecs.unsqueeze(0), post_vecs.unsqueeze(1), dim=-1)
    avg_cos = cos[~torch.eye(len(ids), dtype=bool)].mean().item()
    return avg_cos


class FGAHook:
    def __init__(self, model, target_token_id, gain):
        self.gain = gain
        unembed = model.lm_head.weight.data[target_token_id].float()
        self.direction = unembed / (unembed.norm() + 1e-8)
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        else:
            h = output.float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return h.to(output.dtype)

    def register(self, model, layer_idx):
        self.handle = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def evaluate(model, tok, test_set, code_mode=True, fga_gain=20):
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - 4
    results = []
    for prompt, expected, cat in test_set:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[-1]
        hook = FGAHook(model, exp_id, fga_gain)
        hook.register(model, fga_layer)
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        hook.remove()
        pred_id = logits.argmax().item()
        results.append({'cat': cat, 'correct': int(pred_id == exp_id),
                        'expected': expected, 'pred': tok.decode([pred_id])})
    w = sum(r['correct'] for r in results if r['cat'] == 'word')
    wt = max(1, sum(1 for r in results if r['cat'] == 'word'))
    n = sum(r['correct'] for r in results if r['cat'] == 'number')
    nt = max(1, sum(1 for r in results if r['cat'] == 'number'))
    return w/wt, n/nt, results


def main():
    print("[P155] Dual Surgery - The 14B Breakthrough")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = 'Qwen/Qwen2.5-14B'
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except:
        print("  14B not available, aborting")
        return
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16)

    configs = {}

    # A: Embed only (from P149 - should be 0%)
    print("\n  === A: Embed surgery only + S&S ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    disperse_embeddings(model, tok, strength=2.0)
    w, n, details = evaluate(model, tok, TEST_SET, fga_gain=20)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['A_embed_only'] = {'word': w, 'num': n}
    del model; gc.collect(); torch.cuda.empty_cache()

    # B: LM_head only + S&S
    print("\n  === B: LM_head surgery only + S&S ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    lm_cos = disperse_lm_head(model, tok, strength=2.0)
    print(f"    LM_head cos after surgery: {lm_cos:.4f}")
    w, n, details = evaluate(model, tok, TEST_SET, fga_gain=20)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['B_lmhead_only'] = {'word': w, 'num': n, 'lm_cos': lm_cos}
    del model; gc.collect(); torch.cuda.empty_cache()

    # C: DUAL surgery (embed + lm_head) + S&S g=20
    print("\n  === C: Dual Surgery + S&S g=20 ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    disperse_embeddings(model, tok, strength=2.0)
    lm_cos = disperse_lm_head(model, tok, strength=2.0)
    print(f"    LM_head cos: {lm_cos:.4f}")
    w, n, details = evaluate(model, tok, TEST_SET, fga_gain=20)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    for d in details:
        if d['cat'] == 'number':
            print(f"      {d['expected']:8s} -> {d['pred']:10s} {'OK' if d['correct'] else 'MISS'}")
    configs['C_dual_g20'] = {'word': w, 'num': n, 'lm_cos': lm_cos}
    del model; gc.collect(); torch.cuda.empty_cache()

    # D: Dual surgery + S&S g=50
    print("\n  === D: Dual Surgery + S&S g=50 ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    disperse_embeddings(model, tok, strength=2.0)
    lm_cos = disperse_lm_head(model, tok, strength=2.0)
    w, n, details = evaluate(model, tok, TEST_SET, fga_gain=50)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    for d in details:
        if d['cat'] == 'number':
            print(f"      {d['expected']:8s} -> {d['pred']:10s} {'OK' if d['correct'] else 'MISS'}")
    configs['D_dual_g50'] = {'word': w, 'num': n}
    del model; gc.collect(); torch.cuda.empty_cache()

    # E: Dual surgery (stronger s=5) + S&S g=50
    print("\n  === E: Dual Surgery(s=5) + S&S g=50 ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    disperse_embeddings(model, tok, strength=5.0)
    lm_cos = disperse_lm_head(model, tok, strength=5.0)
    print(f"    LM_head cos: {lm_cos:.4f}")
    w, n, details = evaluate(model, tok, TEST_SET, fga_gain=50)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    for d in details:
        if d['cat'] == 'number':
            print(f"      {d['expected']:8s} -> {d['pred']:10s} {'OK' if d['correct'] else 'MISS'}")
    configs['E_dual5_g50'] = {'word': w, 'num': n, 'lm_cos': lm_cos}
    del model; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase155_dual.json'), 'w') as f:
        json.dump({'phase': '155', 'name': 'Dual Surgery 14B',
                   'configs': configs}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    names = ['Embed\nonly', 'LM_head\nonly', 'Dual\ng=20', 'Dual\ng=50', 'Dual(s=5)\ng=50']
    keys = list(configs.keys())
    num_vals = [configs[k]['num'] for k in keys]
    word_vals = [configs[k]['word'] for k in keys]
    x = np.arange(len(names)); w_bar = 0.35
    ax.bar(x-w_bar/2, word_vals, w_bar, label='Word', color='#3498db', alpha=0.8)
    ax.bar(x+w_bar/2, num_vals, w_bar, label='Number', color='#e74c3c', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w_bar/2, word_vals[i]+0.02, f'{word_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
        ax.text(x[i]+w_bar/2, num_vals[i]+0.02, f'{num_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=11); ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Phase 155: Dual Surgery on 14B\nCan dispersing BOTH embed + lm_head break 14B?',
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase155_dual.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    best = max(configs.items(), key=lambda x: x[1]['num'])
    if best[1]['num'] > 0.5:
        print(f"  -> 14B CONQUERED! {best[0]}: num={best[1]['num']:.0%}")
    elif best[1]['num'] > 0:
        print(f"  -> Partial progress: {best[0]}: num={best[1]['num']:.0%}")
    else:
        print(f"  -> 14B still immune even with dual surgery")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 155] Complete.")

if __name__ == '__main__':
    main()
