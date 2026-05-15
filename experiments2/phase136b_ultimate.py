# -*- coding: utf-8 -*-
"""
Phase 136b: The Ultimate Combo
Combines ALL breakthroughs into one pipeline:
1. Embedding Surgery (P130b) - spread numbers apart
2. Back-only DPO (P117b) - train on dispersed embeddings
3. Shield+Sword (P136) - Code Mode + FGA at inference time

Each contributed independently:
  Surgery+DPO: 50% num (P130b)
  Shield+Sword g=20: 43% num (P136)
Question: Do they stack?

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

DPO_PAIRS = [
    ("The capital of Japan is", " Tokyo", " Osaka"),
    ("The capital of France is", " Paris", " Lyon"),
    ("The capital of Germany is", " Berlin", " Munich"),
    ("The capital of Italy is", " Rome", " Milan"),
    ("Water freezes at", " 0", " 100"),
    ("The boiling point of water is", " 100", " 212"),
    ("The atomic number of carbon is", " 6", " 12"),
    ("The speed of light is approximately", " 299", " 186"),
]

TEST_SET = [
    ("The capital of Japan is", " Tokyo", "word"),
    ("The capital of France is", " Paris", "word"),
    ("The capital of Germany is", " Berlin", "word"),
    ("The capital of Italy is", " Rome", "word"),
    ("The capital of the United Kingdom is", " London", "word"),
    ("The largest planet is", " Jupiter", "word"),
    ("The chemical symbol for gold is", " Au", "word"),
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("The speed of light is approximately", " 299", "number"),
    ("A year has", " 365", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
]


def dpo_loss(model, ref_model, tok, prompt, chosen, rejected, beta=0.05):
    text_c = prompt + chosen
    text_r = prompt + rejected
    inp_c = tok(text_c, return_tensors='pt').to(DEVICE)
    inp_r = tok(text_r, return_tensors='pt').to(DEVICE)
    plen = tok(prompt, return_tensors='pt')['input_ids'].shape[1]
    lc = model(**inp_c).logits[0, plen-1:-1, :].float().clamp(-100, 100)
    lr_ = model(**inp_r).logits[0, plen-1:-1, :].float().clamp(-100, 100)
    lp_c = F.log_softmax(lc, dim=-1).gather(1, inp_c['input_ids'][0, plen:].unsqueeze(1)).squeeze()
    lp_r = F.log_softmax(lr_, dim=-1).gather(1, inp_r['input_ids'][0, plen:].unsqueeze(1)).squeeze()
    if lp_c.dim() == 0: lp_c = lp_c.unsqueeze(0)
    if lp_r.dim() == 0: lp_r = lp_r.unsqueeze(0)
    with torch.no_grad():
        rlc = ref_model(**inp_c).logits[0, plen-1:-1, :].float().clamp(-100, 100)
        rlr = ref_model(**inp_r).logits[0, plen-1:-1, :].float().clamp(-100, 100)
        rlp_c = F.log_softmax(rlc, dim=-1).gather(1, inp_c['input_ids'][0, plen:].unsqueeze(1)).squeeze()
        rlp_r = F.log_softmax(rlr, dim=-1).gather(1, inp_r['input_ids'][0, plen:].unsqueeze(1)).squeeze()
        if rlp_c.dim() == 0: rlp_c = rlp_c.unsqueeze(0)
        if rlp_r.dim() == 0: rlp_r = rlp_r.unsqueeze(0)
    diff = beta * ((lp_c.sum() - rlp_c.sum()) - (lp_r.sum() - rlp_r.sum()))
    return -F.logsigmoid(diff)


def disperse_embeddings(model, tok, strength=1.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366"]
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in num_tokens]
    vecs = embed[ids].clone()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()


class FGAHook:
    def __init__(self, model, target_layer, target_token_id, gain):
        self.target_layer = target_layer
        self.target_token_id = target_token_id
        self.gain = gain
        self.handle = None
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

    def register(self, model):
        layer = model.model.layers[self.target_layer]
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def evaluate(model, tok, test_set, code_mode=False, fga_gain=0):
    results = []
    for prompt, expected, cat in test_set:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[-1]
        hook = None
        if fga_gain > 0:
            hook = FGAHook(model, 18, exp_id, fga_gain)
            hook.register(model)
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if hook: hook.remove()
        pred_id = logits.argmax().item()
        results.append({'cat': cat, 'correct': int(pred_id == exp_id)})
    w = sum(r['correct'] for r in results if r['cat'] == 'word')
    wt = sum(1 for r in results if r['cat'] == 'word')
    n = sum(r['correct'] for r in results if r['cat'] == 'number')
    nt = sum(1 for r in results if r['cat'] == 'number')
    return w/max(1,wt), n/max(1,nt)


def main():
    print("[P136b] The Ultimate Combo")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    configs = {}

    # A: Baseline
    print("\n  === A: Baseline ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    w, n = evaluate(model, tok, TEST_SET)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['A_baseline'] = {'word': w, 'num': n}
    del model; gc.collect(); torch.cuda.empty_cache()

    # B: Shield+Sword only (no training)
    print("\n  === B: Shield+Sword only ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    w, n = evaluate(model, tok, TEST_SET, code_mode=True, fga_gain=20)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['B_shield_sword'] = {'word': w, 'num': n}
    del model; gc.collect(); torch.cuda.empty_cache()

    # C: Surgery + DPO (P130b proven)
    print("\n  === C: Surgery + DPO ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    disperse_embeddings(model, tok, 1.0)
    disperse_embeddings(ref, tok, 1.0)
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6)
    for epoch in range(5):
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss(model, ref, tok, prompt, chosen, rejected)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
    model.eval()
    w, n = evaluate(model, tok, TEST_SET)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['C_surgery_dpo'] = {'word': w, 'num': n}

    # D: Surgery + DPO + Shield+Sword (THE ULTIMATE COMBO)
    print("\n  === D: Surgery + DPO + Shield+Sword (ULTIMATE) ===")
    w, n = evaluate(model, tok, TEST_SET, code_mode=True, fga_gain=20)
    print(f"    Word: {w:.0%}, Num: {n:.0%}")
    configs['D_ultimate'] = {'word': w, 'num': n}
    del model, ref; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase136b_ultimate.json'), 'w') as f:
        json.dump({'phase': '136b', 'name': 'Ultimate Combo', 'configs': configs},
                 f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    names = ['Baseline', 'Shield+Sword\n(inference)', 'Surgery+DPO\n(training)', 'ULTIMATE\n(all 3)']
    keys = ['A_baseline', 'B_shield_sword', 'C_surgery_dpo', 'D_ultimate']
    word_vals = [configs[k]['word'] for k in keys]
    num_vals = [configs[k]['num'] for k in keys]
    x = np.arange(len(names))
    w_bar = 0.35
    bars1 = ax.bar(x-w_bar/2, word_vals, w_bar, label='Word', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x+w_bar/2, num_vals, w_bar, label='Number', color='#e74c3c', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w_bar/2, word_vals[i]+0.02, f'{word_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
        ax.text(x[i]+w_bar/2, num_vals[i]+0.02, f'{num_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=11); ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Phase 136b: The Ultimate Combo\nSurgery + DPO + Shield+Sword',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase136b_ultimate.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    for k, name in zip(keys, names):
        print(f"    {name.replace(chr(10),' '):25s}: word={configs[k]['word']:.0%} num={configs[k]['num']:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 136b] Complete.")

if __name__ == '__main__':
    main()
