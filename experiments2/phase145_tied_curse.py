# -*- coding: utf-8 -*-
"""
Phase 145: The Tied-Embedding Curse (CPU experiment)
Does weight tying between input embeddings and lm_head explain
why GPT-2's L2 Distance Law fails?

Hypothesis: When embed_tokens and lm_head share weights, surgery
on input embeddings also distorts the output projection, creating
geometric self-contradiction that blocks DPO.

Test: Untie GPT-2's weights, then surgery + DPO.

Model: GPT-2 Small (CPU, float32) - parallel with GPU experiments
"""
import torch, json, os, gc, numpy as np, time
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import Parameter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = 'cpu'
NUM_TOKENS = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]

DPO_PAIRS = [
    ("Water freezes at", " 0", " 100"),
    ("The boiling point of water is", " 100", " 200"),
    ("The speed of light is", " 299", " 186"),
]

EVAL_FACTS = [
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The speed of light is", " 299", "number"),
    ("The capital of France is", " Paris", "word"),
    ("The capital of Japan is", " Tokyo", "word"),
]


def dpo_loss_gpt2(model, ref_model, tok, prompt, chosen, rejected, beta=0.1):
    text_c = prompt + chosen
    text_r = prompt + rejected
    inp_c = tok(text_c, return_tensors='pt')
    inp_r = tok(text_r, return_tensors='pt')
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


def untie_embeddings(model):
    """Untie input and output embeddings by cloning the weight."""
    if model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr():
        model.lm_head.weight = Parameter(model.lm_head.weight.clone())
        return True
    return False


def disperse_gpt2(model, tok, strength=1.0, input_only=True):
    """Disperse GPT-2's number embeddings."""
    embed = model.transformer.wte.weight.data
    ids = [tok.encode(t)[0] for t in NUM_TOKENS]
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()
    # Optionally also disperse lm_head (if untied)
    if not input_only and model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr():
        lm = model.lm_head.weight.data
        vecs_lm = lm[ids].clone().float()
        center_lm = vecs_lm.mean(dim=0)
        for i, idx in enumerate(ids):
            diff = vecs_lm[i] - center_lm
            direction = diff / (diff.norm() + 1e-8)
            lm[idx] += strength * direction * lm[idx].norm()


def gram_schmidt_gpt2(model, tok, input_only=True):
    embed = model.transformer.wte.weight.data
    ids = [tok.encode(t)[0] for t in NUM_TOKENS]
    vecs = embed[ids].clone().float()
    norms = vecs.norm(dim=-1, keepdim=True)
    ortho = torch.zeros_like(vecs)
    for i in range(len(vecs)):
        v = vecs[i].clone()
        for j in range(i):
            proj = torch.dot(v, ortho[j]) / (torch.dot(ortho[j], ortho[j]) + 1e-8)
            v = v - proj * ortho[j]
        ortho[i] = v / (v.norm() + 1e-8) * norms[i]
    for i, idx in enumerate(ids):
        embed[idx] = ortho[i].to(embed.dtype)


def scale_norms_gpt2(model, tok, factor):
    embed = model.transformer.wte.weight.data
    ids = [tok.encode(t)[0] for t in NUM_TOKENS]
    for idx in ids:
        embed[idx] *= factor


def train_eval_gpt2(model, ref, tok, n_layers=12, epochs=10):
    boundary = int(n_layers * 0.83)
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"h.{i}." in name and "mlp" in name: p.requires_grad = True
    # Also allow lm_head if untied
    if model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr():
        model.lm_head.weight.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable: return 0.0
    opt = torch.optim.AdamW(trainable, lr=5e-5)
    for epoch in range(epochs):
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss_gpt2(model, ref, tok, prompt, chosen, rejected)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
    model.eval()
    correct = 0
    for prompt, expected, _ in EVAL_FACTS:
        if _ == 'word': continue  # only test numbers
        inp = tok(prompt, return_tensors='pt')
        exp_id = tok.encode(expected)[0]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if logits.argmax().item() == exp_id: correct += 1
    return correct / max(1, sum(1 for _, _, c in EVAL_FACTS if c == 'number'))


def main():
    print("[P145] The Tied-Embedding Curse (CPU)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

    # Check if GPT-2 actually ties embeddings
    model_tmp = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    tied = model_tmp.lm_head.weight.data_ptr() == model_tmp.transformer.wte.weight.data_ptr()
    print(f"  GPT-2 embeddings tied: {tied}")
    del model_tmp; gc.collect()

    # Also check Qwen for comparison
    from transformers import AutoModelForCausalLM
    qwen_tmp = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', local_files_only=True)
    qwen_tied = qwen_tmp.lm_head.weight.data_ptr() == qwen_tmp.model.embed_tokens.weight.data_ptr()
    print(f"  Qwen-0.5B embeddings tied: {qwen_tied}")
    del qwen_tmp; gc.collect()

    configs = {}
    scale = 3.0  # Moderate scale for GPT-2

    # A: Tied (original) + GS + DPO (from P143 - should be 0%)
    print("\n  === A: Tied + GS + DPO ===")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    ref = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
    gram_schmidt_gpt2(model, tok); gram_schmidt_gpt2(ref, tok)
    scale_norms_gpt2(model, tok, scale); scale_norms_gpt2(ref, tok, scale)
    acc = train_eval_gpt2(model, ref, tok)
    print(f"    DPO num acc = {acc:.0%}")
    configs['A_tied_gs'] = {'acc': acc, 'tied': True}
    del model, ref; gc.collect()

    # B: Untied + GS (input only) + DPO
    print("\n  === B: Untied + GS(input) + DPO ===")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    ref = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
    untied_m = untie_embeddings(model); untied_r = untie_embeddings(ref)
    print(f"    Untied model: {untied_m}, ref: {untied_r}")
    gram_schmidt_gpt2(model, tok, input_only=True)
    gram_schmidt_gpt2(ref, tok, input_only=True)
    scale_norms_gpt2(model, tok, scale); scale_norms_gpt2(ref, tok, scale)
    acc = train_eval_gpt2(model, ref, tok)
    print(f"    DPO num acc = {acc:.0%}")
    configs['B_untied_gs_input'] = {'acc': acc, 'tied': False}
    del model, ref; gc.collect()

    # C: Untied + Disperse(input only) + DPO
    print("\n  === C: Untied + Disperse(input) + DPO ===")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    ref = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
    untie_embeddings(model); untie_embeddings(ref)
    disperse_gpt2(model, tok, strength=2.0, input_only=True)
    disperse_gpt2(ref, tok, strength=2.0, input_only=True)
    acc = train_eval_gpt2(model, ref, tok)
    print(f"    DPO num acc = {acc:.0%}")
    configs['C_untied_disp_input'] = {'acc': acc, 'tied': False}
    del model, ref; gc.collect()

    # D: Untied + Disperse(both independently) + DPO
    print("\n  === D: Untied + Disperse(both) + DPO ===")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    ref = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
    untie_embeddings(model); untie_embeddings(ref)
    disperse_gpt2(model, tok, strength=2.0, input_only=False)
    disperse_gpt2(ref, tok, strength=2.0, input_only=False)
    acc = train_eval_gpt2(model, ref, tok)
    print(f"    DPO num acc = {acc:.0%}")
    configs['D_untied_disp_both'] = {'acc': acc, 'tied': False}
    del model, ref; gc.collect()

    # E: Untied + Disperse(input) + DPO (more epochs=25)
    print("\n  === E: Untied + Disperse(input) + DPO(25ep) ===")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    ref = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
    untie_embeddings(model); untie_embeddings(ref)
    disperse_gpt2(model, tok, strength=2.0, input_only=True)
    disperse_gpt2(ref, tok, strength=2.0, input_only=True)
    acc = train_eval_gpt2(model, ref, tok, epochs=25)
    print(f"    DPO num acc = {acc:.0%}")
    configs['E_untied_disp_25ep'] = {'acc': acc, 'tied': False}
    del model, ref; gc.collect()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase145_tied_curse.json'), 'w') as f:
        json.dump({'phase': '145', 'name': 'Tied-Embedding Curse',
                   'configs': configs}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    names = ['Tied +\nGS', 'Untied +\nGS(in)', 'Untied +\nDisp(in)', 'Untied +\nDisp(both)', 'Untied +\nDisp(in)\n25ep']
    keys = list(configs.keys())
    acc_vals = [configs[k]['acc'] for k in keys]
    colors = ['#e74c3c' if configs[k].get('tied', False) else '#2ecc71' for k in keys]
    bars = ax.bar(names, acc_vals, color=colors, alpha=0.8, edgecolor='black', lw=1.2)
    for i, v in enumerate(acc_vals):
        ax.text(i, v + 0.02, f'{v:.0%}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('DPO Number Accuracy', fontsize=12)
    ax.set_title('Phase 145: Does Untying Embeddings Unlock GPT-2 DPO?\n'
                '(Red = Tied, Green = Untied)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase145_tied_curse.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    tied_acc = configs['A_tied_gs']['acc']
    best_untied = max(configs[k]['acc'] for k in keys if not configs[k].get('tied', False))
    if best_untied > tied_acc + 0.1:
        print(f"  -> TIED-EMBEDDING CURSE CONFIRMED! Tied={tied_acc:.0%}, Untied={best_untied:.0%}")
    else:
        print(f"  -> Tying is NOT the issue. Tied={tied_acc:.0%}, Untied={best_untied:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 145] Complete.")

if __name__ == '__main__':
    main()
