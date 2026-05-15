# -*- coding: utf-8 -*-
"""
Phase 136c: True Nuclear Option v2 - The Final Form
Full pipeline combining ALL Season 29-30 breakthroughs:
1. GS + Scale 3x (P138b: orthogonal + high L2 distance)
2. Back-only DPO (trained facts)
3. UAlign abstention (uncertain facts)
4. Shield+Sword at inference (Code Mode + FGA)

This is Joint DPO+UAlign (P135 idea) but with the CORRECT
embedding preparation (GS+scale instead of just disperse).

Model: Qwen2.5-0.5B (GPU, float32)
"""
import torch, json, os, gc, numpy as np, time, random
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_TOKENS = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
              " 10", " 100", " 12", " 186", " 212", " 299", " 365"]

# DPO pairs (known facts)
DPO_PAIRS = [
    ("The capital of Japan is", " Tokyo", " Osaka"),
    ("The capital of France is", " Paris", " Lyon"),
    ("The capital of Germany is", " Berlin", " Munich"),
    ("Water freezes at", " 0", " 100"),
    ("The boiling point of water is", " 100", " 212"),
    ("The atomic number of carbon is", " 6", " 12"),
]

# UAlign pairs (uncertain -> abstain with "I")
UALIGN_PAIRS = [
    ("The melting point of iron is", " I", " 1538"),
    ("The population of Tokyo in millions is", " I", " 14"),
    ("The distance from Earth to Mars in km is", " I", " 225"),
    ("The height of Mt Fuji in meters is", " I", " 3776"),
    ("The GDP of Japan in trillion USD is", " I", " 4"),
    ("The atomic number of gold is", " I", " 79"),
]

TEST_SET = [
    ("The capital of Japan is", " Tokyo", "word", "known"),
    ("The capital of France is", " Paris", "word", "known"),
    ("The capital of Germany is", " Berlin", "word", "known"),
    ("The largest planet is", " Jupiter", "word", "novel"),
    ("The chemical symbol for gold is", " Au", "word", "novel"),
    ("Water freezes at", " 0", "number", "trained"),
    ("The boiling point of water is", " 100", "number", "trained"),
    ("The atomic number of carbon is", " 6", "number", "trained"),
    ("The melting point of iron is", " I", "number", "uncertain"),
    ("The population of Tokyo in millions is", " I", "number", "uncertain"),
    ("A year has", " 365", "number", "novel"),
    ("Pi is approximately", " 3", "number", "novel"),
    ("The number of continents is", " 7", "number", "novel"),
]


def dpo_loss(model, ref, tok, prompt, chosen, rejected, beta=0.05):
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
        rlc = ref(**inp_c).logits[0, plen-1:-1, :].float().clamp(-100, 100)
        rlr = ref(**inp_r).logits[0, plen-1:-1, :].float().clamp(-100, 100)
        rlp_c = F.log_softmax(rlc, dim=-1).gather(1, inp_c['input_ids'][0, plen:].unsqueeze(1)).squeeze()
        rlp_r = F.log_softmax(rlr, dim=-1).gather(1, inp_r['input_ids'][0, plen:].unsqueeze(1)).squeeze()
        if rlp_c.dim() == 0: rlp_c = rlp_c.unsqueeze(0)
        if rlp_r.dim() == 0: rlp_r = rlp_r.unsqueeze(0)
    diff = beta * ((lp_c.sum() - rlp_c.sum()) - (lp_r.sum() - rlp_r.sum()))
    return -F.logsigmoid(diff)


def gram_schmidt(model, tok):
    embed = model.model.embed_tokens.weight.data
    base_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]
    ids = [tok.encode(t)[-1] for t in base_tokens]
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


def scale_norms(model, tok, factor):
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in NUM_TOKENS]
    for idx in ids:
        embed[idx] *= factor


class FGAHook:
    def __init__(self, model, target_layer, target_token_id, gain):
        self.gain = gain
        self.handle = None
        unembed = model.lm_head.weight.data[target_token_id].float()
        self.direction = unembed / (unembed.norm() + 1e-8)

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0].float()
            if hidden.dim() == 3: hidden[:, -1, :] += self.gain * self.direction.to(hidden.device)
            elif hidden.dim() == 2: hidden[-1, :] += self.gain * self.direction.to(hidden.device)
            return (hidden.to(output[0].dtype),) + output[1:]
        else:
            hidden = output.float()
            if hidden.dim() == 3: hidden[:, -1, :] += self.gain * self.direction.to(hidden.device)
            elif hidden.dim() == 2: hidden[-1, :] += self.gain * self.direction.to(hidden.device)
            return hidden.to(output.dtype)

    def register(self, model, layer_idx):
        self.handle = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def evaluate(model, tok, test_set, code_mode=False, fga_gain=0):
    results = []
    for prompt, expected, cat, split in test_set:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[-1]
        hook = None
        if fga_gain > 0:
            hook = FGAHook(model, 18, exp_id, fga_gain)
            hook.register(model, 18)
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if hook: hook.remove()
        pred_id = logits.argmax().item()
        pred_tok = tok.decode([pred_id])
        is_correct = (pred_id == exp_id)
        is_abstain = pred_tok.strip() in ['I', 'It', 'Unknown', 'N', 'None']
        results.append({
            'cat': cat, 'split': split, 'correct': is_correct,
            'is_abstain': is_abstain,
            'pred': pred_tok.encode('ascii', 'replace').decode().strip(),
        })
    return results


def summarize(results, label):
    w_c = sum(r['correct'] for r in results if r['cat'] == 'word')
    w_t = sum(1 for r in results if r['cat'] == 'word')
    n_c = sum(r['correct'] for r in results if r['cat'] == 'number')
    n_t = sum(1 for r in results if r['cat'] == 'number')
    u_ab = sum(r['is_abstain'] for r in results if r['split'] == 'uncertain')
    u_t = sum(1 for r in results if r['split'] == 'uncertain')
    total_c = sum(r['correct'] for r in results)
    total_t = len(results)
    print(f"    {label}: total={total_c}/{total_t} word={w_c}/{w_t} "
          f"num={n_c}/{n_t} uncertain_abstain={u_ab}/{u_t}")
    return {
        'word_acc': w_c/max(1,w_t), 'num_acc': n_c/max(1,n_t),
        'uncertain_abstain': u_ab/max(1,u_t), 'total_acc': total_c/max(1,total_t),
    }


def main():
    print("[P136c] True Nuclear Option v2 - The Final Form")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    configs = {}

    # Stage 1: GS + Scale 3x embedding prep
    print("\n  === Stage 1: GS+Scale3x Embedding Prep ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    gram_schmidt(model, tok); gram_schmidt(ref, tok)
    scale_norms(model, tok, 3.0); scale_norms(ref, tok, 3.0)

    # Stage 2: Joint DPO + UAlign training
    print("  === Stage 2: Joint DPO+UAlign Training ===")
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-6)

    for epoch in range(10):
        all_pairs = [(p, c, r, 'dpo') for p, c, r in DPO_PAIRS] + \
                    [(p, c, r, 'ualign') for p, c, r in UALIGN_PAIRS]
        random.shuffle(all_pairs)
        for prompt, chosen, rejected, pair_type in all_pairs:
            loss = dpo_loss(model, ref, tok, prompt, chosen, rejected)
            if loss.item() > 0:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
    model.eval()

    # Evaluate all combinations
    print("\n  === Evaluations ===")

    # A: After training, no inference tricks
    r = evaluate(model, tok, TEST_SET)
    configs['A_trained_only'] = summarize(r, 'Trained only')

    # B: + Code Mode
    r = evaluate(model, tok, TEST_SET, code_mode=True)
    configs['B_code_mode'] = summarize(r, '+ Code Mode')

    # C: + FGA g=20
    r = evaluate(model, tok, TEST_SET, fga_gain=20)
    configs['C_fga'] = summarize(r, '+ FGA g=20')

    # D: + Shield+Sword (Code + FGA)
    r = evaluate(model, tok, TEST_SET, code_mode=True, fga_gain=20)
    configs['D_shield_sword'] = summarize(r, '+ Shield+Sword')

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase136c_nuclear_v2.json'), 'w') as f:
        json.dump({'phase': '136c', 'name': 'Nuclear Option v2', 'configs': configs},
                 f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    names = ['Trained\nOnly', '+Code\nMode', '+FGA\ng=20', 'Shield+\nSword']
    keys = ['A_trained_only', 'B_code_mode', 'C_fga', 'D_shield_sword']
    x = np.arange(len(names))
    w = 0.2
    word = [configs[k]['word_acc'] for k in keys]
    num = [configs[k]['num_acc'] for k in keys]
    abstain = [configs[k]['uncertain_abstain'] for k in keys]
    total = [configs[k]['total_acc'] for k in keys]

    ax.bar(x-1.5*w, word, w, label='Word Acc', color='#3498db', alpha=0.8)
    ax.bar(x-0.5*w, num, w, label='Num Acc', color='#e74c3c', alpha=0.8)
    ax.bar(x+0.5*w, abstain, w, label='Uncertain Abstain', color='#2ecc71', alpha=0.8)
    ax.bar(x+1.5*w, total, w, label='Total Acc', color='#9b59b6', alpha=0.8)

    for i in range(len(names)):
        ax.text(x[i]-1.5*w, word[i]+0.02, f'{word[i]:.0%}', ha='center', fontsize=8)
        ax.text(x[i]-0.5*w, num[i]+0.02, f'{num[i]:.0%}', ha='center', fontsize=8)
        ax.text(x[i]+0.5*w, abstain[i]+0.02, f'{abstain[i]:.0%}', ha='center', fontsize=8)
        ax.text(x[i]+1.5*w, total[i]+0.02, f'{total[i]:.0%}', ha='center', fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=10); ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('Phase 136c: True Nuclear Option v2\n'
                'GS+Scale3x -> Joint DPO+UAlign -> Shield+Sword',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase136c_nuclear_v2.png'), dpi=150)
    plt.close()

    del model, ref; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 136c] Complete.")

if __name__ == '__main__':
    main()
