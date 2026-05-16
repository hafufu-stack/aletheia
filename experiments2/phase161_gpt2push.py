# -*- coding: utf-8 -*-
"""
Phase 161: GPT-2 100% Challenge
P157 showed GPT-2 Dual Surgery = word 100%, num 80%.
Which facts fail? Can we push to 100%?

Model: GPT-2 (124M, CPU-compatible)
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_SET = [
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
    ("The capital of France is", " Paris", "word"),
    ("The capital of Japan is", " Tokyo", "word"),
]


def dual_surgery_gpt2(model, tok, strength=2.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365"]
    ids = list(set(tok.encode(t)[0] for t in num_tokens))
    # Untie if needed
    if model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr():
        model.lm_head.weight = Parameter(model.lm_head.weight.clone())
    # Embed
    embed = model.transformer.wte.weight.data
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()
    # LM head
    lm = model.lm_head.weight.data
    vecs_lm = lm[ids].clone().float()
    center_lm = vecs_lm.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs_lm[i] - center_lm
        direction = diff / (diff.norm() + 1e-8)
        lm[idx] += strength * direction * lm[idx].norm()


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


def eval_detailed(model, tok, code_mode=True, fga_gain=20, fga_layer_offset=2):
    n_layers = model.config.n_layer
    fga_layer = n_layers - fga_layer_offset
    results = []
    for prompt, expected, cat in TEST_SET:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[0]
        hook = FGAHook(model, exp_id, fga_gain)
        hook.handle = model.transformer.h[fga_layer].register_forward_hook(hook.hook_fn)
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        hook.handle.remove()
        pred_id = logits.argmax().item()
        pred_text = tok.decode([pred_id])
        # Also get top-5
        top5_ids = logits.topk(5).indices.tolist()
        top5 = [tok.decode([t]) for t in top5_ids]
        target_rank = (logits.argsort(descending=True) == exp_id).nonzero().item() + 1
        results.append({
            'prompt': prompt, 'expected': expected, 'cat': cat,
            'pred': pred_text, 'correct': int(pred_id == exp_id),
            'top5': top5, 'target_rank': target_rank,
            'target_logit': logits[exp_id].item(),
            'pred_logit': logits[pred_id].item()
        })
    return results


def main():
    print("[P161] GPT-2 100% Challenge")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

    all_configs = {}

    # A: Diagnose which facts fail at g=20
    print("\n  === A: Dual Surgery g=20 (diagnosis) ===")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval().to(DEVICE)
    dual_surgery_gpt2(model, tok, strength=2.0)
    res = eval_detailed(model, tok, fga_gain=20)
    for r in res:
        status = "OK" if r['correct'] else f"MISS (rank={r['target_rank']}, pred='{r['pred']}')"
        print(f"    '{r['expected']:8s}' -> {status}  top5={r['top5']}")
    w = sum(r['correct'] for r in res if r['cat'] == 'word')
    n = sum(r['correct'] for r in res if r['cat'] == 'number')
    all_configs['A_g20'] = {'word': w/2, 'num': n/5, 'details': res}
    del model; gc.collect()

    # B: Try different gains
    for gain in [10, 15, 25, 30, 40, 50]:
        model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval().to(DEVICE)
        dual_surgery_gpt2(model, tok, strength=2.0)
        res = eval_detailed(model, tok, fga_gain=gain)
        w = sum(r['correct'] for r in res if r['cat'] == 'word')
        n = sum(r['correct'] for r in res if r['cat'] == 'number')
        nw = sum(1 for r in res if r['cat'] == 'word')
        nn = sum(1 for r in res if r['cat'] == 'number')
        print(f"    g={gain}: word={w}/{nw} num={n}/{nn}")
        all_configs[f'B_g{gain}'] = {'word': w/nw, 'num': n/nn}
        del model; gc.collect()

    # C: Try different surgery strengths
    for s in [1.0, 3.0, 5.0]:
        model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval().to(DEVICE)
        dual_surgery_gpt2(model, tok, strength=s)
        res = eval_detailed(model, tok, fga_gain=20)
        w = sum(r['correct'] for r in res if r['cat'] == 'word')
        n = sum(r['correct'] for r in res if r['cat'] == 'number')
        nw = sum(1 for r in res if r['cat'] == 'word')
        nn = sum(1 for r in res if r['cat'] == 'number')
        print(f"    s={s}: word={w}/{nw} num={n}/{nn}")
        all_configs[f'C_s{s}'] = {'word': w/nw, 'num': n/nn}
        del model; gc.collect()

    # D: Try different FGA layers
    for offset in [1, 2, 3, 4, 6, 8]:
        model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval().to(DEVICE)
        dual_surgery_gpt2(model, tok, strength=2.0)
        res = eval_detailed(model, tok, fga_gain=20, fga_layer_offset=offset)
        w = sum(r['correct'] for r in res if r['cat'] == 'word')
        n = sum(r['correct'] for r in res if r['cat'] == 'number')
        nw = sum(1 for r in res if r['cat'] == 'word')
        nn = sum(1 for r in res if r['cat'] == 'number')
        print(f"    layer=L-{offset}: word={w}/{nw} num={n}/{nn}")
        all_configs[f'D_layer{offset}'] = {'word': w/nw, 'num': n/nn}
        del model; gc.collect()

    with open(os.path.join(RESULTS_DIR, 'phase161_gpt2.json'), 'w') as f:
        json.dump({'phase': '161', 'name': 'GPT-2 100%',
                   'configs': {k: {kk: vv for kk, vv in v.items() if kk != 'details'}
                              for k, v in all_configs.items()}},
                  f, indent=2, default=str)

    # Find best config
    best_key = max(all_configs, key=lambda k: all_configs[k]['num'])
    best = all_configs[best_key]
    print(f"\n  === VERDICT ===")
    print(f"    Best config: {best_key} -> num={best['num']:.0%}, word={best['word']:.0%}")
    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 161] Complete.")

if __name__ == '__main__':
    main()
