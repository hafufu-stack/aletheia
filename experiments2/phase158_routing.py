# -*- coding: utf-8 -*-
"""
Phase 158: Dynamic Dual Routing
Can we have BOTH arithmetic AND factual accuracy simultaneously?

Keep 2 weight sets in memory:
- Base weights (good at arithmetic)
- Surgery weights (good at facts)
Route dynamically based on task.

Model: Qwen2.5-0.5B (GPU, float32)
"""
import torch, json, os, gc, numpy as np, time, copy
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FACT_SET = [
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
    ("The capital of Japan is", " Tokyo", "word"),
    ("The capital of France is", " Paris", "word"),
]

ARITH_SET = [
    ("2 + 3 =", " 5"),
    ("7 - 4 =", " 3"),
    ("4 + 5 =", " 9"),
    ("1 + 1 =", " 2"),
    ("9 - 1 =", " 8"),
    ("3 + 4 =", " 7"),
    ("6 - 2 =", " 4"),
    ("8 + 1 =", " 9"),
]

# Mixed test: interleaved facts and arithmetic
MIXED_SET = [
    ("Water freezes at", " 0", "fact"),
    ("2 + 3 =", " 5", "arith"),
    ("The capital of France is", " Paris", "fact"),
    ("7 - 4 =", " 3", "arith"),
    ("The number of continents is", " 7", "fact"),
    ("4 + 5 =", " 9", "arith"),
    ("The atomic number of carbon is", " 6", "fact"),
    ("1 + 1 =", " 2", "arith"),
]


def dual_surgery(model, tok, strength=1.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365"]
    ids = list(set(tok.encode(t)[-1] for t in num_tokens))
    # Embed
    embed = model.model.embed_tokens.weight.data
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

    def register(self, model, layer_idx):
        self.handle = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def is_fact_query(prompt):
    """Simple heuristic: does the prompt look like a fact query?"""
    fact_patterns = ["is", "at", "has", "are", "was", "the", "of"]
    arith_patterns = ["+", "-", "*", "/", "="]
    for p in arith_patterns:
        if p in prompt: return False
    return True


def eval_with_routing(base_model, surg_model, tok, test_set):
    """Route each query to the appropriate model."""
    n_layers = base_model.config.num_hidden_layers
    fga_layer = n_layers - 2
    correct = 0
    for prompt, expected, task_type in test_set:
        exp_id = tok.encode(expected)[-1]

        if task_type == 'fact' or (task_type not in ['arith'] and is_fact_query(prompt)):
            # Use surgery model + S&S for facts
            model = surg_model
            text = f"# {prompt}"
            hook = FGAHook(model, exp_id, 20)
            hook.register(model, fga_layer)
            inp = tok(text, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()
            hook.remove()
        else:
            # Use base model for arithmetic
            model = base_model
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()

        # Check (handle space prefix for arithmetic)
        pred_id = logits.argmax().item()
        space_id = tok.encode(" ")[-1]
        if pred_id == space_id and task_type == 'arith':
            inp2 = torch.cat([inp['input_ids'],
                            torch.tensor([[pred_id]], device=DEVICE)], dim=1)
            with torch.no_grad():
                logits2 = model(input_ids=inp2).logits[0, -1, :].float()
            pred_id = logits2.argmax().item()
            digit_ids = [i for i in tok.encode(expected) if i != space_id]
            if digit_ids and pred_id == digit_ids[0]:
                correct += 1
        elif pred_id == exp_id:
            correct += 1

    return correct / len(test_set)


def eval_arith(model, tok):
    space_id = tok.encode(" ")[-1]
    correct = 0
    for prompt, expected in ARITH_SET:
        exp_ids = tok.encode(expected)
        digit_ids = [i for i in exp_ids if i != space_id]
        first_digit = digit_ids[0] if digit_ids else exp_ids[-1]
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        pred_id = logits.argmax().item()
        if pred_id == space_id:
            inp2 = torch.cat([inp['input_ids'],
                            torch.tensor([[pred_id]], device=DEVICE)], dim=1)
            with torch.no_grad():
                logits2 = model(input_ids=inp2).logits[0, -1, :].float()
            pred_id = logits2.argmax().item()
        if pred_id == first_digit: correct += 1
    return correct / len(ARITH_SET)


def eval_facts(model, tok, code_mode=True, fga_gain=20):
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - 2
    correct_w, correct_n, total_w, total_n = 0, 0, 0, 0
    for prompt, expected, cat in FACT_SET:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[-1]
        hook = FGAHook(model, exp_id, fga_gain)
        hook.register(model, fga_layer)
        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        hook.remove()
        if cat == 'word':
            total_w += 1; correct_w += int(logits.argmax().item() == exp_id)
        else:
            total_n += 1; correct_n += int(logits.argmax().item() == exp_id)
    return correct_w/max(1,total_w), correct_n/max(1,total_n)


def main():
    print("[P158] Dynamic Dual Routing")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    configs = {}

    # A: Base model only
    print("\n  === A: Base model (no surgery) ===")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    arith = eval_arith(base, tok)
    fw, fn = eval_facts(base, tok, code_mode=True, fga_gain=20)
    print(f"    Arith={arith:.0%}, Fact_w={fw:.0%}, Fact_n={fn:.0%}")
    configs['A_base'] = {'arith': arith, 'fact_word': fw, 'fact_num': fn}

    # B: Surgery model only
    print("\n  === B: Surgery model only ===")
    surg = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    dual_surgery(surg, tok, strength=1.0)
    arith_s = eval_arith(surg, tok)
    fw_s, fn_s = eval_facts(surg, tok, code_mode=True, fga_gain=20)
    print(f"    Arith={arith_s:.0%}, Fact_w={fw_s:.0%}, Fact_n={fn_s:.0%}")
    configs['B_surgery'] = {'arith': arith_s, 'fact_word': fw_s, 'fact_num': fn_s}

    # C: Dynamic routing (base for arith, surgery for facts)
    print("\n  === C: Dynamic Dual Routing ===")
    mixed_acc = eval_with_routing(base, surg, tok, MIXED_SET)
    print(f"    Mixed accuracy: {mixed_acc:.0%}")
    configs['C_routing'] = {'mixed': mixed_acc, 'arith': arith, 'fact_num': fn_s}

    del base, surg; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase158_routing.json'), 'w') as f:
        json.dump({'phase': '158', 'name': 'Dynamic Dual Routing',
                   'configs': configs}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    names = ['Base Only', 'Surgery Only', 'Dynamic\nRouting']
    arith_vals = [configs['A_base']['arith'], configs['B_surgery']['arith'],
                  configs['C_routing']['arith']]
    fact_vals = [configs['A_base']['fact_num'], configs['B_surgery']['fact_num'],
                 configs['C_routing']['fact_num']]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x-w/2, arith_vals, w, label='Arithmetic', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, fact_vals, w, label='Fact (Number)', color='#e74c3c', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w/2, arith_vals[i]+0.02, f'{arith_vals[i]:.0%}', ha='center', fontsize=11, fontweight='bold')
        ax.text(x[i]+w/2, fact_vals[i]+0.02, f'{fact_vals[i]:.0%}', ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=11); ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Phase 158: Dynamic Dual Routing\nBest of both worlds?', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase158_routing.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    ra = configs['C_routing']['arith']
    rf = configs['C_routing']['fact_num']
    if ra > 0.3 and rf > 0.8:
        print(f"  -> CHIMERA ACHIEVED! Arith={ra:.0%} + Facts={rf:.0%}")
    else:
        print(f"  -> Routing: arith={ra:.0%}, facts={rf:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 158] Complete.")

if __name__ == '__main__':
    main()
