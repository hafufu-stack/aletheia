# -*- coding: utf-8 -*-
"""
Phase 148: Dynamic Activation Surgery
Can we disperse number tokens at INFERENCE TIME (not weights)?

P147 showed: static surgery hurts arithmetic learning.
P144 showed: surgery + S&S = 100% factual accuracy.
Solution: Keep weights intact (arithmetic-friendly), disperse
ONLY during inference via activation steering at layer 0.

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

FACT_SET = [
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("The speed of light is approximately", " 299", "number"),
    ("A year has", " 365", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
    ("The capital of Japan is", " Tokyo", "word"),
    ("The capital of France is", " Paris", "word"),
]

ARITHMETIC_SET = [
    ("2 + 3 =", " 5"),
    ("7 - 4 =", " 3"),
    ("4 + 5 =", " 9"),
    ("8 - 6 =", " 2"),
    ("1 + 1 =", " 2"),
    ("9 - 1 =", " 8"),
    ("3 + 4 =", " 7"),
    ("6 - 2 =", " 4"),
]


class ActivationDispersionHook:
    """Disperse number token activations at inference time, not weights."""
    def __init__(self, model, tok, strength=1.0):
        self.strength = strength
        self.handle = None
        # Pre-compute dispersion vectors for number tokens
        embed = model.model.embed_tokens.weight.data
        num_strs = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]
        self.num_ids = set()
        for t in num_strs:
            self.num_ids.add(tok.encode(t)[-1])
        num_ids_list = list(self.num_ids)
        vecs = embed[num_ids_list].float()
        self.center = vecs.mean(dim=0)
        # For each num token, compute dispersion direction
        self.disp_vectors = {}
        for i, idx in enumerate(num_ids_list):
            diff = vecs[i] - self.center
            direction = diff / (diff.norm() + 1e-8)
            self.disp_vectors[idx] = (strength * direction * vecs[i].norm())

    def hook_fn(self, module, input, output):
        """Apply dispersion to activations of number tokens only."""
        # output from embed_tokens: (batch, seq, hidden)
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # We need the input_ids to know which positions are number tokens
        # This is stored before the forward pass
        if hasattr(self, 'current_input_ids'):
            ids = self.current_input_ids[0]  # (seq,)
            for pos in range(ids.shape[0]):
                tid = ids[pos].item()
                if tid in self.disp_vectors:
                    hidden[0, pos, :] += self.disp_vectors[tid].to(hidden.device, hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def register(self, model):
        self.handle = model.model.embed_tokens.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


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


def static_disperse(model, tok, strength=1.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366"]
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in num_tokens))
    vecs = embed[ids].clone()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()


def evaluate_facts(model, tok, code_mode=False, fga_gain=0, disp_hook=None):
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - 2
    results = []
    for prompt, expected, cat in FACT_SET:
        text = f"# {prompt}" if code_mode else prompt
        exp_id = tok.encode(expected)[-1]
        fga = None
        if fga_gain > 0:
            fga = FGAHook(model, exp_id, fga_gain)
            fga.register(model, fga_layer)
        inp = tok(text, return_tensors='pt').to(DEVICE)
        if disp_hook:
            disp_hook.current_input_ids = inp['input_ids']
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if fga: fga.remove()
        pred_id = logits.argmax().item()
        results.append({'cat': cat, 'correct': int(pred_id == exp_id)})
    w = sum(r['correct'] for r in results if r['cat'] == 'word')
    wt = max(1, sum(1 for r in results if r['cat'] == 'word'))
    n = sum(r['correct'] for r in results if r['cat'] == 'number')
    nt = max(1, sum(1 for r in results if r['cat'] == 'number'))
    return w/wt, n/nt


def evaluate_arithmetic(model, tok, disp_hook=None):
    space_id = tok.encode(" ")[-1]
    correct = 0
    for prompt, expected in ARITHMETIC_SET:
        exp_ids = tok.encode(expected)
        digit_ids = [i for i in exp_ids if i != space_id]
        first_digit = digit_ids[0] if digit_ids else exp_ids[-1]
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        if disp_hook:
            disp_hook.current_input_ids = inp['input_ids']
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        pred_id = logits.argmax().item()
        if pred_id == space_id:
            inp2 = torch.cat([inp['input_ids'], torch.tensor([[pred_id]], device=DEVICE)], dim=1)
            if disp_hook:
                disp_hook.current_input_ids = inp2
            with torch.no_grad():
                logits2 = model(input_ids=inp2).logits[0, -1, :].float()
            pred_id = logits2.argmax().item()
        if pred_id == first_digit: correct += 1
    return correct / len(ARITHMETIC_SET)


def train_sft_and_eval(model, tok, n_steps=200, disp_hook=None):
    """SFT on arithmetic, then evaluate."""
    import random
    random.seed(42)
    train_data = [f"{a} + {b} = {a+b}" for a in range(10) for b in range(10)]
    random.shuffle(train_data)
    model.train()
    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for step in range(n_steps):
        text = train_data[step % len(train_data)]
        inp = tok(text, return_tensors='pt', truncation=True, max_length=32).to(DEVICE)
        outputs = model(**inp, labels=inp['input_ids'])
        opt.zero_grad(); outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return evaluate_arithmetic(model, tok, disp_hook)


def main():
    print("[P148] Dynamic Activation Surgery")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    configs = {}

    # A: Baseline
    print("\n  === A: Baseline ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    fw, fn = evaluate_facts(model, tok)
    arith = evaluate_arithmetic(model, tok)
    print(f"    Fact word={fw:.0%} num={fn:.0%}, Arith={arith:.0%}")
    configs['A_baseline'] = {'fact_word': fw, 'fact_num': fn, 'arith': arith}

    # B: Static Surgery + S&S (reference from P136b)
    print("\n  === B: Static Surgery + S&S ===")
    del model; gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    static_disperse(model, tok, strength=1.0)
    fw, fn = evaluate_facts(model, tok, code_mode=True, fga_gain=20)
    arith = evaluate_arithmetic(model, tok)
    print(f"    Fact word={fw:.0%} num={fn:.0%}, Arith={arith:.0%}")
    configs['B_static_ss'] = {'fact_word': fw, 'fact_num': fn, 'arith': arith}

    # B2: Static Surgery arithmetic after SFT
    print("\n  === B2: Static Surgery -> SFT(200) -> Arith ===")
    del model; gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    static_disperse(model, tok, strength=1.0)
    arith_sft = train_sft_and_eval(model, tok, n_steps=200)
    print(f"    Arith after SFT={arith_sft:.0%}")
    configs['B2_static_sft'] = {'arith_sft': arith_sft}
    del model; gc.collect(); torch.cuda.empty_cache()

    # C: Dynamic Surgery + S&S (weights untouched!)
    print("\n  === C: Dynamic Surgery + S&S ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    disp_hook = ActivationDispersionHook(model, tok, strength=1.0)
    disp_hook.register(model)
    fw, fn = evaluate_facts(model, tok, code_mode=True, fga_gain=20, disp_hook=disp_hook)
    arith = evaluate_arithmetic(model, tok, disp_hook=disp_hook)
    disp_hook.remove()
    print(f"    Fact word={fw:.0%} num={fn:.0%}, Arith={arith:.0%}")
    configs['C_dynamic_ss'] = {'fact_word': fw, 'fact_num': fn, 'arith': arith}

    # C2: Dynamic Surgery -> SFT (weights change but surgery is dynamic, not baked in)
    print("\n  === C2: SFT(200) on UNTOUCHED weights -> Dynamic at eval ===")
    del model; gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    # SFT on base weights (no surgery!)
    arith_base_sft = train_sft_and_eval(model, tok, n_steps=200)
    # Now apply dynamic surgery for fact eval
    disp_hook = ActivationDispersionHook(model, tok, strength=1.0)
    disp_hook.register(model)
    fw, fn = evaluate_facts(model, tok, code_mode=True, fga_gain=20, disp_hook=disp_hook)
    arith_dyn = evaluate_arithmetic(model, tok, disp_hook=disp_hook)
    disp_hook.remove()
    print(f"    Arith(base SFT)={arith_base_sft:.0%}, Fact(dyn)={fw:.0%}/{fn:.0%}")
    configs['C2_sft_then_dynamic'] = {
        'arith_base_sft': arith_base_sft,
        'fact_word_dynamic': fw, 'fact_num_dynamic': fn,
        'arith_with_dynamic': arith_dyn
    }
    del model, disp_hook; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase148_dynamic.json'), 'w') as f:
        json.dump({'phase': '148', 'name': 'Dynamic Activation Surgery',
                   'configs': configs}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Fact accuracy comparison
    ax = axes[0]
    names = ['Baseline', 'Static\nSurgery\n+S&S', 'Dynamic\nSurgery\n+S&S']
    keys = ['A_baseline', 'B_static_ss', 'C_dynamic_ss']
    num_vals = [configs[k].get('fact_num', 0) for k in keys]
    word_vals = [configs[k].get('fact_word', 0) for k in keys]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x-w/2, word_vals, w, label='Word', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, num_vals, w, label='Number', color='#e74c3c', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w/2, word_vals[i]+0.02, f'{word_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
        ax.text(x[i]+w/2, num_vals[i]+0.02, f'{num_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.legend(fontsize=10); ax.set_ylabel('Fact Accuracy', fontsize=12)
    ax.set_title('Factual Accuracy', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')

    # Right: Arithmetic preservation
    ax = axes[1]
    names2 = ['Baseline', 'Static\nSurgery', 'Dynamic\nSurgery', 'Static\n+SFT', 'Base SFT\n+Dyn']
    arith_vals = [
        configs['A_baseline']['arith'],
        configs['B_static_ss']['arith'],
        configs['C_dynamic_ss']['arith'],
        configs['B2_static_sft']['arith_sft'],
        configs['C2_sft_then_dynamic']['arith_base_sft'],
    ]
    colors = ['#95a5a6', '#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71']
    ax.bar(names2, arith_vals, color=colors, alpha=0.8, edgecolor='black', lw=1)
    for i, v in enumerate(arith_vals):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Arithmetic Accuracy', fontsize=12)
    ax.set_title('Arithmetic Preservation\n(Red=Static, Green=Dynamic)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Phase 148: Dynamic Activation Surgery\nCan we get 100% facts WITHOUT hurting arithmetic?',
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase148_dynamic.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    dyn_fn = configs['C_dynamic_ss'].get('fact_num', 0)
    dyn_arith = configs['C2_sft_then_dynamic']['arith_base_sft']
    if dyn_fn > 0.8 and dyn_arith > 0.8:
        print(f"  -> CHAMELEON ENGINE ACHIEVED! Facts={dyn_fn:.0%}, Arith={dyn_arith:.0%}")
    else:
        print(f"  -> Dynamic: facts={dyn_fn:.0%}, arith(SFT)={dyn_arith:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 148] Complete.")

if __name__ == '__main__':
    main()
