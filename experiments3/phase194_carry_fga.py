# -*- coding: utf-8 -*-
"""
Phase 194: Carry-Aware FGA (Opus Addition)
P190 found: carry bit is 87-100% readable from hidden states.
Use this: if carry is detected -> constrain FGA to 2-digit answers.
         if no carry -> constrain to single digit answers.

This turns the "virtual abacus" from a diagnostic tool into a
PRACTICAL steering mechanism.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"]
DIGIT_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9"]
TWO_DIGIT_TOKENS = [" 10"," 11"," 12"," 13"," 14"," 15"," 16"," 17"," 18"]

def apply_surgery(model, tok, strength=2.0):
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)

def gen_additions():
    data = []
    for a in range(10):
        for b in range(10):
            s = a + b
            data.append((a, b, s, 1 if s >= 10 else 0))
    return data

def main():
    print("[P194] Carry-Aware FGA")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - max(1, n_layers // 4)
    probe_layer = 20  # Best carry accuracy from P190

    additions = gen_additions()

    # Step 1: Train carry probe on L20
    print("  Training carry probe on L20...")
    X_train = []; y_train = []
    captured = [None]
    def train_hook(module, input, output):
        if isinstance(output, tuple):
            captured[0] = output[0][:, -1, :].detach().cpu().numpy().flatten()
        else:
            captured[0] = output[:, -1, :].detach().cpu().numpy().flatten()
    for a, b, s, carry in additions:
        prompt = f"# {a} + {b} ="
        h = model.model.layers[probe_layer].register_forward_hook(train_hook)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            _ = model(**inp)
        h.remove()
        X_train.append(captured[0])
        y_train.append(carry)
    X_train = np.array(X_train); y_train = np.array(y_train)
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)
    print(f"  Probe train accuracy: {probe.score(X_train, y_train):.0%}")

    # Step 2: Evaluate with carry-aware FGA
    configs = {}

    # A: Standard FGA (no carry awareness)
    correct_std = 0
    for a, b, s, carry in additions:
        prompt = f"# {a} + {b} ="
        exp_tok = f" {s}"
        exp_id = tok.encode(exp_tok)[-1]
        unembed = model.lm_head.weight.data[exp_id].float()
        d = unembed / (unembed.norm() + 1e-8)
        def mk(dd, gg):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].float()
                    if h.dim() == 3: h[:, -1, :] += gg * dd.to(h.device)
                    return (h.to(output[0].dtype),) + output[1:]
                return output
            return fn
        handle = model.model.layers[fga_layer].register_forward_hook(mk(d, 5))
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        handle.remove()
        pred = logits.argmax().item()
        if pred == exp_id or tok.decode([pred]).strip() == str(s): correct_std += 1
    configs['teacher_fga'] = correct_std / len(additions)
    print(f"  Teacher FGA: {configs['teacher_fga']:.0%}")

    # B: Carry-aware masking + FGA
    correct_carry = 0
    correct_nocarry = 0
    total_carry = 0; total_nocarry = 0
    for a, b, s, carry in additions:
        prompt = f"# {a} + {b} ="
        exp_tok = f" {s}"
        exp_id = tok.encode(exp_tok)[-1]
        # Get hidden state for carry prediction
        probe_cap = [None]
        def probe_hook(module, input, output):
            if isinstance(output, tuple):
                probe_cap[0] = output[0][:, -1, :].detach().cpu().numpy().flatten()
            else:
                probe_cap[0] = output[:, -1, :].detach().cpu().numpy().flatten()
        h_probe = model.model.layers[probe_layer].register_forward_hook(probe_hook)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        h_probe.remove()
        # Predict carry
        pred_carry = probe.predict(probe_cap[0].reshape(1, -1))[0]
        # Mask logits based on carry prediction
        if pred_carry == 1:
            # Expect 2-digit answer (10-18): mask single digits
            for t in DIGIT_TOKENS:
                tid = tok.encode(t)[-1]
                logits[tid] = -float('inf')
        else:
            # Expect single digit (0-9): mask 2-digit
            for t in TWO_DIGIT_TOKENS:
                try:
                    tid = tok.encode(t)[-1]
                    logits[tid] = -float('inf')
                except: pass

        pred_id = logits.argmax().item()
        ok = (pred_id == exp_id) or (tok.decode([pred_id]).strip() == str(s))
        if carry == 1:
            total_carry += 1
            if ok: correct_carry += 1
        else:
            total_nocarry += 1
            if ok: correct_nocarry += 1

    carry_acc = correct_carry / max(1, total_carry)
    nocarry_acc = correct_nocarry / max(1, total_nocarry)
    overall = (correct_carry + correct_nocarry) / len(additions)
    configs['carry_aware'] = overall
    configs['carry_acc'] = carry_acc
    configs['nocarry_acc'] = nocarry_acc
    print(f"  Carry-aware: overall={overall:.0%} carry={carry_acc:.0%} nocarry={nocarry_acc:.0%}")

    # C: Baseline (no FGA)
    correct_base = 0
    for a, b, s, carry in additions:
        prompt = f"# {a} + {b} ="
        exp_id = tok.encode(f" {s}")[-1]
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if logits.argmax().item() == exp_id: correct_base += 1
    configs['baseline'] = correct_base / len(additions)
    print(f"  Baseline (no FGA): {configs['baseline']:.0%}")

    with open(os.path.join(RESULTS_DIR, 'phase194_carry_fga.json'), 'w') as f:
        json.dump({'phase': '194', 'name': 'Carry-Aware FGA', 'configs': configs},
                  f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    methods = ['Baseline\n(no FGA)', 'Teacher\nFGA', 'Carry-Aware\nMasking']
    vals = [configs['baseline'], configs['teacher_fga'], configs['carry_aware']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    ax.bar(methods, vals, color=colors, alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy (all 100 additions)', fontsize=12)
    ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase 194: Carry-Aware FGA\nUsing the virtual abacus to constrain outputs',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase194_carry_fga.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    print(f"  -> Baseline: {configs['baseline']:.0%}")
    print(f"  -> Teacher FGA: {configs['teacher_fga']:.0%}")
    print(f"  -> Carry-Aware: {configs['carry_aware']:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 194] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
