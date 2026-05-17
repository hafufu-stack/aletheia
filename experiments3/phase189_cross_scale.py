# -*- coding: utf-8 -*-
"""
Phase 189: Cross-Scale Autopoiesis
Use 1.5B as Oracle BASE (stronger knowledge) + 0.5B Surgery as output.
14B is too large for dual-model setup; 1.5B is practical and tests the
"bigger BASE = better Oracle" hypothesis.

Model: Qwen2.5-1.5B (Oracle) + Qwen2.5-0.5B (Surgery+FGA)
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

FACT_TEST = [
    ("# The capital of Japan is", " Tokyo"),
    ("# The capital of France is", " Paris"),
    ("# The capital of Germany is", " Berlin"),
    ("# The largest planet is", " Jupiter"),
    ("# Water freezes at", " 0"),
    ("# The boiling point of water is", " 100"),
    ("# A year has", " 365"),
    ("# The number of continents is", " 7"),
    ("# Pi is approximately", " 3"),
    ("# The atomic number of carbon is", " 6"),
]
ARITH_TEST = [
    ("# 1 + 1 =", " 2"), ("# 3 + 4 =", " 7"), ("# 5 + 5 =", " 10"),
    ("# 8 + 1 =", " 9"), ("# 6 + 3 =", " 9"), ("# 4 + 4 =", " 8"),
]
NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"]

def apply_surgery(model, tok, strength=2.0):
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)

def cross_scale_oracle(oracle_model, oracle_tok, surg_model, surg_tok,
                       prompt, fga_gain=5, fga_layer=None):
    n_layers = surg_model.config.num_hidden_layers
    if fga_layer is None:
        fga_layer = n_layers - max(1, n_layers // 4)
    # Oracle (1.5B): get prediction
    inp_o = oracle_tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        oracle_logits = oracle_model(**inp_o).logits[0, -1, :].float()
    oracle_pred_id = oracle_logits.argmax().item()
    oracle_text = oracle_tok.decode([oracle_pred_id]).strip()
    # Map oracle token to surgery vocab
    surg_pred_id = surg_tok.encode(f" {oracle_text}")[-1] if oracle_text else oracle_pred_id
    # Surgery (0.5B) + FGA
    unembed = surg_model.lm_head.weight.data[surg_pred_id].float()
    direction = unembed / (unembed.norm() + 1e-8)
    def fh(module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += fga_gain * direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        return output
    handle = surg_model.model.layers[fga_layer].register_forward_hook(fh)
    inp_s = surg_tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        final_logits = surg_model(**inp_s).logits[0, -1, :].float()
    handle.remove()
    return final_logits.argmax().item(), oracle_pred_id, oracle_text

def main():
    print("[P189] Cross-Scale Autopoiesis")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load 1.5B Oracle
    print("  Loading Qwen-1.5B Oracle...")
    oracle_id = 'Qwen/Qwen2.5-1.5B'
    oracle_tok = AutoTokenizer.from_pretrained(oracle_id, local_files_only=True)
    if oracle_tok.pad_token is None: oracle_tok.pad_token = oracle_tok.eos_token
    oracle_model = AutoModelForCausalLM.from_pretrained(
        oracle_id, local_files_only=True, torch_dtype=torch.float16).to(DEVICE)

    # Load 0.5B Surgery
    print("  Loading Qwen-0.5B Surgery...")
    surg_id = 'Qwen/Qwen2.5-0.5B'
    surg_tok = AutoTokenizer.from_pretrained(surg_id, local_files_only=True)
    if surg_tok.pad_token is None: surg_tok.pad_token = surg_tok.eos_token
    surg_model = AutoModelForCausalLM.from_pretrained(
        surg_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(surg_model, surg_tok, strength=2.0)

    # First: 1.5B Oracle accuracy alone
    print("\n  === 1.5B Oracle Raw Accuracy ===")
    oracle_fact_ok = 0
    for prompt, expected in FACT_TEST:
        exp_text = expected.strip()
        inp = oracle_tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = oracle_model(**inp).logits[0, -1, :].float()
        pred = oracle_tok.decode([logits.argmax().item()]).strip()
        ok = pred == exp_text
        if ok: oracle_fact_ok += 1
        print(f"    '{exp_text:>8s}' -> '{pred:>8s}' {'OK' if ok else 'MISS'}")
    oracle_raw = oracle_fact_ok / len(FACT_TEST)
    print(f"  1.5B raw fact accuracy: {oracle_raw:.0%}")

    # 0.5B Oracle accuracy for comparison
    base05 = AutoModelForCausalLM.from_pretrained(
        surg_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    base05_ok = 0
    for prompt, expected in FACT_TEST:
        inp = surg_tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = base05(**inp).logits[0, -1, :].float()
        exp_id = surg_tok.encode(expected)[-1]
        if logits.argmax().item() == exp_id: base05_ok += 1
    base05_raw = base05_ok / len(FACT_TEST)
    del base05; gc.collect(); torch.cuda.empty_cache()
    print(f"  0.5B raw fact accuracy: {base05_raw:.0%}")

    # Cross-scale Oracle
    configs = {}
    for gain in [5, 10]:
        print(f"\n  === Cross-Scale (1.5B Oracle -> 0.5B Surgery) g={gain} ===")
        fc, ac = 0, 0
        for prompt, expected in FACT_TEST:
            exp_id = surg_tok.encode(expected)[-1]
            final_id, oracle_id_val, oracle_text = cross_scale_oracle(
                oracle_model, oracle_tok, surg_model, surg_tok, prompt, fga_gain=gain)
            ok = (final_id == exp_id) or (surg_tok.decode([final_id]).strip() == expected.strip())
            if ok: fc += 1
        for prompt, expected in ARITH_TEST:
            exp_id = surg_tok.encode(expected)[-1]
            final_id, _, _ = cross_scale_oracle(
                oracle_model, oracle_tok, surg_model, surg_tok, prompt, fga_gain=gain)
            ok = (final_id == exp_id) or (surg_tok.decode([final_id]).strip() == expected.strip())
            if ok: ac += 1
        fa = fc / len(FACT_TEST)
        aa = ac / len(ARITH_TEST)
        print(f"  fact={fa:.0%} arith={aa:.0%}")
        configs[f'cross_g{gain}'] = {'fact': fa, 'arith': aa, 'gain': gain}

    configs['oracle_1_5B_raw'] = {'fact': oracle_raw}
    configs['oracle_0_5B_raw'] = {'fact': base05_raw}

    with open(os.path.join(RESULTS_DIR, 'phase189_cross_scale.json'), 'w') as f:
        json.dump({'phase': '189', 'name': 'Cross-Scale Autopoiesis',
                   'configs': configs}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    labels = ['0.5B\nOracle Raw', '1.5B\nOracle Raw', 'Cross-Scale\ng=5', 'Cross-Scale\ng=10']
    vals = [base05_raw, oracle_raw,
            configs['cross_g5']['fact'], configs['cross_g10']['fact']]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    ax.bar(labels, vals, color=colors, alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fact Accuracy', fontsize=12)
    ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase 189: Cross-Scale Autopoiesis\n'
                 '1.5B Oracle + 0.5B Surgery+FGA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase189_cross_scale.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    print(f"  -> 0.5B Oracle: {base05_raw:.0%}")
    print(f"  -> 1.5B Oracle: {oracle_raw:.0%}")
    if oracle_raw > base05_raw + 0.1:
        print("  -> BIGGER BASE = BETTER ORACLE confirmed!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 189] Complete.")
    del oracle_model, surg_model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
