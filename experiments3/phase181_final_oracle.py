# -*- coding: utf-8 -*-
"""
Phase 181: The Final Oracle - Base Output as FGA Target
P180 revealed: Logit Lens accuracy = 0% even on BASE model (0.5B too small).

New approach: Skip Logit Lens entirely.
Use BASE model's FINAL output (top-1 token) as FGA target for SURGERY model.

BASE model = "what does the model believe?"
SURGERY + FGA = "amplify that belief with geometric steering"

Model: Qwen2.5-0.5B (GPU)
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
    ("# The atomic number of carbon is", " 6"),
    ("# A year has", " 365"),
    ("# The number of continents is", " 7"),
    ("# Pi is approximately", " 3"),
]

ARITH_TEST = [
    ("# 1 + 1 =", " 2"), ("# 3 + 4 =", " 7"), ("# 5 + 5 =", " 10"),
    ("# 8 + 1 =", " 9"), ("# 6 + 3 =", " 9"), ("# 4 + 4 =", " 8"),
    ("# 7 + 2 =", " 9"), ("# 2 + 6 =", " 8"),
]

UNKNOWN_TEST = [
    "# The capital of Xylandia is",
    "# The 937th digit of pi is",
    "# The winner of the 2030 World Cup is",
    "# The GDP of Atlantis is",
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


def compute_entropy(logits):
    probs = F.softmax(logits.float(), dim=-1)
    return -(probs * torch.log(probs + 1e-10)).sum().item()


def final_oracle(base_model, surg_model, tok, prompt, fga_gain=5, fga_layer=None):
    """Use BASE top-1 prediction as SURGERY FGA target."""
    n_layers = surg_model.config.num_hidden_layers
    if fga_layer is None:
        fga_layer = n_layers - max(1, n_layers // 4)

    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    # Phase 1: BASE model top-1
    with torch.no_grad():
        base_logits = base_model(**inp).logits[0, -1, :].float()
    base_pred_id = base_logits.argmax().item()
    entropy = compute_entropy(base_logits)

    # Phase 2: SURGERY model + FGA toward base prediction
    unembed = surg_model.lm_head.weight.data[base_pred_id].float()
    direction = unembed / (unembed.norm() + 1e-8)

    def fh(module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += fga_gain * direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        h = output.float()
        if h.dim() == 3: h[:, -1, :] += fga_gain * direction.to(h.device)
        return h.to(output.dtype)

    handle = surg_model.model.layers[fga_layer].register_forward_hook(fh)
    with torch.no_grad():
        final_logits = surg_model(**inp).logits[0, -1, :].float()
    handle.remove()

    final_pred_id = final_logits.argmax().item()
    return final_pred_id, base_pred_id, entropy


def main():
    print("[P181] The Final Oracle")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    surg = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(surg, tok, strength=2.0)

    configs = {}
    # Compare methods
    methods = {
        'A_base_only': ('Base model only, no Surgery, no FGA', False, False, 0),
        'B_surg_fga_teacher': ('Surgery + Teacher FGA g=5', True, True, 5),
        'C_final_oracle_g5': ('Final Oracle (base->surg FGA) g=5', True, False, 5),
        'D_final_oracle_g10': ('Final Oracle g=10', True, False, 10),
    }

    for mname, (desc, use_surg, teacher, fga_gain) in methods.items():
        print(f"\n  === {mname}: {desc} ===")
        fact_correct = 0
        arith_correct = 0
        details_f = []
        details_a = []

        for prompt, expected in FACT_TEST:
            exp_id = tok.encode(expected)[-1]
            if not use_surg:
                # Base model only
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    logits = base(**inp).logits[0, -1, :].float()
                pred_id = logits.argmax().item()
            elif teacher:
                # Teacher FGA (knows ground truth)
                n_layers = surg.config.num_hidden_layers
                fga_layer = n_layers - max(1, n_layers // 4)
                unembed = surg.lm_head.weight.data[exp_id].float()
                d = unembed / (unembed.norm() + 1e-8)
                def mk(dd, gg):
                    def fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0].float()
                            if h.dim() == 3: h[:, -1, :] += gg * dd.to(h.device)
                            return (h.to(output[0].dtype),) + output[1:]
                        return output
                    return fn
                hh = surg.model.layers[fga_layer].register_forward_hook(mk(d, fga_gain))
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    logits = surg(**inp).logits[0, -1, :].float()
                hh.remove()
                pred_id = logits.argmax().item()
            else:
                # Final Oracle
                pred_id, base_pred, _ = final_oracle(base, surg, tok, prompt, fga_gain)

            ok = (pred_id == exp_id) or (tok.decode([pred_id]).strip() == expected.strip())
            if ok: fact_correct += 1
            details_f.append({'expected': expected.strip(),
                              'pred': tok.decode([pred_id]).strip(), 'correct': ok})

        for prompt, expected in ARITH_TEST:
            exp_id = tok.encode(expected)[-1]
            if not use_surg:
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    logits = base(**inp).logits[0, -1, :].float()
                pred_id = logits.argmax().item()
            elif teacher:
                n_layers = surg.config.num_hidden_layers
                fga_layer = n_layers - max(1, n_layers // 4)
                unembed = surg.lm_head.weight.data[exp_id].float()
                d = unembed / (unembed.norm() + 1e-8)
                def mk2(dd, gg):
                    def fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0].float()
                            if h.dim() == 3: h[:, -1, :] += gg * dd.to(h.device)
                            return (h.to(output[0].dtype),) + output[1:]
                        return output
                    return fn
                hh = surg.model.layers[fga_layer].register_forward_hook(mk2(d, fga_gain))
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    logits = surg(**inp).logits[0, -1, :].float()
                hh.remove()
                pred_id = logits.argmax().item()
            else:
                pred_id, base_pred, _ = final_oracle(base, surg, tok, prompt, fga_gain)

            ok = (pred_id == exp_id) or (tok.decode([pred_id]).strip() == expected.strip())
            if ok: arith_correct += 1
            details_a.append({'expected': expected.strip(),
                              'pred': tok.decode([pred_id]).strip(), 'correct': ok})

        fa = fact_correct / len(FACT_TEST)
        aa = arith_correct / len(ARITH_TEST)
        print(f"    Fact: {fa:.0%} ({fact_correct}/{len(FACT_TEST)})")
        print(f"    Arith: {aa:.0%} ({arith_correct}/{len(ARITH_TEST)})")
        configs[mname] = {'fact': fa, 'arith': aa, 'desc': desc}

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase181_final_oracle.json'), 'w') as f:
        json.dump({'phase': '181', 'name': 'The Final Oracle',
                   'configs': configs}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    names = list(configs.keys())
    labels = ['Base\nOnly', 'Surgery+\nTeacher FGA', 'Final Oracle\ng=5',
              'Final Oracle\ng=10']
    fact_vals = [configs[n]['fact'] for n in names]
    arith_vals = [configs[n]['arith'] for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x-w/2, fact_vals, w, label='Factual', color='#e74c3c', alpha=0.8)
    ax.bar(x+w/2, arith_vals, w, label='Arithmetic', color='#3498db', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w/2, fact_vals[i]+0.02, f'{fact_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
        ax.text(x[i]+w/2, arith_vals[i]+0.02, f'{arith_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11); ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase 181: The Final Oracle\n'
                 'BASE top-1 prediction as SURGERY FGA target (no ground truth)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase181_final_oracle.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    fo = configs['C_final_oracle_g5']
    teacher = configs['B_surg_fga_teacher']
    print(f"  -> Teacher FGA:  fact={teacher['fact']:.0%} arith={teacher['arith']:.0%}")
    print(f"  -> Final Oracle: fact={fo['fact']:.0%} arith={fo['arith']:.0%}")
    ratio = (fo['fact'] + fo['arith']) / max(0.01, teacher['fact'] + teacher['arith'])
    print(f"  -> Oracle/Teacher ratio: {ratio:.0%}")
    if ratio >= 0.8:
        print("  -> AUTOPOIESIS ACHIEVED! Oracle matches 80%+ of Teacher!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 181] Complete.")

    del base, surg; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
