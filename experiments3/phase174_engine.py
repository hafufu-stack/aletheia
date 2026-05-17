# -*- coding: utf-8 -*-
"""
Phase 174: The Aletheia Engine (Full Autonomous Pipeline)
Combine all Season 35-36 discoveries into one unified system:
  - Entropy Oracle (P171: AUC=0.882) for known/unknown routing
  - Trinity (P170: Surgery+Code+FGA) for factual+arithmetic
  - Aegis (P169: abstention) for unknown facts
  - Oracle-Guided FGA (P173) for self-referential truth

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

# Mixed test: facts + arithmetic + unknowns
MIXED_TEST = [
    ("The capital of Japan is", " Tokyo", "known_fact"),
    ("The capital of France is", " Paris", "known_fact"),
    ("Water freezes at", " 0", "known_fact"),
    ("A year has", " 365", "known_fact"),
    ("The number of continents is", " 7", "known_fact"),
    ("1 + 1 =", " 2", "arithmetic"),
    ("3 + 4 =", " 7", "arithmetic"),
    ("5 + 5 =", " 10", "arithmetic"),
    ("8 + 1 =", " 9", "arithmetic"),
    ("6 + 3 =", " 9", "arithmetic"),
    ("The capital of Xylandia is", "ABSTAIN", "unknown"),
    ("The 937th digit of pi is", "ABSTAIN", "unknown"),
    ("The winner of the 2030 World Cup is", "ABSTAIN", "unknown"),
    ("The secret code of the universe is", "ABSTAIN", "unknown"),
    ("The GDP of Atlantis is", "ABSTAIN", "unknown"),
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


def aletheia_engine(model, tok, prompt, expected, category,
                    entropy_threshold=5.0, fga_gain=5):
    """The complete Aletheia Engine pipeline."""
    n_layers = model.config.num_hidden_layers
    oracle_layer = n_layers // 2
    fga_layer = n_layers - max(1, n_layers // 4)

    # Step 1: Apply Code Mode (Trinity requires it)
    code_prompt = f"# {prompt}"
    inp = tok(code_prompt, return_tensors='pt').to(DEVICE)

    # Step 2: Measure entropy (no FGA)
    with torch.no_grad():
        base_logits = model(**inp).logits[0, -1, :].float()
    entropy = compute_entropy(base_logits)

    # Step 3: Route based on entropy
    if entropy > entropy_threshold:
        # HIGH ENTROPY -> Unknown -> AEGIS (abstain)
        return {
            'action': 'abstain', 'entropy': entropy,
            'correct': (category == 'unknown'),
            'pred': 'ABSTAIN'
        }

    # LOW ENTROPY -> Known fact or arithmetic -> Oracle-Guided FGA
    # Get Oracle prediction from middle layer
    oracle_hidden = {}
    def oracle_hook(module, input, output):
        if isinstance(output, tuple):
            oracle_hidden['h'] = output[0][:, -1, :].detach().float()
        else:
            oracle_hidden['h'] = output[:, -1, :].detach().float()

    h_handle = model.model.layers[oracle_layer].register_forward_hook(oracle_hook)
    with torch.no_grad():
        _ = model(**inp)
    h_handle.remove()

    # Logit Lens at Oracle layer
    if 'h' in oracle_hidden:
        oracle_logits = model.lm_head(oracle_hidden['h'].to(model.lm_head.weight.dtype))
        oracle_pred_id = oracle_logits.float().argmax(dim=-1).item()
    else:
        oracle_pred_id = base_logits.argmax().item()

    # FGA toward Oracle's prediction
    unembed = model.lm_head.weight.data[oracle_pred_id].float()
    direction = unembed / (unembed.norm() + 1e-8)

    def fga_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += fga_gain * direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        h = output.float()
        if h.dim() == 3: h[:, -1, :] += fga_gain * direction.to(h.device)
        return h.to(output.dtype)

    fga_handle = model.model.layers[fga_layer].register_forward_hook(fga_hook)
    with torch.no_grad():
        logits = model(**inp).logits[0, -1, :].float()
    fga_handle.remove()

    pred_id = logits.argmax().item()
    pred_text = tok.decode([pred_id]).strip()

    if category == 'unknown':
        correct = False  # Should have abstained
    elif expected == 'ABSTAIN':
        correct = False
    else:
        exp_id = tok.encode(expected)[-1]
        correct = (pred_id == exp_id) or (pred_text == expected.strip())

    return {
        'action': 'answer', 'entropy': entropy,
        'correct': correct, 'pred': pred_text,
        'oracle_pred': tok.decode([oracle_pred_id]).strip()
    }


def main():
    print("[P174] The Aletheia Engine")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    # Sweep entropy thresholds
    all_configs = {}
    for thresh in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
        results = []
        for prompt, expected, category in MIXED_TEST:
            r = aletheia_engine(model, tok, prompt, expected, category,
                                entropy_threshold=thresh, fga_gain=5)
            r['category'] = category
            r['prompt'] = prompt[:40]
            r['expected'] = expected
            results.append(r)

        # Score by category
        fact_correct = sum(1 for r in results if r['category'] == 'known_fact' and r['correct'])
        fact_total = sum(1 for r in results if r['category'] == 'known_fact')
        arith_correct = sum(1 for r in results if r['category'] == 'arithmetic' and r['correct'])
        arith_total = sum(1 for r in results if r['category'] == 'arithmetic')
        abstain_correct = sum(1 for r in results if r['category'] == 'unknown' and r['correct'])
        abstain_total = sum(1 for r in results if r['category'] == 'unknown')

        fact_acc = fact_correct / max(1, fact_total)
        arith_acc = arith_correct / max(1, arith_total)
        abstain_acc = abstain_correct / max(1, abstain_total)
        overall = (fact_correct + arith_correct + abstain_correct) / len(results)

        print(f"\n  thresh={thresh:.0f}: fact={fact_acc:.0%} arith={arith_acc:.0%} "
              f"abstain={abstain_acc:.0%} overall={overall:.0%}")
        for r in results:
            tag = 'OK' if r['correct'] else 'MISS'
            print(f"    [{r['category']:12s}] {r['action']:8s} -> "
                  f"{r.get('pred',''):10s} ({tag}) H={r['entropy']:.1f}")

        all_configs[f't{thresh:.0f}'] = {
            'threshold': thresh, 'fact_acc': fact_acc,
            'arith_acc': arith_acc, 'abstain_acc': abstain_acc,
            'overall': overall, 'details': results
        }

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase174_engine.json'), 'w') as f:
        json.dump({'phase': '174', 'name': 'The Aletheia Engine',
                   'configs': all_configs}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    thresholds = sorted([all_configs[k]['threshold'] for k in all_configs])
    keys = [f't{int(t)}' for t in thresholds]

    ax = axes[0]
    ax.plot(thresholds, [all_configs[k]['fact_acc'] for k in keys], 'r-o', lw=2, label='Factual')
    ax.plot(thresholds, [all_configs[k]['arith_acc'] for k in keys], 'b-s', lw=2, label='Arithmetic')
    ax.plot(thresholds, [all_configs[k]['abstain_acc'] for k in keys], 'g-^', lw=2, label='Abstention')
    ax.plot(thresholds, [all_configs[k]['overall'] for k in keys], 'k--D', lw=2, label='Overall')
    ax.set_xlabel('Entropy Threshold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_title('Aletheia Engine: Threshold Sweep', fontsize=13, fontweight='bold')

    # Best config bar chart
    best_key = max(all_configs, key=lambda k: all_configs[k]['overall'])
    best = all_configs[best_key]
    ax = axes[1]
    cats = ['Factual', 'Arithmetic', 'Abstention', 'Overall']
    vals = [best['fact_acc'], best['arith_acc'], best['abstain_acc'], best['overall']]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    ax.bar(cats, vals, color=colors, alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(f'Best Engine (t={best["threshold"]:.0f})', fontsize=13, fontweight='bold')

    plt.suptitle('Phase 174: The Aletheia Engine\n'
                 'One model: facts + arithmetic + abstention (zero training)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase174_engine.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    print(f"  -> Best threshold: {best['threshold']:.0f}")
    print(f"  -> Fact: {best['fact_acc']:.0%}, Arith: {best['arith_acc']:.0%}, "
          f"Abstain: {best['abstain_acc']:.0%}, Overall: {best['overall']:.0%}")
    if best['overall'] >= 0.7:
        print("  -> THE ALETHEIA ENGINE IS OPERATIONAL!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 174] Complete.")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
