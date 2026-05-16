# -*- coding: utf-8 -*-
"""
Phase 140: The Arithmetic Uncertainty Principle
Does Embedding Surgery break math ability?

Hypothesis: Number embeddings are clustered (cos=0.73) because LLMs
NEED them close together for arithmetic operations. Dispersing them
cures factual hallucination but may destroy computation.

Test: Run simple arithmetic on baseline vs Surgery model.
If surgery breaks math -> "Uncertainty Principle" confirmed:
  You cannot simultaneously have factual accuracy AND arithmetic ability
  in a single embedding space.

Model: Qwen2.5-0.5B (GPU, float32)
"""
import torch, json, os, gc, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Simple arithmetic tasks (single-token answers)
ARITHMETIC_TASKS = [
    ("2 + 3 =", " 5"),
    ("7 - 4 =", " 3"),
    ("4 + 5 =", " 9"),
    ("8 - 6 =", " 2"),
    ("1 + 1 =", " 2"),
    ("9 - 1 =", " 8"),
    ("3 + 4 =", " 7"),
    ("6 - 2 =", " 4"),
    ("5 + 1 =", " 6"),
    ("7 - 7 =", " 0"),
]

# Code mode versions
ARITHMETIC_CODE = [
    ("# 2 + 3 =", " 5"),
    ("# 7 - 4 =", " 3"),
    ("# 4 + 5 =", " 9"),
    ("# 8 - 6 =", " 2"),
    ("# 1 + 1 =", " 2"),
    ("# 9 - 1 =", " 8"),
    ("# 3 + 4 =", " 7"),
    ("# 6 - 2 =", " 4"),
    ("# 5 + 1 =", " 6"),
    ("# 7 - 7 =", " 0"),
]

# Factual tasks (same as P136b)
FACT_TASKS = [
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("A year has", " 365", "number"),
    ("The number of continents is", " 7", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The capital of Japan is", " Tokyo", "word"),
    ("The capital of France is", " Paris", "word"),
]


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


def evaluate_arithmetic(model, tok, tasks):
    correct = 0
    for prompt, expected in tasks:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        exp_id = tok.encode(expected)[-1]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        pred_id = logits.argmax().item()
        if pred_id == exp_id:
            correct += 1
    return correct / len(tasks)


def evaluate_facts(model, tok, code_mode=False):
    word_correct, word_total = 0, 0
    num_correct, num_total = 0, 0
    for item in FACT_TASKS:
        prompt, expected, cat = item
        if code_mode:
            prompt = f"# {prompt}"
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        exp_id = tok.encode(expected)[-1]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        pred_id = logits.argmax().item()
        if cat == 'word':
            word_total += 1
            if pred_id == exp_id: word_correct += 1
        else:
            num_total += 1
            if pred_id == exp_id: num_correct += 1
    return word_correct / max(1, word_total), num_correct / max(1, num_total)


def main():
    print("[P140] The Arithmetic Uncertainty Principle")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    configs = {}

    # A: Baseline (no surgery)
    print("\n  === A: Baseline (no surgery) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    arith = evaluate_arithmetic(model, tok, ARITHMETIC_TASKS)
    arith_code = evaluate_arithmetic(model, tok, ARITHMETIC_CODE)
    fact_w, fact_n = evaluate_facts(model, tok)
    print(f"    Arithmetic: {arith:.0%}, Arithmetic(code): {arith_code:.0%}")
    print(f"    Facts word: {fact_w:.0%}, Facts num: {fact_n:.0%}")
    configs['A_baseline'] = {
        'arithmetic': arith, 'arithmetic_code': arith_code,
        'fact_word': fact_w, 'fact_num': fact_n
    }
    del model; gc.collect(); torch.cuda.empty_cache()

    # B: After Dispersion Surgery (strength=1.0)
    print("\n  === B: After Surgery (s=1.0) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    disperse_embeddings(model, tok, strength=1.0)
    arith = evaluate_arithmetic(model, tok, ARITHMETIC_TASKS)
    arith_code = evaluate_arithmetic(model, tok, ARITHMETIC_CODE)
    fact_w, fact_n = evaluate_facts(model, tok)
    print(f"    Arithmetic: {arith:.0%}, Arithmetic(code): {arith_code:.0%}")
    print(f"    Facts word: {fact_w:.0%}, Facts num: {fact_n:.0%}")
    configs['B_surgery_1'] = {
        'arithmetic': arith, 'arithmetic_code': arith_code,
        'fact_word': fact_w, 'fact_num': fact_n
    }
    del model; gc.collect(); torch.cuda.empty_cache()

    # C: After strong Surgery (strength=2.0)
    print("\n  === C: After Surgery (s=2.0) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    disperse_embeddings(model, tok, strength=2.0)
    arith = evaluate_arithmetic(model, tok, ARITHMETIC_TASKS)
    arith_code = evaluate_arithmetic(model, tok, ARITHMETIC_CODE)
    fact_w, fact_n = evaluate_facts(model, tok)
    print(f"    Arithmetic: {arith:.0%}, Arithmetic(code): {arith_code:.0%}")
    print(f"    Facts word: {fact_w:.0%}, Facts num: {fact_n:.0%}")
    configs['C_surgery_2'] = {
        'arithmetic': arith, 'arithmetic_code': arith_code,
        'fact_word': fact_w, 'fact_num': fact_n
    }
    del model; gc.collect(); torch.cuda.empty_cache()

    # D: After extreme Surgery (strength=3.0) - should maximize fact accuracy
    print("\n  === D: After Surgery (s=3.0) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    disperse_embeddings(model, tok, strength=3.0)
    arith = evaluate_arithmetic(model, tok, ARITHMETIC_TASKS)
    arith_code = evaluate_arithmetic(model, tok, ARITHMETIC_CODE)
    fact_w, fact_n = evaluate_facts(model, tok)
    print(f"    Arithmetic: {arith:.0%}, Arithmetic(code): {arith_code:.0%}")
    print(f"    Facts word: {fact_w:.0%}, Facts num: {fact_n:.0%}")
    configs['D_surgery_3'] = {
        'arithmetic': arith, 'arithmetic_code': arith_code,
        'fact_word': fact_w, 'fact_num': fact_n
    }
    del model; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase140_uncertainty.json'), 'w') as f:
        json.dump({'phase': '140', 'name': 'Arithmetic Uncertainty Principle',
                   'configs': configs}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = ['Baseline', 'Surgery\ns=1.0', 'Surgery\ns=2.0', 'Surgery\ns=3.0']
    keys = ['A_baseline', 'B_surgery_1', 'C_surgery_2', 'D_surgery_3']

    # Left: Arithmetic vs surgery strength
    ax = axes[0]
    arith_vals = [configs[k]['arithmetic'] for k in keys]
    arith_code_vals = [configs[k]['arithmetic_code'] for k in keys]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x-w/2, arith_vals, w, label='Natural', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, arith_code_vals, w, label='Code Mode', color='#2ecc71', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w/2, arith_vals[i]+0.02, f'{arith_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i]+w/2, arith_code_vals[i]+0.02, f'{arith_code_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Arithmetic Accuracy', fontsize=12)
    ax.set_title('Arithmetic Ability', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.3)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Tradeoff (arithmetic vs fact_num)
    ax = axes[1]
    fact_num_vals = [configs[k]['fact_num'] for k in keys]
    strengths = [0, 1, 2, 3]
    ax.plot(arith_vals, fact_num_vals, 'ro-', lw=2.5, markersize=12, zorder=5)
    for i, s in enumerate(strengths):
        ax.annotate(f's={s}', (arith_vals[i], fact_num_vals[i]),
                   fontsize=10, fontweight='bold', ha='left',
                   xytext=(10, 5), textcoords='offset points')
    ax.set_xlabel('Arithmetic Accuracy', fontsize=12)
    ax.set_ylabel('Numerical Fact Accuracy (pre-DPO)', fontsize=12)
    ax.set_title('The Uncertainty Principle?\nFact Accuracy vs Math Ability', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    ax.plot([0, 1], [1, 0], 'k--', alpha=0.3, label='Perfect tradeoff')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 140: The Arithmetic Uncertainty Principle',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase140_uncertainty.png'), dpi=150,
               bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    for k, name in zip(keys, ['Baseline', 'Surgery s=1', 'Surgery s=2', 'Surgery s=3']):
        c = configs[k]
        print(f"    {name:15s}: arith={c['arithmetic']:.0%} fact_num={c['fact_num']:.0%}")
    # Check if uncertainty principle holds
    a_base = configs['A_baseline']['arithmetic']
    a_surg = configs['D_surgery_3']['arithmetic']
    if a_surg < a_base * 0.5:
        print("  -> UNCERTAINTY PRINCIPLE CONFIRMED: Surgery destroys arithmetic!")
    else:
        print("  -> Arithmetic preserved after surgery (no tradeoff)")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 140] Complete.")

if __name__ == '__main__':
    main()
