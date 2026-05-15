# -*- coding: utf-8 -*-
"""
Phase 132: TSV Steering for Dark Matter (Truthfulness Separator Vector)
Since DPO cannot move numerical tokens in weight space (P129),
we extract a "truthfulness direction" vector and add it at inference time.

Method:
1. Collect hidden states at L22 for correct vs incorrect completions
2. Compute the mean difference vector (TSV = h_correct - h_incorrect)
3. At inference, add alpha * TSV to the hidden state at L22
4. Sweep alpha to find optimal steering strength
5. Compare: word facts vs numerical facts

This is an Inference-Time Intervention (ITI) approach:
- No training required
- No weight modification
- Just add a vector at the right layer

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

# Calibration set (for extracting TSV)
CALIBRATION_PAIRS = [
    ("The capital of Japan is", " Tokyo", " Osaka"),
    ("The capital of France is", " Paris", " Lyon"),
    ("The capital of Germany is", " Berlin", " Munich"),
    ("The capital of Italy is", " Rome", " Milan"),
    ("The capital of Spain is", " Madrid", " Barcelona"),
    ("Water freezes at", " 0", " 100"),
    ("The boiling point of water is", " 100", " 212"),
    ("The atomic number of carbon is", " 6", " 12"),
]

# Test set (unseen facts)
TEST_PAIRS = [
    ("The capital of the United Kingdom is", " London", " Manchester", "word"),
    ("The largest planet is", " Jupiter", " Saturn", "word"),
    ("The author of Romeo and Juliet is", " William", " Charles", "word"),
    ("The chemical symbol for gold is", " Au", " Ag", "word"),
    ("The largest ocean on Earth is the", " Pacific", " Atlantic", "word"),
    ("The number of planets in the solar system is", " 8", " 9", "number"),
    ("A year has", " 365", " 366", "number"),
    ("The atomic number of oxygen is", " 8", " 16", "number"),
    ("The speed of light is approximately", " 299", " 186", "number"),
    ("The number of continents is", " 7", " 6", "number"),
]


def extract_hidden_states(model, tok, prompt, answer, target_layers):
    """Get hidden states at specified layers for prompt+answer."""
    text = prompt + answer
    inp = tok(text, return_tensors='pt').to(DEVICE)
    prompt_len = tok(prompt, return_tensors='pt')['input_ids'].shape[1]
    hooks = {}
    handles = []
    for layer_idx in target_layers:
        def make_hook(idx):
            def hook_fn(module, input, output):
                # output is (hidden_states, ...) tuple
                h = output[0] if isinstance(output, tuple) else output
                # Get hidden state at the last prompt position (prediction point)
                hooks[idx] = h[0, prompt_len - 1, :].detach().clone()
            return hook_fn
        handle = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(handle)
    with torch.no_grad():
        model(**inp)
    for h in handles:
        h.remove()
    return hooks


def eval_with_steering(model, tok, pairs, target_layer, tsv, alpha):
    """Evaluate with TSV steering at target_layer."""
    results = {'word_correct': 0, 'word_total': 0,
               'num_correct': 0, 'num_total': 0, 'details': []}

    for item in pairs:
        if len(item) == 4:
            prompt, answer, _, cat = item
        else:
            prompt, answer, _ = item
            cat = 'unknown'

        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        c_id = tok.encode(answer)[-1]

        # Hook to add TSV to hidden state
        def make_hook(tsv_vec, a):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                # Add steering vector to all positions (broadcast)
                steering = a * tsv_vec.unsqueeze(0).unsqueeze(0).to(h.dtype).to(h.device)
                h_new = h + steering
                if isinstance(output, tuple):
                    return (h_new,) + output[1:]
                return h_new
            return hook_fn

        handle = model.model.layers[target_layer].register_forward_hook(
            make_hook(tsv, alpha))
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        handle.remove()

        pred = logits.argmax().item()
        correct = (pred == c_id)
        rank = int((logits.argsort(descending=True) == c_id).nonzero(as_tuple=True)[0].item())

        if cat == 'word':
            results['word_total'] += 1; results['word_correct'] += int(correct)
        elif cat == 'number':
            results['num_total'] += 1; results['num_correct'] += int(correct)

        results['details'].append({
            'prompt': prompt[:40], 'cat': cat, 'correct': correct, 'rank': rank,
        })

    total = results['word_total'] + results['num_total']
    results['total_acc'] = (results['word_correct']+results['num_correct']) / total if total else 0
    results['word_acc'] = results['word_correct'] / results['word_total'] if results['word_total'] else 0
    results['num_acc'] = results['num_correct'] / results['num_total'] if results['num_total'] else 0
    return results


def main():
    print("[P132] TSV Steering for Dark Matter")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

    target_layers = [10, 18, 22]

    # === Step 1: Extract TSV from calibration set ===
    print("\n  === Step 1: Extracting TSV ===")
    tsv_vectors = {l: [] for l in target_layers}

    for prompt, correct_ans, wrong_ans in CALIBRATION_PAIRS:
        h_correct = extract_hidden_states(model, tok, prompt, correct_ans, target_layers)
        h_wrong = extract_hidden_states(model, tok, prompt, wrong_ans, target_layers)
        for l in target_layers:
            diff = h_correct[l] - h_wrong[l]
            tsv_vectors[l].append(diff)

    # Average TSV per layer
    tsvs = {}
    for l in target_layers:
        tsv_mean = torch.stack(tsv_vectors[l]).mean(dim=0)
        tsv_norm = tsv_mean / (tsv_mean.norm() + 1e-8)
        tsvs[l] = tsv_norm
        print(f"    L{l}: TSV norm={tsv_mean.norm():.4f}, "
              f"mean component={tsv_mean.mean():.6f}")

    # === Step 2: Baseline ===
    print("\n  === Step 2: Baseline ===")
    base = eval_with_steering(model, tok, TEST_PAIRS, target_layers[0], tsvs[target_layers[0]], 0.0)
    print(f"    Total={base['total_acc']:.0%} Word={base['word_acc']:.0%} Num={base['num_acc']:.0%}")

    # === Step 3: Sweep alpha at each layer ===
    print("\n  === Step 3: TSV Steering Sweep ===")
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    sweep_results = {}

    for l in target_layers:
        sweep_results[l] = {}
        for alpha in alphas:
            r = eval_with_steering(model, tok, TEST_PAIRS, l, tsvs[l], alpha)
            sweep_results[l][alpha] = r
            print(f"    L{l} alpha={alpha:5.1f}: Total={r['total_acc']:.0%} "
                  f"Word={r['word_acc']:.0%} Num={r['num_acc']:.0%}")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save
    out = {'phase': '132', 'name': 'TSV Steering for Dark Matter',
           'baseline': {k: v for k, v in base.items() if k != 'details'}}
    out['sweep'] = {}
    for l in target_layers:
        out['sweep'][f'L{l}'] = {
            str(a): {'total': v['total_acc'], 'word': v['word_acc'], 'num': v['num_acc']}
            for a, v in sweep_results[l].items()
        }
    with open(os.path.join(RESULTS_DIR, 'phase132_tsv_steering.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Total accuracy by layer and alpha
    ax = axes[0]
    colors = {10: '#e74c3c', 18: '#2ecc71', 22: '#3498db'}
    for l in target_layers:
        totals = [sweep_results[l][a]['total_acc'] for a in alphas]
        ax.plot(alphas, totals, '-o', color=colors[l], label=f'L{l}', linewidth=2)
    ax.axhline(y=base['total_acc'], color='gray', linestyle='--', label='Baseline')
    ax.set_xlabel('Steering Strength (alpha)')
    ax.set_ylabel('Total Accuracy')
    ax.legend(); ax.set_title('TSV Steering: Total', fontweight='bold')
    ax.set_ylim(-0.05, 1.1)

    # Panel 2: Numerical accuracy specifically
    ax = axes[1]
    for l in target_layers:
        nums = [sweep_results[l][a]['num_acc'] for a in alphas]
        ax.plot(alphas, nums, '-s', color=colors[l], label=f'L{l} (num)', linewidth=2)
    ax.axhline(y=base['num_acc'], color='gray', linestyle='--', label='Baseline')
    ax.set_xlabel('Steering Strength (alpha)')
    ax.set_ylabel('Numerical Accuracy')
    ax.legend(); ax.set_title('TSV Steering: Numerical Facts', fontweight='bold')
    ax.set_ylim(-0.05, 1.1)

    # Panel 3: Word vs Num at best alpha per layer
    ax = axes[2]
    configs = ['Baseline']
    word_vals = [base['word_acc']]
    num_vals = [base['num_acc']]
    for l in target_layers:
        best_a = max(alphas, key=lambda a: sweep_results[l][a]['total_acc'])
        configs.append(f'L{l}\nalpha={best_a}')
        word_vals.append(sweep_results[l][best_a]['word_acc'])
        num_vals.append(sweep_results[l][best_a]['num_acc'])
    x = np.arange(len(configs))
    w = 0.35
    ax.bar(x-w/2, word_vals, w, label='Word', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, num_vals, w, label='Number', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel('Accuracy'); ax.legend()
    ax.set_title('Best TSV Config: Word vs Num', fontweight='bold')
    ax.set_ylim(0, 1.15)

    fig.suptitle('Phase 132: TSV Steering - Can Inference-Time Vectors Move Numerical Tokens?',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase132_tsv_steering.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 132] Complete.")

if __name__ == '__main__':
    main()
