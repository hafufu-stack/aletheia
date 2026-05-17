# -*- coding: utf-8 -*-
"""
Phase 193: Why Autoregressive FGA Fails (Root Cause Analysis)
P185/P188/P191: FGA ALWAYS makes autoregression worse (50% -> 43%).
Why? Two hypotheses:
  H1: FGA corrupts internal state, making next-token prediction worse
  H2: FGA shifts the output distribution so narrowly that diversity dies

Experiment: Measure hidden state divergence after FGA step.
Compare: base model's next-step prediction quality
         WITH and WITHOUT FGA applied to previous step.

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

# Use multi-token facts where we know ground truth
TESTS = [
    ("# The year Columbus reached America is", [" 1", "4", "9", "2"]),
    ("# Mount Fuji is", [" 3", "7", "7", "6"]),
    ("# The year the Moon landing was", [" 1", "9", "6", "9"]),
    ("# The year WWII ended is", [" 1", "9", "4", "5"]),
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

def main():
    print("[P193] Why Autoregressive FGA Fails")
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

    # For each test, generate token-by-token with and without FGA
    # Measure: (1) hidden state divergence at each step
    #          (2) next-token prediction quality after FGA corruption
    all_metrics = {'with_fga': [], 'without_fga': []}
    step_data = []

    for prompt, expected_tokens in TESTS:
        exp_ids = [tok.encode(t)[-1] for t in expected_tokens]
        # Path A: No FGA (pure greedy)
        ids_a = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        preds_a = []; entropies_a = []
        for step in range(len(exp_ids)):
            with torch.no_grad():
                logits = model(input_ids=ids_a).logits[0, -1, :].float()
            pred = logits.argmax().item()
            preds_a.append(pred)
            entropies_a.append(compute_entropy(logits))
            # Feed GROUND TRUTH for fair comparison
            ids_a = torch.cat([ids_a, torch.tensor([[exp_ids[step]]], device=DEVICE)], dim=1)

        # Path B: With FGA at each step (teacher-forced FGA)
        ids_b = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        preds_b = []; entropies_b = []
        for step in range(len(exp_ids)):
            target_id = exp_ids[step]
            unembed = model.lm_head.weight.data[target_id].float()
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
            with torch.no_grad():
                logits = model(input_ids=ids_b).logits[0, -1, :].float()
            handle.remove()
            pred = logits.argmax().item()
            preds_b.append(pred)
            entropies_b.append(compute_entropy(logits))
            # Feed the FGA-predicted token (NOT ground truth) to see corruption
            ids_b = torch.cat([ids_b, torch.tensor([[pred]], device=DEVICE)], dim=1)

        # Path C: FGA but feed ground truth anyway (isolate FGA state corruption)
        ids_c = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        preds_c = []; entropies_c = []
        for step in range(len(exp_ids)):
            target_id = exp_ids[step]
            unembed = model.lm_head.weight.data[target_id].float()
            d = unembed / (unembed.norm() + 1e-8)
            handle = model.model.layers[fga_layer].register_forward_hook(mk(d, 5))
            with torch.no_grad():
                logits = model(input_ids=ids_c).logits[0, -1, :].float()
            handle.remove()
            pred = logits.argmax().item()
            preds_c.append(pred)
            entropies_c.append(compute_entropy(logits))
            ids_c = torch.cat([ids_c, torch.tensor([[exp_ids[step]]], device=DEVICE)], dim=1)

        for step in range(len(exp_ids)):
            correct_a = 1 if preds_a[step] == exp_ids[step] else 0
            correct_b = 1 if preds_b[step] == exp_ids[step] else 0
            correct_c = 1 if preds_c[step] == exp_ids[step] else 0
            step_data.append({
                'step': step, 'prompt': prompt[:30],
                'no_fga_correct': correct_a, 'no_fga_entropy': entropies_a[step],
                'fga_autoregressive_correct': correct_b, 'fga_ar_entropy': entropies_b[step],
                'fga_teacher_correct': correct_c, 'fga_teacher_entropy': entropies_c[step],
            })

    # Aggregate by step
    steps = sorted(set(d['step'] for d in step_data))
    agg = {}
    for s in steps:
        sd = [d for d in step_data if d['step'] == s]
        agg[s] = {
            'no_fga': np.mean([d['no_fga_correct'] for d in sd]),
            'fga_ar': np.mean([d['fga_autoregressive_correct'] for d in sd]),
            'fga_teacher': np.mean([d['fga_teacher_correct'] for d in sd]),
            'no_fga_entropy': np.mean([d['no_fga_entropy'] for d in sd]),
            'fga_ar_entropy': np.mean([d['fga_ar_entropy'] for d in sd]),
        }
        print(f"  Step {s}: no_fga={agg[s]['no_fga']:.0%} "
              f"fga_ar={agg[s]['fga_ar']:.0%} fga_teacher={agg[s]['fga_teacher']:.0%}")

    with open(os.path.join(RESULTS_DIR, 'phase193_fga_failure.json'), 'w') as f:
        json.dump({'phase': '193', 'name': 'Why FGA Fails',
                   'aggregated': {str(k): v for k, v in agg.items()},
                   'step_data': step_data}, f, indent=2, default=str)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.plot(steps, [agg[s]['no_fga'] for s in steps], 'b-o', lw=2.5, label='No FGA (base)')
    ax.plot(steps, [agg[s]['fga_ar'] for s in steps], 'r-s', lw=2.5, label='FGA autoregressive')
    ax.plot(steps, [agg[s]['fga_teacher'] for s in steps], 'g-^', lw=2.5, label='FGA teacher-forced')
    ax.set_xlabel('Token Step', fontsize=13); ax.set_ylabel('Accuracy', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_title('Per-Step Accuracy', fontsize=13, fontweight='bold')

    ax = axes[1]
    ax.plot(steps, [agg[s]['no_fga_entropy'] for s in steps], 'b-o', lw=2, label='No FGA')
    ax.plot(steps, [agg[s]['fga_ar_entropy'] for s in steps], 'r-s', lw=2, label='FGA AR')
    ax.set_xlabel('Token Step', fontsize=13); ax.set_ylabel('Entropy', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_title('Entropy per Step', fontsize=13, fontweight='bold')

    plt.suptitle('Phase 193: Why Autoregressive FGA Fails\n'
                 'Diagnosing error accumulation mechanism',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase193_fga_failure.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    overall_a = np.mean([d['no_fga_correct'] for d in step_data])
    overall_b = np.mean([d['fga_autoregressive_correct'] for d in step_data])
    overall_c = np.mean([d['fga_teacher_correct'] for d in step_data])
    print(f"  -> No FGA overall: {overall_a:.0%}")
    print(f"  -> FGA autoregressive: {overall_b:.0%}")
    print(f"  -> FGA teacher-forced: {overall_c:.0%}")
    if overall_c > overall_b + 0.1:
        print("  -> FGA token prediction error cascades!")
    if overall_a > overall_b:
        print("  -> FGA HURTS autoregressive generation")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 193] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
