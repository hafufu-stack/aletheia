# -*- coding: utf-8 -*-
"""
Phase 173: Oracle-Guided FGA (Self-Referential Truth Engine)
Use the model's OWN intermediate prediction (Logit Lens at L10)
as the FGA target. No external ground truth needed.

The "Autopoietic" loop:
  1. Forward pass captures L10's top-1 prediction (the Oracle)
  2. That prediction's lm_head vector becomes FGA direction
  3. FGA at L_final-4 amplifies L10's belief
  4. Model outputs the amplified prediction

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

MULTI_TOKEN = [
    ("# The year Columbus reached America is", " 1492"),
    ("# Mount Fuji is", " 3776"),
    ("# The year the Moon landing was", " 1969"),
    ("# The year WWII ended is", " 1945"),
]

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"," 299"]


def apply_surgery(model, tok, strength=2.0):
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)


def oracle_guided_generate(model, tok, prompt, max_tokens=5, fga_gain=5,
                           oracle_layer=None, fga_layer=None):
    """Self-referential generation: L_oracle's prediction -> FGA at L_fga."""
    n_layers = model.config.num_hidden_layers
    if oracle_layer is None:
        oracle_layer = n_layers // 2  # middle layer as Oracle
    if fga_layer is None:
        fga_layer = n_layers - max(1, n_layers // 4)

    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    ids = inp['input_ids']
    generated_tokens = []
    oracle_predictions = []

    for step in range(max_tokens):
        # Step 1: Forward pass to Oracle layer, capture hidden state
        oracle_hidden = {}
        def oracle_hook(module, input, output):
            if isinstance(output, tuple):
                oracle_hidden['h'] = output[0][:, -1, :].detach().float()
            else:
                oracle_hidden['h'] = output[:, -1, :].detach().float()

        h_oracle = model.model.layers[oracle_layer].register_forward_hook(oracle_hook)

        # First pass: get Oracle's prediction (no FGA)
        with torch.no_grad():
            _ = model(input_ids=ids)
        h_oracle.remove()

        # Logit Lens: project Oracle hidden state through lm_head
        if 'h' in oracle_hidden:
            oracle_logits = model.lm_head(oracle_hidden['h'].to(model.lm_head.weight.dtype))
            oracle_pred_id = oracle_logits.float().argmax(dim=-1).item()
        else:
            oracle_pred_id = 0

        oracle_predictions.append(oracle_pred_id)

        # Step 2: Use Oracle's prediction as FGA target
        unembed = model.lm_head.weight.data[oracle_pred_id].float()
        direction = unembed / (unembed.norm() + 1e-8)

        # FGA hook at later layer
        fga_handle = None
        def fga_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0].float()
                if h.dim() == 3: h[:, -1, :] += fga_gain * direction.to(h.device)
                elif h.dim() == 2: h[-1, :] += fga_gain * direction.to(h.device)
                return (h.to(output[0].dtype),) + output[1:]
            h = output.float()
            if h.dim() == 3: h[:, -1, :] += fga_gain * direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += fga_gain * direction.to(h.device)
            return h.to(output.dtype)

        fga_handle = model.model.layers[fga_layer].register_forward_hook(fga_hook)

        # Second pass: with FGA boosting Oracle's prediction
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1, :].float()
        fga_handle.remove()

        # Output token
        pred_id = logits.argmax().item()
        generated_tokens.append(pred_id)
        ids = torch.cat([ids, torch.tensor([[pred_id]], device=DEVICE)], dim=1)

    output_text = tok.decode(generated_tokens)
    oracle_text = tok.decode(oracle_predictions)
    return output_text, oracle_text, generated_tokens, oracle_predictions


def main():
    print("[P173] Oracle-Guided FGA")
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
    results = {}

    # A: Single-token facts (compare baseline, teacher-forced, oracle-guided)
    print("\n  === A: Single-Token Facts ===")
    for mode_name, fga_gain, use_oracle in [
        ('baseline', 0, False), ('teacher_g5', 5, False), ('oracle_g5', 5, True)
    ]:
        correct = 0
        details = []
        for prompt, expected in FACT_TEST:
            exp_id = tok.encode(expected)[-1]
            if use_oracle:
                output, oracle_out, gen_ids, _ = oracle_guided_generate(
                    model, tok, prompt, max_tokens=1, fga_gain=fga_gain)
                pred_id = gen_ids[0]
            elif fga_gain > 0:
                # Teacher-forced FGA
                fga_layer = n_layers - max(1, n_layers // 4)
                unembed = model.lm_head.weight.data[exp_id].float()
                direction = unembed / (unembed.norm() + 1e-8)
                def make_hook(d, g):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0].float()
                            if h.dim() == 3: h[:, -1, :] += g * d.to(h.device)
                            return (h.to(output[0].dtype),) + output[1:]
                        h = output.float()
                        if h.dim() == 3: h[:, -1, :] += g * d.to(h.device)
                        return h.to(output.dtype)
                    return hook_fn
                handle = model.model.layers[fga_layer].register_forward_hook(
                    make_hook(direction, fga_gain))
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    logits = model(**inp).logits[0, -1, :].float()
                handle.remove()
                pred_id = logits.argmax().item()
            else:
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    logits = model(**inp).logits[0, -1, :].float()
                pred_id = logits.argmax().item()

            is_correct = (pred_id == exp_id)
            if is_correct: correct += 1
            details.append({'expected': expected.strip(),
                            'pred': tok.decode([pred_id]).strip(),
                            'correct': is_correct})

        acc = correct / len(FACT_TEST)
        print(f"    {mode_name}: {acc:.0%}")
        results[f'single_{mode_name}'] = {'acc': acc, 'details': details}

    # B: Multi-token facts (oracle-guided autoregressive)
    print("\n  === B: Multi-Token Oracle-Guided ===")
    for fga_gain in [0, 5, 10]:
        multi_results = []
        for prompt, expected in MULTI_TOKEN:
            n_tokens = len(tok.encode(expected))
            output, oracle_out, _, _ = oracle_guided_generate(
                model, tok, prompt, max_tokens=n_tokens, fga_gain=fga_gain)
            is_match = output.strip().startswith(expected.strip())
            safe_exp = expected.strip().encode('ascii', 'replace').decode()
            safe_out = output.strip()[:15].encode('ascii', 'replace').decode()
            print(f"    g={fga_gain}: {safe_exp:>8s} -> {safe_out:15s} "
                  f"oracle={oracle_out[:15]} {'OK' if is_match else 'MISS'}")
            multi_results.append({'expected': expected, 'output': output[:20],
                                  'oracle': oracle_out[:20], 'match': is_match})
        full_acc = sum(r['match'] for r in multi_results) / len(multi_results)
        results[f'multi_g{fga_gain}'] = {'acc': full_acc, 'details': multi_results}

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase173_oracle_fga.json'), 'w') as f:
        json.dump({'phase': '173', 'name': 'Oracle-Guided FGA',
                   'results': results}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    modes = ['baseline', 'teacher_g5', 'oracle_g5']
    accs = [results[f'single_{m}']['acc'] for m in modes]
    labels = ['Baseline\n(no FGA)', 'Teacher\nFGA g=5', 'Oracle\nFGA g=5']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    ax.bar(labels, accs, color=colors, alpha=0.8)
    for i, v in enumerate(accs):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Single-Token: Oracle vs Teacher FGA', fontsize=13, fontweight='bold')

    ax = axes[1]
    mg = [0, 5, 10]
    maccs = [results[f'multi_g{g}']['acc'] for g in mg]
    ax.bar([str(g) for g in mg], maccs, color='#9b59b6', alpha=0.8)
    for i, v in enumerate(maccs):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=14, fontweight='bold')
    ax.set_xlabel('FGA Gain', fontsize=12)
    ax.set_ylabel('Full Match Accuracy', fontsize=12)
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Multi-Token: Oracle-Guided Generation', fontsize=13, fontweight='bold')

    plt.suptitle('Phase 173: Oracle-Guided FGA\nModel uses its OWN L10 prediction as FGA target',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase173_oracle_fga.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    oracle_acc = results['single_oracle_g5']['acc']
    teacher_acc = results['single_teacher_g5']['acc']
    print(f"  -> Oracle FGA: {oracle_acc:.0%} vs Teacher FGA: {teacher_acc:.0%}")
    if oracle_acc >= teacher_acc * 0.8:
        print("  -> AUTOPOIESIS WORKS! Oracle matches Teacher!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 173] Complete.")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
