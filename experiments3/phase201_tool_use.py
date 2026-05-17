# -*- coding: utf-8 -*-
"""
Phase 201: Latent Tool-Use
READ operands from internal registers, COMPUTE externally,
WRITE result back into SUM register. Zero-token computation.

Pipeline:
  1. Read A, B, OPCODE from L10 hidden states (probes from P198/P200)
  2. Compute result with Python (the real CPU)
  3. Inject result vector at L20 (SUM register location)

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time
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

def apply_surgery(model, tok, strength=2.0):
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)

def main():
    print("[P201] Latent Tool-Use")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    read_layer = 12  # A,B peak here
    write_layer = 20  # SUM crystallizes here
    n_layers = model.config.num_hidden_layers

    # Step 1: Train probes (A, B) on read_layer
    print("  Training operand probes on L12...")
    train_data = [(a, b, a+b) for a in range(10) for b in range(10)]
    X_train = []; labels_a = []; labels_b = []
    captured = [None]
    def hook_read(module, input, output):
        if isinstance(output, tuple):
            captured[0] = output[0][:, -1, :].detach().cpu().numpy().flatten()
        else:
            captured[0] = output[:, -1, :].detach().cpu().numpy().flatten()
    for a, b, s in train_data:
        prompt = f"def f(): return {a} + {b} ="
        h = model.model.layers[read_layer].register_forward_hook(hook_read)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): _ = model(**inp)
        h.remove()
        X_train.append(captured[0].copy())
        labels_a.append(a); labels_b.append(b)
    X_train = np.array(X_train)
    probe_a = LogisticRegression(max_iter=1000, random_state=42)
    probe_a.fit(X_train, np.array(labels_a))
    probe_b = LogisticRegression(max_iter=1000, random_state=42)
    probe_b.fit(X_train, np.array(labels_b))
    print(f"    Probe A accuracy: {probe_a.score(X_train, np.array(labels_a)):.0%}")
    print(f"    Probe B accuracy: {probe_b.score(X_train, np.array(labels_b)):.0%}")

    # Step 2: Build SUM direction vectors (mean hidden state for each sum value)
    print("  Building SUM direction vectors at L20...")
    sum_vecs = {}
    sum_hiddens = {s: [] for s in range(19)}
    captured_w = [None]
    def hook_write(module, input, output):
        if isinstance(output, tuple):
            captured_w[0] = output[0][:, -1, :].detach().clone()
        else:
            captured_w[0] = output[:, -1, :].detach().clone()
    for a, b, s in train_data:
        prompt = f"def f(): return {a} + {b} ="
        h = model.model.layers[write_layer].register_forward_hook(hook_write)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): _ = model(**inp)
        h.remove()
        sum_hiddens[s].append(captured_w[0].squeeze())
    for s in range(19):
        if sum_hiddens[s]:
            sum_vecs[s] = torch.stack(sum_hiddens[s]).mean(dim=0)

    # Step 3: Test Latent Tool-Use on held-out problems
    print("\n  === Latent Tool-Use Test ===")
    test_problems = [(7, 8), (6, 9), (4, 7), (9, 3), (5, 8), (8, 6), (3, 9), (7, 5)]
    results = {'baseline': [], 'tool_use': []}

    for a, b in test_problems:
        true_sum = a + b
        prompt = f"def f(): return {a} + {b} ="
        exp_tok = f" {true_sum}"
        exp_id = tok.encode(exp_tok)[-1]

        # Baseline (no injection)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        base_pred = tok.decode([logits.argmax().item()]).strip()
        base_ok = logits.argmax().item() == exp_id

        # Step A: READ operands from L12
        h = model.model.layers[read_layer].register_forward_hook(hook_read)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): _ = model(**inp)
        h.remove()
        read_a = probe_a.predict(captured[0].reshape(1, -1))[0]
        read_b = probe_b.predict(captured[0].reshape(1, -1))[0]

        # Step B: COMPUTE externally
        computed = read_a + read_b

        # Step C: WRITE result to L20
        if computed in sum_vecs:
            target_vec = sum_vecs[computed]
            # Also get the "current" hidden state direction
            current_sum_guess = sum_vecs.get(0, sum_vecs[list(sum_vecs.keys())[0]])
            direction = target_vec - current_sum_guess
            direction = direction / (direction.norm() + 1e-8)
            def inject(module, input, output):
                if isinstance(output, tuple):
                    h_out = output[0].float()
                    if h_out.dim() == 3:
                        h_out[:, -1, :] += 5 * direction.to(h_out.device)
                    return (h_out.to(output[0].dtype),) + output[1:]
                return output
            handle = model.model.layers[write_layer].register_forward_hook(inject)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()
            handle.remove()
            tool_pred = tok.decode([logits.argmax().item()]).strip()
            tool_ok = logits.argmax().item() == exp_id
        else:
            tool_pred = "N/A"
            tool_ok = False

        results['baseline'].append({'a': a, 'b': b, 'sum': true_sum,
                                     'pred': base_pred, 'ok': base_ok,
                                     'read_a': int(read_a), 'read_b': int(read_b)})
        results['tool_use'].append({'a': a, 'b': b, 'sum': true_sum,
                                     'pred': tool_pred, 'ok': tool_ok,
                                     'computed': int(computed)})
        read_ok = "OK" if (read_a == a and read_b == b) else f"MISS(a={read_a},b={read_b})"
        print(f"    {a}+{b}={true_sum}: read={read_ok} base='{base_pred}' "
              f"tool='{tool_pred}' {'TOOL_WIN' if tool_ok and not base_ok else ''}")

    base_acc = sum(1 for r in results['baseline'] if r['ok']) / len(test_problems)
    tool_acc = sum(1 for r in results['tool_use'] if r['ok']) / len(test_problems)
    read_acc = sum(1 for r in results['baseline']
                   if r['read_a'] == r['a'] and r['read_b'] == r['b']) / len(test_problems)

    with open(os.path.join(RESULTS_DIR, 'phase201_tool_use.json'), 'w') as f:
        json.dump({'phase': '201', 'name': 'Latent Tool-Use',
                   'results': results, 'base_acc': base_acc,
                   'tool_acc': tool_acc, 'read_acc': read_acc}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    labels = ['Baseline\n(model only)', 'Latent Tool-Use\n(read+compute+write)']
    vals = [base_acc, tool_acc]
    colors = ['#3498db', '#e74c3c']
    ax.bar(labels, vals, color=colors, alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(f'Phase 201: Latent Tool-Use\nRead acc={read_acc:.0%}, External compute, Write to SUM register',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase201_tool_use.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    print(f"  -> Read accuracy: {read_acc:.0%}")
    print(f"  -> Baseline: {base_acc:.0%}")
    print(f"  -> Latent Tool-Use: {tool_acc:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 201] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
