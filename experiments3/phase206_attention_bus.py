# -*- coding: utf-8 -*-
"""
Phase 206: Attention Bus Analysis
Which attention heads connect operand tokens to the output position?
These heads form the "data bus" of the Neural CPU.

For "def f(): return A + B =", which heads at "=" attend to "A" and "B"?

Model: Qwen2.5-0.5B (GPU)
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
    print("[P206] Attention Bus Analysis")
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
    n_heads = model.config.num_attention_heads

    # Collect attention patterns for arithmetic
    problems = [(a, b) for a in [3, 5, 7, 9] for b in [2, 4, 6, 8]]
    # For each problem, find which heads attend from "=" to A and B tokens

    attn_to_A = np.zeros((n_layers, n_heads))
    attn_to_B = np.zeros((n_layers, n_heads))
    attn_to_op = np.zeros((n_layers, n_heads))  # attention to "+"
    count = 0

    for a, b in problems:
        prompt = f"def f(): return {a} + {b} ="
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        tokens = tok.convert_ids_to_tokens(inp['input_ids'][0])

        # Find positions of A, +, B, = in token list
        pos_eq = len(tokens) - 1  # last token is "="
        # Find numeric token positions
        a_str = str(a); b_str = str(b)
        pos_a = None; pos_b = None; pos_plus = None
        for i, t in enumerate(tokens):
            clean = t.replace('_', '').replace(' ', '').replace('\u0120', '')
            if clean == a_str and pos_a is None and i > 3:  # skip "def f"
                pos_a = i
            if clean == '+':
                pos_plus = i
            if clean == b_str and pos_plus is not None and pos_b is None:
                pos_b = i

        if pos_a is None or pos_b is None:
            continue

        with torch.no_grad():
            outputs = model(**inp, output_attentions=True)

        for l in range(n_layers):
            attn = outputs.attentions[l][0].float()  # (n_heads, seq, seq)
            # Attention from "=" position to each other position
            attn_from_eq = attn[:, pos_eq, :]  # (n_heads, seq)
            attn_to_A[l] += attn_from_eq[:, pos_a].cpu().numpy()
            attn_to_B[l] += attn_from_eq[:, pos_b].cpu().numpy()
            if pos_plus is not None:
                attn_to_op[l] += attn_from_eq[:, pos_plus].cpu().numpy()
        count += 1

    attn_to_A /= max(count, 1)
    attn_to_B /= max(count, 1)
    attn_to_op /= max(count, 1)

    # Find top attention heads
    print(f"\n  Analyzed {count} problems, {n_layers} layers x {n_heads} heads")
    print("\n  === Top 10 Heads Attending to Operand A ===")
    flat_a = [(attn_to_A[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
    flat_a.sort(reverse=True)
    for val, l, h in flat_a[:10]:
        print(f"    L{l:2d} H{h:2d}: {val:.4f}")

    print("\n  === Top 10 Heads Attending to Operand B ===")
    flat_b = [(attn_to_B[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
    flat_b.sort(reverse=True)
    for val, l, h in flat_b[:10]:
        print(f"    L{l:2d} H{h:2d}: {val:.4f}")

    print("\n  === Top 10 Heads Attending to '+' ===")
    flat_op = [(attn_to_op[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
    flat_op.sort(reverse=True)
    for val, l, h in flat_op[:10]:
        print(f"    L{l:2d} H{h:2d}: {val:.4f}")

    results = {
        'attn_to_A': attn_to_A.tolist(),
        'attn_to_B': attn_to_B.tolist(),
        'attn_to_op': attn_to_op.tolist(),
        'n_problems': count
    }
    with open(os.path.join(RESULTS_DIR, 'phase206_attention_bus.json'), 'w') as f:
        json.dump({'phase': '206', 'name': 'Attention Bus', 'results': results}, f, indent=2)

    # Visualize: heatmap of attention to A and B
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for i, (data, title) in enumerate([
        (attn_to_A, 'Attention to Operand A'),
        (attn_to_B, 'Attention to Operand B'),
        (attn_to_op, 'Attention to "+"')
    ]):
        ax = axes[i]
        im = ax.imshow(data.T, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Head', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.suptitle('Phase 206: Attention Bus\nWhich heads connect "=" to operands?',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase206_attention_bus.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 206] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
