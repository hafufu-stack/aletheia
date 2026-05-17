# -*- coding: utf-8 -*-
"""
Phase 167: Autoregressive FGA Chaining
Multi-token fact extraction with sequential FGA.

The Limitation: Previous experiments only measure single-token accuracy.
Multi-digit numbers ("1538", "3776") span multiple BPE tokens.

Solution: Apply FGA token-by-token, shifting the target at each
autoregressive step. Ground truth tokens provided as in RAG.

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

# Multi-token facts: (prompt, full_answer, category)
MULTI_TOKEN_FACTS = [
    ("# The year Columbus reached America is", " 1492", "year"),
    ("# Mount Fuji is", " 3776", "height"),
    ("# The speed of light in km/s is", " 299792", "speed"),
    ("# The year the French Revolution began is", " 1789", "year"),
    ("# The population of Tokyo in millions is approximately", " 14", "pop"),
    ("# The year World War II ended is", " 1945", "year"),
    ("# Pi to 4 digits is", " 3.14", "math"),
    ("# The boiling point of water in Fahrenheit is", " 212", "temp"),
    ("# The year of the Moon landing is", " 1969", "year"),
    ("# The freezing point of water in Fahrenheit is", " 32", "temp"),
]

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"," 299"]


def disperse_weights(weight_tensor, tok, strength=2.0):
    w = weight_tensor.clone()
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = w[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        w[idx] += (strength * direction * w[idx].float().norm()).to(w.dtype)
    return w


class FGAHook:
    def __init__(self, direction, gain):
        self.gain = gain
        self.direction = direction
        self.handle = None
        self.active = True

    def hook_fn(self, module, input, output):
        if not self.active:
            return output
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        h = output.float()
        if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
        elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
        return h.to(output.dtype)

    def register(self, layer):
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def generate_greedy(model, tok, prompt, max_tokens=10):
    """Standard greedy generation without FGA."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    ids = inp['input_ids']
    generated = []
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1, :].float()
        next_id = logits.argmax().item()
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
    return tok.decode(generated)


def generate_chained_fga(model, tok, prompt, target_tokens, fga_gain=5):
    """Autoregressive FGA: inject target direction at each step."""
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - max(1, n_layers // 3)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    ids = inp['input_ids']
    generated = []
    per_token_correct = []

    for step, target_id in enumerate(target_tokens):
        # Set FGA direction toward this step's target
        unembed = model.lm_head.weight.data[target_id].float()
        direction = unembed / (unembed.norm() + 1e-8)
        hook = FGAHook(direction, fga_gain)
        hook.register(model.model.layers[fga_layer])

        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1, :].float()
        hook.remove()

        pred_id = logits.argmax().item()
        is_correct = (pred_id == target_id)
        generated.append(pred_id)
        per_token_correct.append(is_correct)

        # Always feed the CORRECT token for next step (teacher forcing for RAG)
        ids = torch.cat([ids, torch.tensor([[target_id]], device=DEVICE)], dim=1)

    output_text = tok.decode(generated)
    return output_text, per_token_correct


def main():
    print("[P167] Autoregressive FGA Chaining")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)

    # Apply Dual Surgery (for tied-weight models this modifies both)
    surgery_embed = disperse_weights(model.model.embed_tokens.weight.data, tok, strength=2.0)
    model.model.embed_tokens.weight.data.copy_(surgery_embed)

    results_all = {}
    gains_to_test = [0, 3, 5, 10]

    for gain in gains_to_test:
        print(f"\n  === FGA Gain = {gain} ===")
        all_per_token = []
        all_full_correct = []
        details = []

        for prompt, answer, cat in MULTI_TOKEN_FACTS:
            target_ids = tok.encode(answer)
            # Remove BOS if present
            if target_ids and target_ids[0] == tok.bos_token_id:
                target_ids = target_ids[1:]

            if gain == 0:
                # Baseline: greedy without FGA
                output = generate_greedy(model, tok, prompt, max_tokens=len(target_ids))
                expected_text = answer.strip()
                is_full_correct = output.strip().startswith(expected_text)
                per_tok = [0] * len(target_ids)  # Can't measure per-token without chaining
            else:
                output, per_tok = generate_chained_fga(model, tok, prompt, target_ids, gain)
                is_full_correct = all(per_tok)
                all_per_token.extend(per_tok)

            all_full_correct.append(is_full_correct)
            safe_answer = answer.encode('ascii', 'replace').decode()
            safe_output = output[:15].encode('ascii', 'replace').decode()
            print(f"    {safe_answer:>10s} -> {safe_output:15s} "
                  f"{'FULL OK' if is_full_correct else 'MISS'} "
                  f"per-tok: {per_tok}")
            details.append({'prompt': prompt[:40], 'answer': answer,
                            'output': output[:20], 'per_token': per_tok,
                            'full_correct': is_full_correct, 'cat': cat})

        full_acc = sum(all_full_correct) / len(all_full_correct)
        per_token_acc = sum(all_per_token) / max(1, len(all_per_token)) if all_per_token else 0
        print(f"    Full accuracy: {full_acc:.0%} ({sum(all_full_correct)}/{len(all_full_correct)})")
        if all_per_token:
            print(f"    Per-token accuracy: {per_token_acc:.0%}")

        results_all[f'gain_{gain}'] = {
            'full_acc': full_acc, 'per_token_acc': per_token_acc,
            'details': details}

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase167_chaining.json'), 'w') as f:
        json.dump({'phase': '167', 'name': 'Autoregressive FGA Chaining',
                   'results': results_all}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    gains = gains_to_test
    full_accs = [results_all[f'gain_{g}']['full_acc'] for g in gains]
    per_tok_accs = [results_all[f'gain_{g}']['per_token_acc'] for g in gains]
    ax.bar([str(g) for g in gains], full_accs, color='#e74c3c', alpha=0.8)
    for i, v in enumerate(full_accs):
        ax.text(i, v + 0.02, f'{v:.0%}', ha='center', fontsize=12, fontweight='bold')
    ax.set_xlabel('FGA Gain', fontsize=12)
    ax.set_ylabel('Full-Number Accuracy', fontsize=12)
    ax.set_title('Multi-Token Extraction\n(All tokens correct)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar([str(g) for g in gains[1:]], per_tok_accs[1:], color='#3498db', alpha=0.8)
    for i, v in enumerate(per_tok_accs[1:]):
        ax.text(i, v + 0.02, f'{v:.0%}', ha='center', fontsize=12, fontweight='bold')
    ax.set_xlabel('FGA Gain', fontsize=12)
    ax.set_ylabel('Per-Token Accuracy', fontsize=12)
    ax.set_title('Per-Token FGA Accuracy\n(Each digit individually)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.3); ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Phase 167: Autoregressive FGA Chaining\n'
                 'Can sequential FGA extract multi-digit numbers?',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase167_chaining.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    best_g = max(gains_to_test, key=lambda g: results_all[f'gain_{g}']['full_acc'])
    best = results_all[f'gain_{best_g}']
    print(f"  -> Best gain={best_g}: full={best['full_acc']:.0%}, "
          f"per-token={best['per_token_acc']:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 167] Complete.")


if __name__ == '__main__':
    main()
