# -*- coding: utf-8 -*-
"""
Phase 163: The Optimal FGA Layer
L-4 was chosen arbitrarily. What's the BEST layer for FGA?
Sweep ALL layers to find the optimal injection point.

Model: Qwen2.5-0.5B and 1.5B (GPU)
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

TEST_SET = [
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
    ("The capital of France is", " Paris", "word"),
    ("The capital of Japan is", " Tokyo", "word"),
]


def dual_surgery(model, tok, strength=2.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365"]
    ids = list(set(tok.encode(t)[-1] for t in num_tokens))
    embed = model.model.embed_tokens.weight.data
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)
    lm = model.lm_head.weight.data
    vecs_lm = lm[ids].clone().float()
    center_lm = vecs_lm.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs_lm[i] - center_lm
        direction = diff / (diff.norm() + 1e-8)
        lm[idx] += (strength * direction * lm[idx].float().norm()).to(lm.dtype)


class FGAHook:
    def __init__(self, model, target_token_id, gain):
        self.gain = gain
        unembed = model.lm_head.weight.data[target_token_id].float()
        self.direction = unembed / (unembed.norm() + 1e-8)
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        else:
            h = output.float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return h.to(output.dtype)

    def register(self, model, layer_idx):
        self.handle = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def sweep_layers(model, tok, gain=20):
    n_layers = model.config.num_hidden_layers
    results = []
    for layer in range(n_layers):
        correct_w, correct_n = 0, 0
        total_w, total_n = 0, 0
        for prompt, expected, cat in TEST_SET:
            text = f"# {prompt}"
            exp_id = tok.encode(expected)[-1]
            hook = FGAHook(model, exp_id, gain)
            hook.register(model, layer)
            inp = tok(text, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()
            hook.remove()
            if cat == 'word':
                total_w += 1; correct_w += int(logits.argmax().item() == exp_id)
            else:
                total_n += 1; correct_n += int(logits.argmax().item() == exp_id)
        results.append({
            'layer': layer, 'word': correct_w/max(1,total_w),
            'num': correct_n/max(1,total_n)
        })
    return results


def test_model(model_id, name, dtype, gain=20, strength=2.0):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=dtype).eval().to(DEVICE)
    dual_surgery(model, tok, strength=strength)
    results = sweep_layers(model, tok, gain=gain)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    # Find best layer
    best = max(results, key=lambda r: r['num'])
    print(f"    Best layer: {best['layer']} (L-{n_layers - best['layer']}): num={best['num']:.0%}")
    # Find all 100% layers
    perfect = [r for r in results if r['num'] >= 1.0]
    if perfect:
        print(f"    100% layers: {[r['layer'] for r in perfect]} "
              f"(L-{[n_layers - r['layer'] for r in perfect]})")

    del model; gc.collect(); torch.cuda.empty_cache()
    return {'name': name, 'd': d, 'n_layers': n_layers, 'gain': gain, 'results': results}


def main():
    print("[P163] The Optimal FGA Layer")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    all_results = {}

    # 0.5B
    print("\n  === 0.5B ===")
    all_results['0.5B'] = test_model('Qwen/Qwen2.5-0.5B', '0.5B', torch.float32, gain=5)

    # 1.5B
    print("\n  === 1.5B ===")
    all_results['1.5B'] = test_model('Qwen/Qwen2.5-1.5B', '1.5B', torch.float16, gain=20)

    with open(os.path.join(RESULTS_DIR, 'phase163_layers.json'), 'w') as f:
        json.dump({'phase': '163', 'name': 'Optimal FGA Layer',
                   'results': {k: v for k, v in all_results.items()}},
                  f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, (name, data) in enumerate(all_results.items()):
        ax = axes[i]
        layers = [r['layer'] for r in data['results']]
        num_acc = [r['num'] for r in data['results']]
        word_acc = [r['word'] for r in data['results']]
        n_layers = data['n_layers']

        ax.plot(layers, num_acc, 'r-o', lw=2, markersize=4, label='Number', alpha=0.8)
        ax.plot(layers, word_acc, 'b-s', lw=2, markersize=4, label='Word', alpha=0.8)
        ax.axvline(x=n_layers-4, color='green', ls='--', alpha=0.7, label=f'L-4 (={n_layers-4})')

        # Highlight best
        best = max(data['results'], key=lambda r: r['num'])
        ax.axvline(x=best['layer'], color='red', ls=':', alpha=0.7,
                  label=f'Best (L-{n_layers - best["layer"]})')

        ax.set_xlabel(f'Layer Index (0 to {n_layers-1})', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{name} (d={data["d"]}, g={data["gain"]})\n'
                    f'Best: layer {best["layer"]} (L-{n_layers-best["layer"]})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.15)

    plt.suptitle('Phase 163: Optimal FGA Layer\nWhich layer is the best injection point?',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase163_layers.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 163] Complete.")

if __name__ == '__main__':
    main()
