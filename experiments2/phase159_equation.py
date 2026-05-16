# -*- coding: utf-8 -*-
"""
Phase 159: The Sword Energy Equation
Find the critical FGA gain g* for each model scale.
Derive the scaling law: g* = f(d, L)

Model: Qwen2.5-0.5B, 1.5B, 14B (all with Dual Surgery)
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
]


def dual_surgery(model, tok, embed_key, strength=2.0):
    """Apply Dual Surgery (embed + lm_head) generically."""
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365"]
    ids = list(set(tok.encode(t)[-1] for t in num_tokens))

    # Get embed tensor
    if embed_key == 'qwen':
        embed = model.model.embed_tokens.weight.data
    else:
        return  # unsupported

    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)

    # lm_head
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


def sweep_gain(model, tok, gains):
    """Sweep FGA gain and measure num accuracy."""
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - 4
    results = []
    for g in gains:
        correct = 0
        for prompt, expected, cat in TEST_SET:
            text = f"# {prompt}"
            exp_id = tok.encode(expected)[-1]
            hook = FGAHook(model, exp_id, g)
            hook.register(model, fga_layer)
            inp = tok(text, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()
            hook.remove()
            if logits.argmax().item() == exp_id: correct += 1
        acc = correct / len(TEST_SET)
        results.append({'gain': g, 'acc': acc})
    return results


def main():
    print("[P159] The Sword Energy Equation")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    gains = [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100]

    models_config = [
        ('Qwen/Qwen2.5-0.5B', '0.5B', None, torch.float32),
        ('Qwen/Qwen2.5-1.5B', '1.5B', None, torch.float16),
        ('Qwen/Qwen2.5-14B', '14B',
         BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16), torch.float16),
    ]

    all_results = {}
    model_info = {}

    for model_id, name, bnb, dtype in models_config:
        print(f"\n  === {name} ===")
        try:
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            if tok.pad_token is None: tok.pad_token = tok.eos_token

            if bnb:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, local_files_only=True, quantization_config=bnb,
                    device_map="auto", torch_dtype=dtype)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, local_files_only=True, torch_dtype=dtype).eval().to(DEVICE)

            d = model.config.hidden_size
            L = model.config.num_hidden_layers
            model_info[name] = {'d': d, 'L': L}

            # Apply Dual Surgery
            dual_surgery(model, tok, 'qwen', strength=2.0)

            # Sweep gains
            res = sweep_gain(model, tok, gains)
            all_results[name] = res

            for r in res:
                if r['acc'] > 0:
                    print(f"    g={r['gain']:4d}: {r['acc']:.0%}")

            # Find g*
            g_star = None
            for r in res:
                if r['acc'] >= 0.8:
                    g_star = r['gain']
                    break
            print(f"    g* (80% threshold) = {g_star}")

            del model; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"    SKIP: {e}")

    with open(os.path.join(RESULTS_DIR, 'phase159_equation.json'), 'w') as f:
        json.dump({'phase': '159', 'name': 'Sword Energy Equation',
                   'model_info': model_info,
                   'results': {k: v for k, v in all_results.items()}},
                  f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Gain sweep curves
    ax = axes[0]
    colors = {'0.5B': '#2ecc71', '1.5B': '#3498db', '14B': '#e74c3c'}
    for name, res in all_results.items():
        gs = [r['gain'] for r in res]
        accs = [r['acc'] for r in res]
        ax.plot(gs, accs, '-o', color=colors.get(name, 'gray'), lw=2.5,
               markersize=8, label=name)
    ax.axhline(y=0.8, color='gray', ls='--', alpha=0.5, label='g* threshold (80%)')
    ax.set_xlabel('FGA Gain (g)', fontsize=12)
    ax.set_ylabel('Number Accuracy', fontsize=12)
    ax.set_title('Critical Gain g* by Model Scale', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    # Right: g* vs model properties
    ax = axes[1]
    g_stars = []
    dims = []
    layers = []
    names_plot = []
    for name, res in all_results.items():
        g_star = None
        for r in res:
            if r['acc'] >= 0.8:
                g_star = r['gain']
                break
        if g_star and name in model_info:
            g_stars.append(g_star)
            dims.append(model_info[name]['d'])
            layers.append(model_info[name]['L'])
            names_plot.append(name)

    if len(g_stars) >= 2:
        ax.scatter(dims, g_stars, s=200, c=[colors.get(n, 'gray') for n in names_plot],
                  edgecolor='black', lw=2, zorder=5)
        for i, n in enumerate(names_plot):
            ax.annotate(f'{n}\nd={dims[i]}, L={layers[i]}\ng*={g_stars[i]}',
                       (dims[i], g_stars[i]), fontsize=9, ha='center',
                       xytext=(0, 15), textcoords='offset points')
        # Fit power law
        if len(g_stars) >= 2:
            log_d = np.log(dims)
            log_g = np.log(g_stars)
            slope = np.polyfit(log_d, log_g, 1)[0]
            ax.set_xlabel(f'Hidden Dimension (d)', fontsize=12)
            ax.set_ylabel(f'Critical Gain g*', fontsize=12)
            ax.set_title(f'Scaling Law: g* ~ d^{slope:.2f}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 159: The Sword Energy Equation',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase159_equation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 159] Complete.")

if __name__ == '__main__':
    main()
