# -*- coding: utf-8 -*-
"""
Phase 177: Oracle Layer Optimization
P173 used L10 as Oracle -> only 40% accuracy.
P172 found Code-Math Neurons at L14-L15.

Hypothesis: Oracle should READ from L14-L15 (Code-Math hub),
not from L10 (generic middle layer).

Sweep Oracle layer from L2 to L22 to find optimal.

Model: Qwen2.5-0.5B (GPU, 24 layers)
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


def oracle_at_layer(model, tok, prompt, oracle_layer, fga_layer, fga_gain=5):
    """Single-token Oracle-Guided FGA at specified Oracle layer."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    # Capture Oracle hidden state
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

    # Logit Lens: Oracle's prediction
    if 'h' in oracle_hidden:
        oracle_logits = model.lm_head(oracle_hidden['h'].to(model.lm_head.weight.dtype))
        oracle_pred_id = oracle_logits.float().argmax(dim=-1).item()
    else:
        return None, None, None

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

    final_pred_id = logits.argmax().item()
    return final_pred_id, oracle_pred_id, tok.decode([oracle_pred_id]).strip()


def main():
    print("[P177] Oracle Layer Optimization")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    n_layers = model.config.num_hidden_layers  # 24
    fga_layer = n_layers - max(1, n_layers // 4)  # L18

    # Also measure: how accurate is the Logit Lens itself at each layer?
    layer_results = {}
    print("\n  === Oracle Layer Sweep ===")
    print(f"  FGA injection at L{fga_layer}")

    for oracle_l in range(2, n_layers - 2):
        correct_oracle = 0  # Logit Lens accuracy
        correct_final = 0   # Final output after FGA
        for prompt, expected in FACT_TEST:
            exp_id = tok.encode(expected)[-1]
            final_id, oracle_id, oracle_text = oracle_at_layer(
                model, tok, prompt, oracle_l, fga_layer, fga_gain=5)
            if oracle_id is not None:
                if oracle_id == exp_id: correct_oracle += 1
                if final_id == exp_id: correct_final += 1

        oracle_acc = correct_oracle / len(FACT_TEST)
        final_acc = correct_final / len(FACT_TEST)
        layer_results[oracle_l] = {'oracle_acc': oracle_acc, 'final_acc': final_acc}
        marker = " ***" if final_acc >= 0.8 else ""
        print(f"    Oracle=L{oracle_l:2d}: logit_lens={oracle_acc:.0%} "
              f"final_fga={final_acc:.0%}{marker}")

    # Find best Oracle layer
    best_layer = max(layer_results, key=lambda l: layer_results[l]['final_acc'])
    best_result = layer_results[best_layer]

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase177_oracle_layer.json'), 'w') as f:
        json.dump({'phase': '177', 'name': 'Oracle Layer Optimization',
                   'layer_results': {str(k): v for k, v in layer_results.items()},
                   'best_oracle_layer': best_layer,
                   'best_final_acc': best_result['final_acc'],
                   'fga_layer': fga_layer}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    layers = sorted(layer_results.keys())
    oracle_accs = [layer_results[l]['oracle_acc'] for l in layers]
    final_accs = [layer_results[l]['final_acc'] for l in layers]

    ax.plot(layers, oracle_accs, 'b-o', lw=2, markersize=6, label='Logit Lens Accuracy')
    ax.plot(layers, final_accs, 'r-s', lw=2.5, markersize=7, label='Final (Oracle+FGA) Accuracy')
    ax.axvline(x=best_layer, color='green', ls='--', lw=2, alpha=0.7,
               label=f'Best Oracle: L{best_layer}')
    ax.axvline(x=fga_layer, color='gray', ls=':', alpha=0.5,
               label=f'FGA injection: L{fga_layer}')
    # Shade Code-Math neuron region (P172: L14-L15)
    ax.axvspan(13.5, 15.5, alpha=0.1, color='orange', label='Code-Math Neurons (P172)')
    ax.set_xlabel('Oracle Layer', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_title('Phase 177: Oracle Layer Optimization\n'
                 'Which layer should the Oracle read from?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase177_oracle_layer.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    print(f"  -> Best Oracle layer: L{best_layer} (final acc: {best_result['final_acc']:.0%})")
    print(f"  -> Logit Lens at L{best_layer}: {best_result['oracle_acc']:.0%}")
    p173_layer = n_layers // 2
    p173_acc = layer_results.get(p173_layer, {}).get('final_acc', 0)
    print(f"  -> P173 used L{p173_layer}: {p173_acc:.0%}")
    if best_result['final_acc'] > p173_acc + 0.1:
        print(f"  -> IMPROVEMENT: +{best_result['final_acc'] - p173_acc:.0%}!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 177] Complete.")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
