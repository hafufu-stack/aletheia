# -*- coding: utf-8 -*-
"""
Phase 110: Cross-Lingual Truth Invariance
Is truth language-independent? Test same facts in English, Japanese, French.
Qwen-1.5B (supports multilingual).
GPU.
"""
import torch, json, os, gc, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Same facts in 3 languages
MULTILINGUAL_FACTS = {
    'English': [
        ("The capital of Japan is", " Tokyo"),
        ("The capital of France is", " Paris"),
        ("The capital of Germany is", " Berlin"),
        ("The largest planet is", " Jupiter"),
        ("The chemical symbol for gold is", " Au"),
        ("Water freezes at", " 0"),
        ("The boiling point of water is", " 100"),
        ("The atomic number of carbon is", " 6"),
    ],
    'Japanese': [
        ("日本の首都は", "東京"),
        ("フランスの首都は", "パリ"),
        ("ドイツの首都は", "ベルリン"),
        ("太陽系最大の惑星は", "木星"),
        ("金の元素記号は", "Au"),
        ("水が凍る温度は摂氏", "0"),
        ("水の沸点は摂氏", "100"),
        ("炭素の原子番号は", "6"),
    ],
    'French': [
        ("La capitale du Japon est", " Tokyo"),
        ("La capitale de la France est", " Paris"),
        ("La capitale de l'Allemagne est", " Berlin"),
        ("La plus grande planete est", " Jupiter"),
        ("Le symbole chimique de l'or est", " Au"),
        ("L'eau gele a", " 0"),
        ("Le point d'ebullition de l'eau est", " 100"),
        ("Le numero atomique du carbone est", " 6"),
    ],
}

def logit_lens_layer(model, tok, prompt, layer_idx):
    hidden = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden['h'] = output[0][:, -1, :].detach()
        else:
            hidden['h'] = output[:, -1, :].detach()
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**tok(prompt, return_tensors='pt').to(DEVICE))
    handle.remove()
    h = model.model.norm(hidden['h'])
    logits = model.lm_head(h).squeeze()
    return logits

def main():
    print(f"[P110] Cross-Lingual Truth Invariance (device={DEVICE})")

    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B', local_files_only=True, torch_dtype=torch.float16
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', local_files_only=True)
    n_layers = model.config.num_hidden_layers

    results = {}
    for lang, facts in MULTILINGUAL_FACTS.items():
        print(f"\n  === {lang} ===")
        # Standard accuracy
        correct = 0
        total_rank = 0
        for prompt, answer in facts:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if fact_tokens else 0
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :]
            rank = (logits.argsort(descending=True) == fact_id).nonzero()
            rank = rank.item() + 1 if rank.numel() > 0 else logits.shape[0]
            if rank == 1: correct += 1
            total_rank += rank

        acc = correct / len(facts)
        mean_rank = total_rank / len(facts)
        print(f"    Output accuracy: {acc:.0%}, mean rank: {mean_rank:.1f}")

        # Logit Lens: find best layer
        layer_accs = {}
        for l_idx in range(n_layers):
            l_correct = 0
            for prompt, answer in facts:
                fact_tokens = tok.encode(answer)
                fact_id = fact_tokens[-1] if fact_tokens else 0
                logits = logit_lens_layer(model, tok, prompt, l_idx)
                if logits.argmax().item() == fact_id:
                    l_correct += 1
            layer_accs[l_idx] = l_correct / len(facts)

        best_layer = max(layer_accs, key=layer_accs.get)
        best_rel = best_layer / n_layers
        print(f"    Best layer: L{best_layer} (relative: {best_rel:.3f}), acc: {layer_accs[best_layer]:.0%}")

        results[lang] = {
            'output_accuracy': acc, 'mean_rank': mean_rank,
            'best_layer': best_layer, 'best_relative': best_rel,
            'best_layer_accuracy': layer_accs[best_layer],
            'layer_accuracies': {str(k): v for k, v in layer_accs.items()},
        }

    # Cross-lingual analysis
    print(f"\n  === CROSS-LINGUAL ANALYSIS ===")
    for lang in results:
        r = results[lang]
        print(f"  {lang:10s}: acc={r['output_accuracy']:.0%}, best=L{r['best_layer']} ({r['best_relative']:.3f})")

    relatives = [results[l]['best_relative'] for l in results]
    print(f"  Best layer std: {np.std(relatives):.4f}")
    print(f"  Truth invariance: {'CONFIRMED' if np.std(relatives) < 0.05 else 'PARTIAL'}")

    out = {
        'phase': 110, 'name': 'Cross-Lingual Truth Invariance',
        'model': 'Qwen/Qwen2.5-1.5B', 'results': results,
        'cross_lingual_std': float(np.std(relatives)),
    }
    with open(os.path.join(RESULTS_DIR, 'phase110_crosslingual.json'), 'w') as f:
        json.dump(out, f, indent=2)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    langs = list(results.keys())
    accs = [results[l]['output_accuracy'] for l in langs]
    bests = [results[l]['best_relative'] for l in langs]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    axes[0].bar(langs, accs, color=colors)
    axes[0].set_ylabel('Output Accuracy'); axes[0].set_title('Accuracy by Language'); axes[0].set_ylim(0, 1.1)

    axes[1].bar(langs, bests, color=colors)
    axes[1].axhline(y=0.94, color='gold', linestyle='--', linewidth=2, label='0.94 constant')
    axes[1].set_ylabel('Best Layer (relative)'); axes[1].set_title('0.94 Constant by Language')
    axes[1].legend(); axes[1].set_ylim(0.8, 1.0)

    fig.suptitle('Phase 110: Cross-Lingual Truth Invariance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase110_crosslingual.png'), dpi=150)
    plt.close()

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("[Phase 110] Complete.")

if __name__ == '__main__':
    main()
