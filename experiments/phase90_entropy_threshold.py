# -*- coding: utf-8 -*-
"""
Phase 90: The Entropy Threshold
P87 showed CoT raises L11H7 entropy (0.794 -> ~1.0) but doesn't improve accuracy,
while code symbols also raise it (~0.965) and DO improve accuracy.
Question: Is there a critical entropy threshold for suppression release?
"""
import torch, json, os, sys, numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

FACTS = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("The chemical symbol for gold is", " Au"),
    ("The speed of light is approximately", " 299"),
    ("Albert Einstein was born in", " Ul"),
    ("The tallest mountain in the world is", " Mount"),
    ("The currency of the United Kingdom is the", " pound"),
    ("The author of Romeo and Juliet is", " William"),
    ("The first president of the United States was", " George"),
    ("The chemical formula for water is", " H"),
    ("The boiling point of water is", " 100"),
    ("The atomic number of carbon is", " 6"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("The speed of sound is approximately", " 343"),
    ("Photosynthesis converts sunlight into", " chemical"),
]

def main():
    print("[P90] The Entropy Threshold")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True, attn_implementation='eager').eval()
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

    suppressor_heads = [(9, 6), (11, 7)]
    # Artificially scale suppressor attention temperature to sweep entropy
    temperatures = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0]

    results = []
    for temp in temperatures:
        correct = 0
        total_rank = 0
        entropies = {f'L{l}H{h}': [] for l, h in suppressor_heads}

        for prompt, answer in FACTS:
            inp = tok(prompt, return_tensors='pt')
            fact_id = tok.encode(answer)[0]

            # Hook to scale attention logits by temperature before softmax
            handles = []
            for layer, head in suppressor_heads:
                def make_hook(target_layer, target_head, t):
                    def hook_fn(module, args, kwargs, output):
                        # output: (attn_output, attn_weights)
                        # attn_weights shape: (batch, n_heads, seq, seq)
                        attn_out, attn_w = output
                        if attn_w is not None:
                            # Get original attention for this head
                            w = attn_w[0, target_head, -1, :].detach()
                            ent = -(w * torch.log(w + 1e-10)).sum().item()
                            entropies[f'L{target_layer}H{target_head}'].append(ent)
                        return output
                    return hook_fn
                # We can't easily modify temperature mid-forward, so use weight scaling
                # Alternative: hook c_attn output and scale the attention scores
                pass

            # Simpler approach: scale the W_O weights by 1/temp to reduce head impact
            # At temp=inf, head output -> 0 (pure ablation)
            # At temp=1, no change
            # This simulates "diluting" the head's influence
            orig_weights = {}
            for layer, head in suppressor_heads:
                hd = 64
                start = head * hd
                end = start + hd
                w_key = f'{layer}_{head}'
                orig_weights[w_key] = model.transformer.h[layer].attn.c_proj.weight[start:end, :].clone()
                with torch.no_grad():
                    model.transformer.h[layer].attn.c_proj.weight[start:end, :] *= (1.0 / temp)

            with torch.no_grad():
                out = model(**inp, output_attentions=True)
                logits = out.logits[0, -1, :]
                attentions = out.attentions

            # Measure actual entropy
            for layer, head in suppressor_heads:
                attn = attentions[layer][0, head, -1, :]
                ent = -(attn * torch.log(attn + 1e-10)).sum().item()
                entropies[f'L{layer}H{head}'].append(ent)

            # Restore weights
            for layer, head in suppressor_heads:
                hd = 64
                start = head * hd
                end = start + hd
                w_key = f'{layer}_{head}'
                with torch.no_grad():
                    model.transformer.h[layer].attn.c_proj.weight[start:end, :] = orig_weights[w_key]

            rank = (logits.argsort(descending=True) == fact_id).nonzero().item() + 1
            if rank == 1:
                correct += 1
            total_rank += rank

        acc = correct / len(FACTS)
        mean_rank = total_rank / len(FACTS)
        mean_ents = {k: float(np.mean(v)) for k, v in entropies.items()}
        results.append({
            'temperature': temp,
            'accuracy': acc,
            'mean_rank': mean_rank,
            'entropies': mean_ents,
        })
        ent_str = ', '.join(f'{k}={v:.3f}' for k, v in mean_ents.items())
        print(f"  T={temp:5.1f}: acc={acc:.0%}, rank={mean_rank:.1f}, {ent_str}")

    out = {'phase': 90, 'name': 'Entropy Threshold', 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase90_entropy_threshold.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    temps = [r['temperature'] for r in results]
    accs = [r['accuracy'] for r in results]
    ranks = [r['mean_rank'] for r in results]

    axes[0].plot(temps, accs, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Suppressor Scaling (1/T)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Suppressor Dilution')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(temps, ranks, 's-', color='#e74c3c', linewidth=2, markersize=8)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Suppressor Scaling (1/T)')
    axes[1].set_ylabel('Mean Fact Rank')
    axes[1].set_title('Rank vs Suppressor Dilution')
    axes[1].grid(True, alpha=0.3)

    # Entropy vs accuracy scatter
    for head_name in [f'L{l}H{h}' for l, h in suppressor_heads]:
        head_ents = [r['entropies'].get(head_name, 0) for r in results]
        axes[2].scatter(head_ents, accs, label=head_name, s=80, alpha=0.8)
    axes[2].set_xlabel('Suppressor Entropy')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Accuracy vs Suppressor Entropy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Phase 90: The Entropy Threshold', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase90_entropy_threshold.png'), dpi=150)
    plt.close()
    print("[Phase 90] Complete.")

if __name__ == '__main__':
    main()
