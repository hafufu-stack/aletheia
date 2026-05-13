# -*- coding: utf-8 -*-
"""
Phase 84: Semantic Cloaking - Dark Matterization of Fact Tokens
Project out suppressor weight directions from fact hidden states at L8,
making facts "invisible" to the Grammar Police.
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
    ("The Pythagorean theorem states that a squared plus b squared equals", " c"),
    ("The human body has a total of", " 206"),
    ("DNA stands for de", "oxy"),
    ("The Great Wall of China is located in", " China"),
    ("Shakespeare was born in", " Strat"),
]

def main():
    print("[P84] Semantic Cloaking - Dark Matterization")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

    # Extract suppressor weight directions (L9H6, L10H7, L11H7)
    suppressors = [(9, 6), (10, 7), (11, 7)]
    suppressor_dirs = []
    for layer, head in suppressors:
        attn = model.transformer.h[layer].attn
        hd = 64  # head dim for GPT-2
        # Extract W_Q, W_K, W_V projections for this head
        wq = attn.c_attn.weight[:, head*hd:(head+1)*hd].detach().float()
        wk = attn.c_attn.weight[:, 768 + head*hd:768 + (head+1)*hd].detach().float()
        wv = attn.c_attn.weight[:, 1536 + head*hd:1536 + (head+1)*hd].detach().float()
        # Combined projection direction (sum of squared norms)
        combined = torch.cat([wq, wk, wv], dim=1)  # (768, 192)
        # Get top principal component
        U, S, V = torch.svd(combined)
        top_dirs = U[:, :3]  # top 3 principal components
        suppressor_dirs.append(top_dirs)

    def project_out(hidden, dirs_list):
        """Remove suppressor directions from hidden state."""
        h = hidden.clone().float()
        for dirs in dirs_list:
            for i in range(dirs.shape[1]):
                d = dirs[:, i].to(h.device)
                d = d / d.norm()
                proj = (h @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
                h = h - proj
        return h

    results = []
    methods = {
        'baseline': None,
        'cloak_L8': 8,
        'cloak_L9': 9,
        'cloak_L10': 10,
    }

    for method_name, cloak_layer in methods.items():
        correct = 0
        total_rank = 0

        for prompt, answer in FACTS:
            inp = tok(prompt, return_tensors='pt')
            fact_id = tok.encode(answer)[0]
            hook_handle = None

            if cloak_layer is not None:
                def make_hook(cl):
                    def hook_fn(module, args, output):
                        hidden = output[0]
                        cloaked = project_out(hidden, suppressor_dirs)
                        return (cloaked.to(hidden.dtype),) + output[1:]
                    return hook_fn
                hook_handle = model.transformer.h[cloak_layer].register_forward_hook(make_hook(cloak_layer))

            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :]

            if hook_handle:
                hook_handle.remove()

            rank = (logits.argsort(descending=True) == fact_id).nonzero().item() + 1
            if rank == 1:
                correct += 1
            total_rank += rank

        acc = correct / len(FACTS)
        mean_rank = total_rank / len(FACTS)
        results.append({
            'method': method_name,
            'accuracy': acc,
            'mean_rank': mean_rank,
        })
        print(f"  {method_name}: acc={acc:.0%}, mean_rank={mean_rank:.1f}")

    # Save results
    out = {'phase': 84, 'name': 'Semantic Cloaking', 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase84_semantic_cloaking.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = [r['method'] for r in results]
    accs = [r['accuracy'] for r in results]
    ranks = [r['mean_rank'] for r in results]

    axes[0].bar(names, accs, color=['gray', '#2ecc71', '#3498db', '#9b59b6'])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('P84: Semantic Cloaking Accuracy')
    axes[0].set_ylim(0, max(accs)*1.3 + 0.05)

    axes[1].bar(names, ranks, color=['gray', '#2ecc71', '#3498db', '#9b59b6'])
    axes[1].set_ylabel('Mean Fact Rank')
    axes[1].set_title('P84: Mean Rank (lower=better)')

    fig.suptitle('Phase 84: Dark Matterization of Fact Tokens', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase84_semantic_cloaking.png'), dpi=150)
    plt.close()
    print("[Phase 84] Complete.")

if __name__ == '__main__':
    main()
