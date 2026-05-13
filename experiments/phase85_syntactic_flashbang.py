# -*- coding: utf-8 -*-
"""
Phase 85: Syntactic Flashbang - Replace suppressor head outputs with mean.
Instead of ablating (zeroing), replace with average of all heads (uniform contribution).
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
    print("[P85] Syntactic Flashbang - Uniform Attention Forcing")
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval()
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

    hd = 64  # head dim for GPT-2
    n_heads = 12

    methods = {
        'baseline': [],
        'flashbang_L11H7': [(11, 7)],
        'flashbang_L9H6': [(9, 6)],
        'flashbang_top3': [(9, 6), (10, 7), (11, 7)],
        'flashbang_all_H7': [(9, 7), (10, 7), (11, 7)],
    }

    results = []
    for method_name, targets in methods.items():
        correct = 0
        total_rank = 0

        for prompt, answer in FACTS:
            inp = tok(prompt, return_tensors='pt')
            fact_id = tok.encode(answer)[0]

            # Hook into the block's forward to intercept attention output
            # before it gets added to residual. We modify the attention
            # output tensor (the concatenated head outputs before c_proj).
            handle_list = []
            for layer, head in targets:
                def make_hook(target_head):
                    def hook_fn(module, input):
                        # pre_hook: input is a tuple, input[0] is the tensor
                        inp_tensor = input[0]
                        if inp_tensor.dim() == 2:
                            # (seq, 768) - no batch dim
                            modified = inp_tensor.clone()
                            start = target_head * hd
                            end = start + hd
                            chunks = [modified[:, i*hd:(i+1)*hd] for i in range(n_heads)]
                            mean_chunk = torch.stack(chunks).mean(dim=0)
                            modified[:, start:end] = mean_chunk
                            return (modified,)
                        else:
                            # (batch, seq, 768)
                            modified = inp_tensor.clone()
                            start = target_head * hd
                            end = start + hd
                            chunks = [modified[:, :, i*hd:(i+1)*hd] for i in range(n_heads)]
                            mean_chunk = torch.stack(chunks).mean(dim=0)
                            modified[:, :, start:end] = mean_chunk
                            return (modified,)
                    return hook_fn
                # Hook on c_proj input (pre-projection concatenated attention output)
                h = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
                    make_hook(head))
                handle_list.append(h)

            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :]

            for h in handle_list:
                h.remove()

            rank = (logits.argsort(descending=True) == fact_id).nonzero().item() + 1
            if rank == 1:
                correct += 1
            total_rank += rank

        acc = correct / len(FACTS)
        mean_rank = total_rank / len(FACTS)
        results.append({'method': method_name, 'accuracy': acc, 'mean_rank': mean_rank})
        print(f"  {method_name}: acc={acc:.0%}, mean_rank={mean_rank:.1f}")

    out = {'phase': 85, 'name': 'Syntactic Flashbang', 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase85_syntactic_flashbang.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [r['method'].replace('flashbang_', 'FB:') for r in results]
    accs = [r['accuracy'] for r in results]
    ranks = [r['mean_rank'] for r in results]
    colors = ['gray', '#e74c3c', '#e67e22', '#2ecc71', '#3498db']

    axes[0].bar(names, accs, color=colors)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('P85: Flashbang Accuracy')
    axes[0].set_ylim(0, max(accs)*1.3 + 0.05)
    axes[0].tick_params(axis='x', rotation=20)

    axes[1].bar(names, ranks, color=colors)
    axes[1].set_ylabel('Mean Fact Rank')
    axes[1].set_title('P85: Mean Rank (lower=better)')
    axes[1].tick_params(axis='x', rotation=20)

    fig.suptitle('Phase 85: Syntactic Flashbang (Uniform Attention Forcing)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase85_syntactic_flashbang.png'), dpi=150)
    plt.close()
    print("[Phase 85] Complete.")

if __name__ == '__main__':
    main()
