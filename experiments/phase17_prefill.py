# -*- coding: utf-8 -*-
"""
Phase 17: Prefill Slingshot
- Force first token to truth, then let model continue freely
- Measure how far the truth trajectory propagates
- Compare: raw prompt vs prefilled prompt
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_model():
    print("[P17] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def generate_free(model, tok, text, max_tokens=20):
    ids = tok(text, return_tensors='pt')['input_ids'].to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def generate_with_entropy(model, tok, text, max_tokens=20):
    ids = tok(text, return_tensors='pt')['input_ids'].to(DEVICE)
    gen = ids.clone()
    entropies = []
    tokens = []
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(gen)
        logits = out.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1).squeeze(0)
        h = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
        entropies.append(h)
        next_tok = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
        tokens.append(tok.decode([next_tok.item()]).encode('ascii', 'replace').decode())
        if next_tok.item() == tok.eos_token_id:
            break
        gen = torch.cat([gen, next_tok], dim=1)
    return ''.join(tokens), entropies, tokens


def main():
    print("=" * 70)
    print("  Phase 17: The Prefill Slingshot")
    print("  Force first token -> truth trajectory propagation")
    print("=" * 70)

    model, tok = load_model()

    # QA with prefill answers
    test_cases = [
        ("The capital of Japan is", " Tokyo"),
        ("The capital of France is", " Paris"),
        ("The capital of Germany is", " Berlin"),
        ("Water freezes at", " 0"),
        ("The largest planet is", " Jupiter"),
        ("DNA stands for", " deoxyribonucle"),
        ("Albert Einstein developed the theory of", " relativ"),
        ("The chemical symbol for gold is", " Au"),
        ("The speed of light is approximately", " 299"),
        ("Shakespeare wrote", " Hamlet"),
    ]

    print("\n[P17a] Comparing: no prefill vs 1-token prefill vs full-word prefill...")
    results = []

    for prompt, answer in test_cases:
        # No prefill (baseline)
        text_none, ent_none, _ = generate_with_entropy(model, tok, prompt, max_tokens=15)

        # 1-token prefill: append first token of answer
        first_token = tok.encode(answer)[0]
        first_tok_text = tok.decode([first_token])
        prefilled_1 = prompt + first_tok_text
        text_1, ent_1, _ = generate_with_entropy(model, tok, prefilled_1, max_tokens=15)

        # Full-word prefill
        prefilled_full = prompt + answer
        text_full, ent_full, _ = generate_with_entropy(model, tok, prefilled_full, max_tokens=15)

        results.append({
            'prompt': prompt,
            'answer': answer.strip(),
            'no_prefill': text_none[:60],
            'prefill_1tok': text_1[:60],
            'prefill_full': text_full[:60],
            'entropy_none': ent_none[:10],
            'entropy_1tok': ent_1[:10],
            'entropy_full': ent_full[:10],
        })

        print(f"\n  {prompt[:35]}")
        print(f"    No prefill: {text_none[:50]}")
        print(f"    1-token:    {first_tok_text.strip()} + {text_1[:45]}")
        print(f"    Full word:  {answer.strip()} + {text_full[:40]}")

    # === Entropy comparison ===
    print("\n[P17b] Entropy analysis...")
    mean_ent_none = np.mean([np.mean(r['entropy_none'][:5]) for r in results])
    mean_ent_1 = np.mean([np.mean(r['entropy_1tok'][:5]) for r in results])
    mean_ent_full = np.mean([np.mean(r['entropy_full'][:5]) for r in results])
    print(f"  Mean entropy (first 5 tokens):")
    print(f"    No prefill: {mean_ent_none:.2f}")
    print(f"    1-token:    {mean_ent_1:.2f}")
    print(f"    Full word:  {mean_ent_full:.2f}")

    # === Slingshot effect measurement ===
    print("\n[P17c] Slingshot effect: does prefill guide subsequent tokens?...")
    slingshot_scores = []
    for r in results:
        # Check if answer appears in generated text
        ans = r['answer'].lower()
        score_none = 1.0 if ans in r['no_prefill'].lower() else 0.0
        score_1 = 1.0 if ans in r['prefill_1tok'].lower() else 0.0
        score_full = 1.0 if ans in r['prefill_full'].lower() else 0.0
        slingshot_scores.append({
            'none': score_none, '1tok': score_1, 'full': score_full
        })

    acc_none = np.mean([s['none'] for s in slingshot_scores])
    acc_1 = np.mean([s['1tok'] for s in slingshot_scores])
    acc_full = np.mean([s['full'] for s in slingshot_scores])
    print(f"  Accuracy: none={acc_none:.0%}, 1-token={acc_1:.0%}, full={acc_full:.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Accuracy comparison
    methods = ['No Prefill', '1-Token', 'Full Word']
    accs = [acc_none*100, acc_1*100, acc_full*100]
    colors = ['red', 'orange', 'green']
    axes[0].bar(methods, accs, color=colors, alpha=0.7)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Prefill Slingshot Accuracy')
    axes[0].set_ylim(-5, 105)

    # Plot 2: Entropy trajectories (first example)
    if results:
        axes[1].plot(results[0]['entropy_none'][:10], 'r.-', label='No prefill', linewidth=2)
        axes[1].plot(results[0]['entropy_1tok'][:10], 'orange', marker='.', label='1-token', linewidth=2)
        axes[1].plot(results[0]['entropy_full'][:10], 'g.-', label='Full word', linewidth=2)
        axes[1].set_xlabel('Token Position')
        axes[1].set_ylabel('Entropy')
        axes[1].set_title(f'Entropy: {results[0]["prompt"][:25]}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Plot 3: Mean entropy comparison
    axes[2].bar(methods, [mean_ent_none, mean_ent_1, mean_ent_full],
                color=colors, alpha=0.7)
    axes[2].set_ylabel('Mean Entropy (first 5 tokens)')
    axes[2].set_title('Uncertainty After Prefill')

    plt.suptitle('Phase 17: The Prefill Slingshot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase17_prefill.png'), dpi=150, bbox_inches='tight')
    plt.close()

    out = {
        'phase': 17, 'name': 'Prefill Slingshot',
        'accuracy': {'none': acc_none, '1tok': acc_1, 'full': acc_full},
        'entropy': {'none': mean_ent_none, '1tok': mean_ent_1, 'full': mean_ent_full},
        'examples': results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase17_prefill.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 17 RESULTS: Prefill Slingshot")
    print("=" * 70)
    print(f"  Accuracy: none={acc_none:.0%}, 1-token={acc_1:.0%}, full={acc_full:.0%}")
    print(f"  Entropy: none={mean_ent_none:.2f}, 1-token={mean_ent_1:.2f}, full={mean_ent_full:.2f}")
    print("=" * 70)
    return out


if __name__ == '__main__':
    main()
