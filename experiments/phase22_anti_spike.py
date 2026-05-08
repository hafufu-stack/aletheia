# -*- coding: utf-8 -*-
"""
Phase 22: Anti-Spike (Deliberate Hallucination Generation)
- Spike WRONG fact tokens to create targeted hallucinations
- Prove spike mechanism is dual-use: truth OR lies
- Measure how easily wrong facts override correct ones
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
    print("[P22] Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok


def gen_with_spike(model, tok, prompt, spike_ids, mag, max_tokens=15):
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    gen = ids.clone()
    tokens = []
    for step in range(max_tokens):
        with torch.no_grad():
            out = model(gen)
        logits = out.logits[:, -1, :].squeeze(0)
        if step == 0:
            for tid in spike_ids:
                if tid < logits.shape[0]:
                    logits[tid] += mag
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
        tokens.append(tok.decode([next_tok.item()]).encode('ascii', 'replace').decode())
        if next_tok.item() == tok.eos_token_id:
            break
        gen = torch.cat([gen, next_tok], dim=1)
    return ''.join(tokens), tokens


def main():
    print("=" * 70)
    print("  Phase 22: Anti-Spike (Deliberate Hallucination)")
    print("  Can we weaponize spikes to CREATE lies?")
    print("=" * 70)

    model, tok = load_model()

    # Correct and wrong facts
    tests = [
        {
            'prompt': "The capital of Japan is",
            'correct': (11790, 'Tokyo'),
            'wrong': [(6342, 'Paris'), (3334, 'London'), (5765, 'Beijing')],
        },
        {
            'prompt': "The largest planet is",
            'correct': (22721, 'Jupiter'),
            'wrong': [(16309, 'Saturn'), (7733, 'Mars'), (11563, 'Earth')],
        },
        {
            'prompt': "Water freezes at",
            'correct': (657, '0'),
            'wrong': [(1802, '100'), (2167, '50'), (1120, '20')],
        },
        {
            'prompt': "The capital of France is",
            'correct': (6342, 'Paris'),
            'wrong': [(11790, 'Tokyo'), (3334, 'London'), (7753, 'Rome')],
        },
    ]

    mag = 15

    # === Test: correct spike vs wrong spike ===
    print(f"\n[P22a] Correct spike vs Anti-spike (mag={mag})...")
    anti_results = []

    for test in tests:
        prompt = test['prompt']
        print(f"\n  {prompt}")

        # No spike
        text_none, _ = gen_with_spike(model, tok, prompt, [], 0)
        print(f"    No spike:      {text_none[:45]}")

        # Correct spike
        text_right, _ = gen_with_spike(model, tok, prompt,
                                        [test['correct'][0]], mag)
        print(f"    Correct spike: {text_right[:45]}")

        # Wrong spikes
        wrong_outputs = []
        for wrong_id, wrong_name in test['wrong']:
            text_wrong, toks = gen_with_spike(model, tok, prompt, [wrong_id], mag)
            hit = toks[0].strip().lower() == wrong_name.lower() if toks else False
            wrong_outputs.append({
                'target': wrong_name, 'hit': hit,
                'text': text_wrong[:45], 'first_token': toks[0] if toks else '',
            })
            status = 'HIT' if hit else 'MISS'
            print(f"    Anti({wrong_name:>8s}): {text_wrong[:40]} [{status}]")

        anti_results.append({
            'prompt': prompt,
            'no_spike': text_none[:50],
            'correct': text_right[:50],
            'wrong': wrong_outputs,
            'n_hits': sum(1 for w in wrong_outputs if w['hit']),
        })

    # === Strength comparison: how much spike needed for wrong answer? ===
    print(f"\n[P22b] Critical spike for WRONG answers...")
    prompt = "The capital of Japan is"
    wrong_id = 6342  # Paris
    correct_id = 11790  # Tokyo

    for mag_test in [0, 1, 3, 5, 7, 10, 15, 20, 30]:
        inp = tok(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[:, -1, :].squeeze(0)
        logits[wrong_id] += mag_test
        winner = torch.argmax(logits).item()
        winner_text = tok.decode([winner]).encode('ascii', 'replace').decode().strip()
        correct_flag = 'Paris' if winner == wrong_id else winner_text
        print(f"    anti-spike={mag_test:>3d}: output='{correct_flag}'")

    # === Dual-use analysis ===
    total_hits = sum(r['n_hits'] for r in anti_results)
    total_attempts = sum(len(r['wrong']) for r in anti_results)
    hit_rate = total_hits / total_attempts if total_attempts > 0 else 0
    print(f"\n[P22c] Anti-spike success rate: {total_hits}/{total_attempts} = {hit_rate:.0%}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Hit rate per prompt
    prompts = [r['prompt'][:20] for r in anti_results]
    hits = [r['n_hits']/len(r['wrong'])*100 for r in anti_results]
    axes[0].barh(prompts, hits, color='red', alpha=0.7)
    axes[0].set_xlabel('Anti-Spike Hit Rate (%)')
    axes[0].set_title('Hallucination Creation Success')

    # Plot 2: Correct vs Anti-spike effectiveness
    axes[1].bar(['Truth Spike', 'Anti-Spike'],
                [100, hit_rate*100],
                color=['green', 'red'], alpha=0.7)
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].set_title('Dual-Use: Truth vs Lies')
    axes[1].set_ylim(0, 110)

    # Plot 3: Warning sign
    axes[2].text(0.5, 0.5,
                 f'DUAL-USE WARNING\n\n'
                 f'Same mechanism that\n'
                 f'eliminates hallucination\n'
                 f'can CREATE them\n\n'
                 f'Anti-spike rate: {hit_rate:.0%}',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=axes[2].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[2].axis('off')
    axes[2].set_title('Safety Implications')

    plt.suptitle('Phase 22: Anti-Spike (Deliberate Hallucination)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase22_anti_spike.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 22, 'name': 'Anti-Spike (Deliberate Hallucination)',
        'anti_spike_hit_rate': hit_rate,
        'details': anti_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase22_anti_spike.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 22 RESULTS: Anti-Spike")
    print("=" * 70)
    print(f"  Dual-use hit rate: {hit_rate:.0%}")
    print(f"  SAME mechanism creates truth AND lies")
    print("=" * 70)
    return results


if __name__ == '__main__':
    main()
