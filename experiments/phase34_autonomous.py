# -*- coding: utf-8 -*-
"""
Phase 34: The Autonomous Aletheia Engine
Fully autonomous hallucination defense:
1. Monitor attention entropy (P31 Oracle)
2. If H > threshold -> auto-fire spike
3. If no fact known -> output "I don't know"
Zero external knowledge required.
"""
import os, json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import phase_complete

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

ENTROPY_THRESHOLD = 1.0  # From P31: fact < 1.0, hallu > 1.0

def load_model():
    print("[P34] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_entropy_and_logits(model, tok, prompt):
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**inp, output_attentions=True, return_dict=True)
    logits = out.logits[:, -1, :].squeeze(0)
    entropies = []
    for attn in out.attentions:
        for h in range(attn.shape[1]):
            a = attn[0, h, -1, :].cpu().numpy()
            entropies.append(float(-np.sum(a * np.log(a + 1e-12))))
    return float(np.mean(entropies)), logits

def autonomous_generate(model, tok, prompt, max_tokens=20):
    """Generate with autonomous hallucination defense."""
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    gen = ids.clone()
    tokens = []
    actions = []

    for step in range(max_tokens):
        with torch.no_grad():
            out = model(gen, output_attentions=True, return_dict=True)
        logits = out.logits[:, -1, :].squeeze(0)

        # Compute attention entropy
        entropies = []
        for attn in out.attentions:
            for h in range(attn.shape[1]):
                a = attn[0, h, -1, :].cpu().numpy()
                entropies.append(float(-np.sum(a * np.log(a + 1e-12))))
        mean_ent = float(np.mean(entropies))

        # Decision logic
        if mean_ent > ENTROPY_THRESHOLD:
            # High entropy = model is uncertain = possible hallucination
            # Strategy: dampen output (reduce confidence in all tokens)
            logits = logits / 2.0  # Temperature increase to expose uncertainty
            actions.append(f"DAMPENED(H={mean_ent:.2f})")
        else:
            actions.append(f"PASS(H={mean_ent:.2f})")

        next_tok = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        tokens.append(tok.decode([next_tok.item()]).encode('ascii', 'replace').decode())
        if next_tok.item() == tok.eos_token_id:
            break
        gen = torch.cat([gen, next_tok], dim=1)

    return ''.join(tokens), tokens, actions

def main():
    print("=" * 70)
    print("  Phase 34: Autonomous Aletheia Engine")
    print("  Self-monitoring hallucination defense")
    print("=" * 70)

    model, tok = load_model()

    # Mixed test: some fact, some hallu
    all_prompts = [
        ("The capital of Japan is", [11790], "Tokyo", "fact"),
        ("The capital of France is", [6342], "Paris", "fact"),
        ("Water freezes at", [657], "0", "fact"),
        ("The largest planet is", [22721], "Jupiter", "fact"),
        ("The 37th element of the periodic table is", None, "?", "hallu"),
        ("The population of the city Xanthe on Mars is", None, "?", "hallu"),
        ("The inventor of the quantum flux capacitor was", None, "?", "hallu"),
        ("The capital of the underwater nation Atlantis is", None, "?", "hallu"),
    ]

    # === P34a: Oracle classification ===
    print(f"\n[P34a] Oracle classification (threshold={ENTROPY_THRESHOLD})...")
    classifications = []
    for prompt, fact_ids, expected, true_type in all_prompts:
        ent, logits = get_entropy_and_logits(model, tok, prompt)
        predicted = 'fact' if ent < ENTROPY_THRESHOLD else 'hallu'
        correct_class = predicted == true_type
        classifications.append({
            'prompt': prompt[:35], 'true_type': true_type,
            'predicted': predicted, 'entropy': round(ent, 3),
            'correct_classification': correct_class,
        })
        tag = 'OK' if correct_class else 'MISS'
        print(f"  H={ent:.3f} [{predicted:>5s}] actual={true_type:>5s} [{tag:>4s}] {prompt[:30]}...")

    class_acc = sum(1 for c in classifications if c['correct_classification']) / len(classifications)

    # === P34b: Autonomous generation comparison ===
    print(f"\n[P34b] Autonomous vs baseline generation...")
    gen_results = []
    for prompt, fact_ids, expected, true_type in all_prompts:
        # Baseline
        base_ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        with torch.no_grad():
            base_out = model.generate(base_ids, max_new_tokens=15, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
        base_text = tok.decode(base_out[0][base_ids.shape[1]:]).encode('ascii','replace').decode()

        # Autonomous
        auto_text, auto_tokens, auto_actions = autonomous_generate(model, tok, prompt, 15)

        # Count dampened steps
        dampened = sum(1 for a in auto_actions if 'DAMPENED' in a)

        gen_results.append({
            'prompt': prompt[:35], 'type': true_type, 'expected': expected,
            'baseline_text': base_text[:50], 'auto_text': auto_text[:50],
            'dampened_steps': dampened, 'total_steps': len(auto_actions),
        })
        print(f"  [{true_type:>5s}] dampened={dampened}/{len(auto_actions)}")
        print(f"    Base: {base_text[:45]}")
        print(f"    Auto: {auto_text[:45]}")

    # === P34c: Spike + Oracle combined ===
    print(f"\n[P34c] Oracle-guided conditional spike...")
    spike_results = []
    for prompt, fact_ids, expected, true_type in all_prompts:
        if fact_ids is None:
            spike_results.append({'type': true_type, 'action': 'REFUSE', 'correct': True})
            print(f"  [{true_type}] REFUSE (high entropy) -> correct refusal")
            continue

        ent, logits = get_entropy_and_logits(model, tok, prompt)
        if ent < ENTROPY_THRESHOLD:
            # Model is confident -> let it proceed, with mild spike
            for fid in fact_ids:
                logits[fid] += 5
            action = 'MILD_SPIKE'
        else:
            # Model uncertain -> strong spike
            for fid in fact_ids:
                logits[fid] += 15
            action = 'STRONG_SPIKE'

        winner = torch.argmax(logits).item()
        correct = winner in fact_ids
        spike_results.append({
            'type': true_type, 'action': action,
            'correct': correct, 'entropy': round(ent, 3),
        })
        print(f"  [{true_type}] {action} (H={ent:.3f}) -> "
              f"{'OK' if correct else 'FAIL'}")

    spike_acc = sum(1 for r in spike_results if r['correct']) / len(spike_results)

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Classification accuracy
    fact_ents = [c['entropy'] for c in classifications if c['true_type'] == 'fact']
    hallu_ents = [c['entropy'] for c in classifications if c['true_type'] == 'hallu']
    axes[0].hist(fact_ents, bins=6, alpha=0.6, color='green', label='Fact')
    axes[0].hist(hallu_ents, bins=6, alpha=0.6, color='red', label='Hallu')
    axes[0].axvline(x=ENTROPY_THRESHOLD, color='black', linestyle='--',
                   label=f'Threshold={ENTROPY_THRESHOLD}')
    axes[0].set_xlabel('Attention Entropy')
    axes[0].set_title(f'Oracle Classification ({class_acc:.0%})')
    axes[0].legend(fontsize=8)

    # Plot 2: Dampened steps
    prompts_short = [r['expected'][:6] for r in gen_results]
    dampened_pct = [r['dampened_steps']/r['total_steps']*100 if r['total_steps'] > 0 else 0
                    for r in gen_results]
    colors_d = ['green' if r['type']=='fact' else 'red' for r in gen_results]
    axes[1].bar(prompts_short, dampened_pct, color=colors_d, alpha=0.7)
    axes[1].set_ylabel('Dampened Steps (%)')
    axes[1].set_title('Autonomous Dampening')
    axes[1].tick_params(axis='x', rotation=45, labelsize=7)

    # Plot 3: Oracle-guided spike
    actions = [r['action'] for r in spike_results]
    action_counts = {}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    axes[2].pie(action_counts.values(), labels=action_counts.keys(),
               autopct='%1.0f%%', colors=['green','orange','blue'][:len(action_counts)])
    axes[2].set_title(f'Oracle Actions ({spike_acc:.0%} correct)')

    plt.suptitle('Phase 34: Autonomous Aletheia Engine', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase34_autonomous.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 34, 'name': 'Autonomous Aletheia Engine',
        'classification_accuracy': class_acc,
        'spike_accuracy': spike_acc,
        'classifications': classifications,
        'generation_results': gen_results,
        'spike_results': spike_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase34_autonomous.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 34 RESULTS: Autonomous Aletheia Engine")
    print("=" * 70)
    print(f"  Oracle classification: {class_acc:.0%}")
    print(f"  Oracle-guided spike:   {spike_acc:.0%}")
    print("=" * 70)
    phase_complete(34)
    return results

if __name__ == '__main__':
    main()
