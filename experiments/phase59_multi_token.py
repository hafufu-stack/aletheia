# -*- coding: utf-8 -*-
"""
Phase 59: Multi-Token Fact Surfing
DT2's idea: autoregressive generation using ONLY L10 (bypassing L11-L12).
Test if multi-token facts emerge correctly with degraded grammar.
"Semantic Superfluidity" - facts without grammar.
"""
import os, json, sys
import numpy as np
import torch
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

def load_model():
    print("[P59] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    tok.pad_token = tok.eos_token
    return mdl, tok

def generate_from_layer(model, tok, prompt, layer, max_tokens=15):
    """Autoregressive generation decoding from intermediate layer only."""
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    gen = ids.clone()
    tokens_out = []

    for step in range(max_tokens):
        h_store = {}
        def make_hook(l):
            def fn(m, a, o):
                h_store[l] = o[0][0, -1, :].detach()
            return fn
        handle = model.transformer.h[layer].register_forward_hook(make_hook(layer))
        with torch.no_grad():
            model(gen)
        handle.remove()

        normed = model.transformer.ln_f(h_store[layer].unsqueeze(0))
        logits = model.lm_head(normed).squeeze(0)
        next_tok = torch.argmax(logits).item()
        tokens_out.append(tok.decode([next_tok]).encode('ascii', 'replace').decode())
        if next_tok == tok.eos_token_id:
            break
        gen = torch.cat([gen, torch.tensor([[next_tok]], device=DEVICE)], dim=1)

    return ''.join(tokens_out), tokens_out

def main():
    print("=" * 70)
    print("  Phase 59: Multi-Token Fact Surfing")
    print("  Autoregressive generation from intermediate layers")
    print("=" * 70)

    model, tok = load_model()

    tests = [
        ("The capital of Japan is", "Tokyo"),
        ("The capital of France is", "Paris"),
        ("The largest planet is", "Jupiter"),
        ("Albert Einstein developed the theory of", "relativity"),
        ("The Earth orbits the", "Sun"),
        ("Shakespeare wrote", "Hamlet"),
        ("Water freezes at", "0 degrees"),
        ("DNA stands for", "deoxyribonucleic acid"),
        ("The chemical symbol for gold is", "Au"),
        ("The boiling point of water is", "100 degrees"),
    ]

    results = []
    for exit_layer in [6, 8, 10, 11]:
        print(f"\n[P59] Generating from Layer {exit_layer}...")
        layer_results = []
        for prompt, expected in tests:
            text, toks = generate_from_layer(model, tok, prompt, exit_layer, 15)
            # Baseline (full model)
            inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
            with torch.no_grad():
                base_out = model.generate(inp, max_new_tokens=15, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
            base_text = tok.decode(base_out[0][inp.shape[1]:]).encode('ascii','replace').decode()

            # Check if expected fact appears in output
            fact_in_layer = expected.lower().split()[0] in text.lower()
            fact_in_base = expected.lower().split()[0] in base_text.lower()

            layer_results.append({
                'prompt': prompt, 'expected': expected,
                'layer_text': text[:60], 'base_text': base_text[:60],
                'fact_in_layer': fact_in_layer, 'fact_in_base': fact_in_base,
            })
            tag_l = 'FACT' if fact_in_layer else 'miss'
            tag_b = 'FACT' if fact_in_base else 'miss'
            print(f"  L{exit_layer} [{tag_l}]: {text[:50]}")
            if exit_layer == 10:
                print(f"  L12 [{tag_b}]: {base_text[:50]}")

        layer_fact_rate = sum(1 for r in layer_results if r['fact_in_layer']) / len(tests)
        base_fact_rate = sum(1 for r in layer_results if r['fact_in_base']) / len(tests)
        results.append({
            'layer': exit_layer, 'fact_rate': layer_fact_rate,
            'base_fact_rate': base_fact_rate, 'details': layer_results,
        })
        print(f"  L{exit_layer} fact rate: {layer_fact_rate:.0%} vs L12: {base_fact_rate:.0%}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    layers = [r['layer'] for r in results]
    fact_rates = [r['fact_rate']*100 for r in results]
    base_rates = [r['base_fact_rate']*100 for r in results]
    x = range(len(layers))
    axes[0].bar([i-0.2 for i in x], fact_rates, 0.4, label='Layer N Gen', color='green', alpha=0.7)
    axes[0].bar([i+0.2 for i in x], base_rates, 0.4, label='Full Model', color='red', alpha=0.7)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([f'L{l}' for l in layers])
    axes[0].set_ylabel('Fact Appearance Rate (%)')
    axes[0].set_title('Multi-Token Fact Recovery by Layer')
    axes[0].legend()

    # Per-prompt comparison for L10
    l10_res = [r for r in results if r['layer'] == 10][0]
    labels = [d['expected'][:8] for d in l10_res['details']]
    l_facts = [1 if d['fact_in_layer'] else 0 for d in l10_res['details']]
    b_facts = [1 if d['fact_in_base'] else 0 for d in l10_res['details']]
    x2 = range(len(labels))
    axes[1].bar([i-0.2 for i in x2], l_facts, 0.4, label='L10 Gen', color='green', alpha=0.7)
    axes[1].bar([i+0.2 for i in x2], b_facts, 0.4, label='Full Gen', color='red', alpha=0.7)
    axes[1].set_xticks(list(x2))
    axes[1].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[1].set_ylabel('Fact Found (1/0)')
    axes[1].set_title('L10 vs Full: Per-Prompt Fact Recovery')
    axes[1].legend()

    plt.suptitle('Phase 59: Multi-Token Fact Surfing', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase59_multi_token.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 59, 'name': 'Multi-Token Fact Surfing',
        'results': results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase59_multi_token.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    for r in results:
        print(f"  L{r['layer']}: fact_rate={r['fact_rate']:.0%}")
    print("=" * 70)
    phase_complete(59)

if __name__ == '__main__':
    main()
