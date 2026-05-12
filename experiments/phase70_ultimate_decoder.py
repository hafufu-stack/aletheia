# -*- coding: utf-8 -*-
"""
Phase 70: The Surgery + Oracle Fusion (Ultimate Decoder)
Combine ALL discoveries: L11H7 ablation (P64) + Oracle-guided contrast (P60)
+ Residual bypass (P67). The final Aletheia decoder.
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
    print("[P70] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

def get_fact_rank(logits, fact_id):
    return int((logits.argsort(descending=True) == fact_id).nonzero().item()) + 1

def main():
    print("=" * 70)
    print("  Phase 70: The Ultimate Aletheia Decoder")
    print("  Surgery + Oracle + Residual Bypass = ?")
    print("=" * 70)

    model, tok = load_model()
    n_heads = model.config.n_head
    hidden_dim = model.config.n_embd
    head_dim = hidden_dim // n_heads

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The Earth orbits the", [4252], "Sun"),
        ("The boiling point of water is", [1802], "100"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("Oxygen has the atomic number", [807], "8"),
        ("DNA stands for", [390], "de"),
        ("The speed of light is approximately", [22626], "299"),
    ]

    methods = {}

    # Method 1: Baseline (L12)
    print("\n[1] Baseline (L12)...")
    m1_correct = 0
    m1_ranks = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        with torch.no_grad():
            out = model(inp)
        r = get_fact_rank(out.logits[0, -1, :], fact_ids[0])
        m1_ranks.append(r)
        if torch.argmax(out.logits[0, -1, :]).item() in fact_ids:
            m1_correct += 1
    methods['L12_baseline'] = {'acc': m1_correct/len(tests), 'med_rank': float(np.median(m1_ranks))}

    # Method 2: L10 only
    print("[2] L10 only...")
    m2_correct = 0
    m2_ranks = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        h10 = {}
        def hook(m, a, o): h10['h'] = o[0][0, -1, :].detach()
        handle = model.transformer.h[10].register_forward_hook(hook)
        with torch.no_grad(): model(inp)
        handle.remove()
        normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
        logits = model.lm_head(normed).squeeze(0)
        r = get_fact_rank(logits, fact_ids[0])
        m2_ranks.append(r)
        if torch.argmax(logits).item() in fact_ids:
            m2_correct += 1
    methods['L10_only'] = {'acc': m2_correct/len(tests), 'med_rank': float(np.median(m2_ranks))}

    # Method 3: L11H7 Surgery
    print("[3] L11H7 Surgery...")
    m3_correct = 0
    m3_ranks = []
    def ablation_hook(module, args, output):
        hs = output[0].clone()
        start = 7 * head_dim
        end = start + head_dim
        hs[:, :, start:end] = 0.0
        return (hs,) + output[1:]
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        handle = model.transformer.h[11].register_forward_hook(ablation_hook)
        with torch.no_grad():
            out = model(inp)
        handle.remove()
        r = get_fact_rank(out.logits[0, -1, :], fact_ids[0])
        m3_ranks.append(r)
        if torch.argmax(out.logits[0, -1, :]).item() in fact_ids:
            m3_correct += 1
    methods['Surgery_L11H7'] = {'acc': m3_correct/len(tests), 'med_rank': float(np.median(m3_ranks))}

    # Method 4: Residual Bypass (alpha=0.9)
    print("[4] Residual Bypass (a=0.9)...")
    m4_correct = 0
    m4_ranks = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        h10 = {}
        def hook(m, a, o): h10['h'] = o[0][0, -1, :].detach()
        handle = model.transformer.h[10].register_forward_hook(hook)
        with torch.no_grad():
            out = model(inp)
        handle.remove()
        normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
        l10_logits = model.lm_head(normed).squeeze(0)
        mixed = 0.1 * out.logits[0, -1, :] + 0.9 * l10_logits
        r = get_fact_rank(mixed, fact_ids[0])
        m4_ranks.append(r)
        if torch.argmax(mixed).item() in fact_ids:
            m4_correct += 1
    methods['Residual_0.9'] = {'acc': m4_correct/len(tests), 'med_rank': float(np.median(m4_ranks))}

    # Method 5: FUSION - Surgery + Residual Bypass
    print("[5] Surgery + Residual...")
    m5_correct = 0
    m5_ranks = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        h10 = {}
        def hook10(m, a, o): h10['h'] = o[0][0, -1, :].detach()
        handle10 = model.transformer.h[10].register_forward_hook(hook10)
        handle11 = model.transformer.h[11].register_forward_hook(ablation_hook)
        with torch.no_grad():
            out = model(inp)
        handle10.remove()
        handle11.remove()
        normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
        l10_logits = model.lm_head(normed).squeeze(0)
        surgery_logits = out.logits[0, -1, :]
        mixed = 0.3 * surgery_logits + 0.7 * l10_logits
        r = get_fact_rank(mixed, fact_ids[0])
        m5_ranks.append(r)
        if torch.argmax(mixed).item() in fact_ids:
            m5_correct += 1
    methods['Surgery+Residual'] = {'acc': m5_correct/len(tests), 'med_rank': float(np.median(m5_ranks))}

    # Method 6: ULTIMATE - Surgery + Residual + Oracle (entropy-gated)
    print("[6] ULTIMATE: Surgery + Residual + Oracle...")
    m6_correct = 0
    m6_ranks = []
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        h10 = {}
        def hook10(m, a, o): h10['h'] = o[0][0, -1, :].detach()
        handle10 = model.transformer.h[10].register_forward_hook(hook10)
        handle11 = model.transformer.h[11].register_forward_hook(ablation_hook)
        with torch.no_grad():
            out = model(inp, output_attentions=True, return_dict=True)
        handle10.remove()
        handle11.remove()

        # Entropy
        ents = []
        for attn in out.attentions:
            for h in range(attn.shape[1]):
                a_vec = attn[0, h, -1, :].cpu().numpy()
                ents.append(float(-np.sum(a_vec * np.log(a_vec + 1e-12))))
        mean_ent = float(np.mean(ents))

        normed = model.transformer.ln_f(h10['h'].unsqueeze(0))
        l10_logits = model.lm_head(normed).squeeze(0)
        surgery_logits = out.logits[0, -1, :]

        # Dynamic mixing based on entropy
        if mean_ent > 0.8:
            # High entropy -> trust L10 more
            mixed = 0.1 * surgery_logits + 0.9 * l10_logits
        else:
            # Low entropy -> trust surgery output
            mixed = 0.5 * surgery_logits + 0.5 * l10_logits

        r = get_fact_rank(mixed, fact_ids[0])
        m6_ranks.append(r)
        if torch.argmax(mixed).item() in fact_ids:
            m6_correct += 1
    methods['ULTIMATE'] = {'acc': m6_correct/len(tests), 'med_rank': float(np.median(m6_ranks))}

    # Print all
    print("\n" + "=" * 70)
    print("  ALL METHODS COMPARISON:")
    for name, data in sorted(methods.items(), key=lambda x: (-x[1]['acc'], x[1]['med_rank'])):
        print(f"  {name:>25s}: acc={data['acc']:.0%} med_rank={data['med_rank']:.0f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = list(methods.keys())
    accs = [methods[n]['acc']*100 for n in names]
    meds = [methods[n]['med_rank'] for n in names]

    # Color coding
    color_map = {'L12_baseline': 'red', 'L10_only': 'green', 'Surgery_L11H7': 'blue',
                'Residual_0.9': 'orange', 'Surgery+Residual': 'purple', 'ULTIMATE': 'gold'}
    colors = [color_map.get(n, 'gray') for n in names]

    axes[0].bar(range(len(names)), accs, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, fontsize=7, rotation=30)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Fact Accuracy: All Methods')

    axes[1].bar(range(len(names)), meds, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=7, rotation=30)
    axes[1].set_ylabel('Median Fact Rank')
    axes[1].set_title('Median Rank (lower=better)')
    axes[1].set_yscale('log')

    plt.suptitle('Phase 70: Ultimate Aletheia Decoder', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase70_ultimate_decoder.png'), dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 70, 'name': 'Ultimate Aletheia Decoder',
        'methods': methods,
    }
    with open(os.path.join(RESULTS_DIR, 'phase70_ultimate_decoder.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("=" * 70)
    phase_complete(70)

if __name__ == '__main__':
    main()
