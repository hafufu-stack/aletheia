# -*- coding: utf-8 -*-
"""
Phase 134: The Nuclear Option - Embedding Surgery + DPO + UAlign
Combines all three V5 breakthroughs:
1. Embedding Surgery (P130b): Spread numerical embeddings apart (cos 0.73 -> 0.02)
2. DPO (P127): Now works on numbers after surgery (0% -> 50%)
3. UAlign (P133): Abstain on remaining uncertain numbers

Pipeline:
  Step 1: Disperse number embeddings (strength=1.0, proven optimal)
  Step 2: DPO training on word + number facts
  Step 3: UAlign training for abstention on uncertain numbers
  Step 4: Evaluate combined system

Expected: The complete solution for numerical hallucination.

Model: Qwen2.5-0.5B (GPU, float32)
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

# DPO training pairs
DPO_PAIRS = [
    ("The capital of Japan is", " Tokyo", " Osaka", "word"),
    ("The capital of France is", " Paris", " Lyon", "word"),
    ("The capital of Germany is", " Berlin", " Munich", "word"),
    ("The capital of Italy is", " Rome", " Milan", "word"),
    ("The capital of Spain is", " Madrid", " Barcelona", "word"),
    ("Water freezes at", " 0", " 100", "number"),
    ("The boiling point of water is", " 100", " 212", "number"),
    ("The atomic number of carbon is", " 6", " 12", "number"),
    ("The speed of light is approximately", " 299", " 186", "number"),
]

# UAlign pairs (abstention for uncertain numbers)
UALIGN_PAIRS = [
    ("The melting point of iron is", " I", " 500"),
    ("The atomic number of nitrogen is", " I", " 14"),
    ("The distance to the moon in km is", " I", " 500"),
    ("The population of Tokyo in millions is", " I", " 50"),
]

# Comprehensive test set
TEST_SET = [
    # Known word facts (should be correct)
    ("The capital of the United Kingdom is", " London", "word", "known"),
    ("The largest planet is", " Jupiter", "word", "known"),
    ("The author of Romeo and Juliet is", " William", "word", "known"),
    # Known number facts (DPO trained)
    ("Water freezes at", " 0", "number", "trained"),
    ("The boiling point of water is", " 100", "number", "trained"),
    ("The atomic number of carbon is", " 6", "number", "trained"),
    # Uncertain number facts (should abstain)
    ("The melting point of iron is", " 15", "number", "uncertain"),
    ("The population of Tokyo in millions is", " 14", "number", "uncertain"),
    # Novel number facts (generalization test)
    ("A year has", " 365", "number", "novel"),
    ("Pi is approximately", " 3", "number", "novel"),
    ("The number of continents is", " 7", "number", "novel"),
    ("A decade has", " 10", "number", "novel"),
    # Novel word facts
    ("The largest ocean is the", " Pacific", "word", "novel"),
    ("The chemical symbol for gold is", " Au", "word", "novel"),
]


def dpo_loss(model, ref_model, tok, prompt, chosen, rejected, beta=0.05):
    text_c = prompt + chosen
    text_r = prompt + rejected
    inp_c = tok(text_c, return_tensors='pt').to(DEVICE)
    inp_r = tok(text_r, return_tensors='pt').to(DEVICE)
    plen = tok(prompt, return_tensors='pt')['input_ids'].shape[1]
    lc = model(**inp_c).logits[0, plen-1:-1, :].float().clamp(-100, 100)
    lr_ = model(**inp_r).logits[0, plen-1:-1, :].float().clamp(-100, 100)
    lp_c = F.log_softmax(lc, dim=-1).gather(1, inp_c['input_ids'][0, plen:].unsqueeze(1)).squeeze()
    lp_r = F.log_softmax(lr_, dim=-1).gather(1, inp_r['input_ids'][0, plen:].unsqueeze(1)).squeeze()
    if lp_c.dim() == 0: lp_c = lp_c.unsqueeze(0)
    if lp_r.dim() == 0: lp_r = lp_r.unsqueeze(0)
    with torch.no_grad():
        rlc = ref_model(**inp_c).logits[0, plen-1:-1, :].float().clamp(-100, 100)
        rlr = ref_model(**inp_r).logits[0, plen-1:-1, :].float().clamp(-100, 100)
        rlp_c = F.log_softmax(rlc, dim=-1).gather(1, inp_c['input_ids'][0, plen:].unsqueeze(1)).squeeze()
        rlp_r = F.log_softmax(rlr, dim=-1).gather(1, inp_r['input_ids'][0, plen:].unsqueeze(1)).squeeze()
        if rlp_c.dim() == 0: rlp_c = rlp_c.unsqueeze(0)
        if rlp_r.dim() == 0: rlp_r = rlp_r.unsqueeze(0)
    diff = beta * ((lp_c.sum() - rlp_c.sum()) - (lp_r.sum() - rlp_r.sum()))
    return -F.logsigmoid(diff)


def disperse_embeddings(model, tok, strength=1.0):
    """Spread numerical token embeddings apart."""
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366"]
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in num_tokens]
    vecs = embed[ids].clone()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()


def evaluate_full(model, tok, test_set):
    """Comprehensive evaluation with category breakdown."""
    results = []
    for prompt, expected, cat, split in test_set:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        exp_id = tok.encode(expected)[-1]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        pred_id = logits.argmax().item()
        pred_tok = tok.decode([pred_id])
        is_correct = (pred_id == exp_id)
        is_abstain = pred_tok.strip() in ['I', 'It', 'Unknown', 'N', 'None']
        is_number = pred_tok.strip().replace('.','').replace('-','').isdigit()
        results.append({
            'prompt': prompt[:45], 'cat': cat, 'split': split,
            'correct': is_correct, 'is_abstain': is_abstain,
            'is_number': is_number,
            'pred': pred_tok.encode('ascii','replace').decode().strip(),
            'conf': probs[pred_id].item(),
        })
    return results


def summarize(results, label):
    """Print category-level summary."""
    cats = {}
    for r in results:
        key = f"{r['cat']}_{r['split']}"
        if key not in cats: cats[key] = {'correct': 0, 'abstain': 0, 'total': 0}
        cats[key]['total'] += 1
        cats[key]['correct'] += int(r['correct'])
        cats[key]['abstain'] += int(r['is_abstain'])
    print(f"    {label}:")
    for k in sorted(cats.keys()):
        v = cats[k]
        print(f"      {k:20s}: correct={v['correct']}/{v['total']} "
              f"abstain={v['abstain']}/{v['total']}")
    return cats


def main():
    print("[P134] The Nuclear Option: Surgery + DPO + UAlign")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    configs = {}

    # === Config A: Vanilla baseline ===
    print("\n  === Config A: Baseline ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    r = evaluate_full(model, tok, TEST_SET)
    configs['A_baseline'] = summarize(r, 'Baseline')
    configs['A_baseline_raw'] = r
    del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # === Config B: DPO only (P129 approach, fails on numbers) ===
    print("\n  === Config B: DPO Only ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6)
    for ep in range(5):
        for prompt, chosen, rejected, _ in DPO_PAIRS:
            loss = dpo_loss(model, ref, tok, prompt, chosen, rejected)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
    model.eval()
    r = evaluate_full(model, tok, TEST_SET)
    configs['B_dpo_only'] = summarize(r, 'DPO Only')
    configs['B_dpo_only_raw'] = r
    del model, ref; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # === Config C: Surgery + DPO ===
    print("\n  === Config C: Surgery + DPO ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    disperse_embeddings(model, tok, strength=1.0)
    disperse_embeddings(ref, tok, strength=1.0)
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6)
    for ep in range(5):
        for prompt, chosen, rejected, _ in DPO_PAIRS:
            loss = dpo_loss(model, ref, tok, prompt, chosen, rejected)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
    model.eval()
    r = evaluate_full(model, tok, TEST_SET)
    configs['C_surgery_dpo'] = summarize(r, 'Surgery + DPO')
    configs['C_surgery_dpo_raw'] = r
    del ref; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # === Config D: Surgery + DPO + UAlign (THE NUCLEAR OPTION) ===
    print("\n  === Config D: Surgery + DPO + UAlign (Nuclear) ===")
    # Continue from Config C model, add UAlign training
    ref2 = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    disperse_embeddings(ref2, tok, strength=1.0)
    model.train()
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6)
    for ep in range(10):
        for prompt, chosen, rejected in UALIGN_PAIRS:
            loss = dpo_loss(model, ref2, tok, prompt, chosen, rejected)
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt2.step()
    model.eval()
    r = evaluate_full(model, tok, TEST_SET)
    configs['D_nuclear'] = summarize(r, 'NUCLEAR OPTION')
    configs['D_nuclear_raw'] = r
    del model, ref2; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save
    out = {'phase': '134', 'name': 'The Nuclear Option'}
    for k in ['A_baseline', 'B_dpo_only', 'C_surgery_dpo', 'D_nuclear']:
        out[k] = configs[k]
    with open(os.path.join(RESULTS_DIR, 'phase134_nuclear.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Per-config accuracy on trained number facts
    ax = axes[0]
    config_names = ['Baseline', 'DPO Only', 'Surgery\n+DPO', 'Nuclear\n(All 3)']
    config_keys = ['A_baseline', 'B_dpo_only', 'C_surgery_dpo', 'D_nuclear']

    def get_acc(cfg_raw, cat, split):
        items = [r for r in cfg_raw if r['cat'] == cat and r['split'] == split]
        if not items: return 0
        return sum(r['correct'] for r in items) / len(items)

    def get_abstain(cfg_raw, cat, split):
        items = [r for r in cfg_raw if r['cat'] == cat and r['split'] == split]
        if not items: return 0
        return sum(r['is_abstain'] for r in items) / len(items)

    word_known = [get_acc(configs[f'{k}_raw'], 'word', 'known') for k in config_keys]
    num_trained = [get_acc(configs[f'{k}_raw'], 'number', 'trained') for k in config_keys]
    x = np.arange(len(config_names))
    w = 0.35
    ax.bar(x-w/2, word_known, w, label='Word (known)', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, num_trained, w, label='Number (trained)', color='#e74c3c', alpha=0.8)
    for i in range(len(config_names)):
        ax.text(x[i]-w/2, word_known[i]+0.03, f'{word_known[i]:.0%}', ha='center', fontsize=9)
        ax.text(x[i]+w/2, num_trained[i]+0.03, f'{num_trained[i]:.0%}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(config_names, fontsize=9)
    ax.set_ylabel('Accuracy'); ax.legend(fontsize=8)
    ax.set_title('Known/Trained Facts', fontweight='bold')
    ax.set_ylim(0, 1.2)

    # Panel 2: Uncertain numbers - abstention rate
    ax = axes[1]
    num_uncertain_abstain = [get_abstain(configs[f'{k}_raw'], 'number', 'uncertain') for k in config_keys]
    num_novel_abstain = [get_abstain(configs[f'{k}_raw'], 'number', 'novel') for k in config_keys]
    ax.bar(x-w/2, num_uncertain_abstain, w, label='Uncertain (abstain)', color='#2ecc71', alpha=0.8)
    ax.bar(x+w/2, num_novel_abstain, w, label='Novel (abstain)', color='#f39c12', alpha=0.8)
    for i in range(len(config_names)):
        ax.text(x[i]-w/2, num_uncertain_abstain[i]+0.03, f'{num_uncertain_abstain[i]:.0%}', ha='center', fontsize=9)
        ax.text(x[i]+w/2, num_novel_abstain[i]+0.03, f'{num_novel_abstain[i]:.0%}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(config_names, fontsize=9)
    ax.set_ylabel('Abstention Rate'); ax.legend(fontsize=8)
    ax.set_title('Uncertain/Novel Number Abstention', fontweight='bold')
    ax.set_ylim(0, 1.2)

    # Panel 3: Overall behavior matrix
    ax = axes[2]
    categories = ['Word\nKnown', 'Num\nTrained', 'Num\nUncertain', 'Num\nNovel', 'Word\nNovel']
    splits = [('word','known'), ('number','trained'), ('number','uncertain'),
              ('number','novel'), ('word','novel')]
    nuc = configs['D_nuclear_raw']
    vals = []
    colors_list = []
    for cat, sp in splits:
        items = [r for r in nuc if r['cat'] == cat and r['split'] == sp]
        if not items:
            vals.append(0); colors_list.append('#95a5a6')
            continue
        acc = sum(r['correct'] for r in items) / len(items)
        abst = sum(r['is_abstain'] for r in items) / len(items)
        if acc > 0.5: vals.append(acc); colors_list.append('#2ecc71')
        elif abst > 0.3: vals.append(abst); colors_list.append('#f39c12')
        else: vals.append(max(acc, 0.05)); colors_list.append('#e74c3c')
    ax.bar(categories, vals, color=colors_list, alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')
    ax.set_ylabel('Rate (green=correct, orange=abstain, red=wrong)')
    ax.set_title('Nuclear Option: Full Behavior', fontweight='bold')
    ax.set_ylim(0, 1.2)

    fig.suptitle('Phase 134: The Nuclear Option - Embedding Surgery + DPO + UAlign',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase134_nuclear.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === FINAL VERDICT ===")
    for k, name in zip(config_keys, config_names):
        raw = configs[f'{k}_raw']
        total_correct = sum(r['correct'] for r in raw) / len(raw)
        total_abstain = sum(r['is_abstain'] for r in raw) / len(raw)
        total_fabricate = 1 - total_correct - total_abstain
        print(f"    {name.replace(chr(10),' '):15s}: correct={total_correct:.0%} "
              f"abstain={total_abstain:.0%} fabricate={total_fabricate:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 134] Complete.")

if __name__ == '__main__':
    main()
