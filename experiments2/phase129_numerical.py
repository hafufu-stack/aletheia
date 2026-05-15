# -*- coding: utf-8 -*-
"""
Phase 129: Why Do Numerical Tokens Resist DPO?
P128 showed DPO fails on numerical facts ("Water freezes at 0",
"boiling point is 100", "speed of light is 299").

Hypotheses:
H1: Numerical token embeddings are geometrically clustered (high pairwise cosine)
H2: Numerical tokens have lower baseline confidence (harder to move)
H3: The chosen-rejected gap for numbers is smaller than for words

Method:
1. Measure pairwise cosine similarity within numerical vs word token groups
2. Compare baseline confidence for numerical vs word DPO pairs
3. Measure the logit gap between chosen and rejected before/after DPO
4. Check if DPO hidden state movement correlates with token embedding distance

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

# Categorize DPO pairs as numerical vs word
DPO_PAIRS_CATEGORIZED = [
    # (prompt, chosen, rejected, category)
    ("The capital of Japan is", " Tokyo", " Osaka", "word"),
    ("The capital of France is", " Paris", " Lyon", "word"),
    ("The capital of Germany is", " Berlin", " Munich", "word"),
    ("The capital of Italy is", " Rome", " Milan", "word"),
    ("The capital of Spain is", " Madrid", " Barcelona", "word"),
    ("The largest planet in the solar system is", " Jupiter", " Saturn", "word"),
    ("The chemical symbol for gold is", " Au", " Ag", "word"),
    ("The author of Romeo and Juliet is", " William", " Charles", "word"),
    ("The first president of the United States was", " George", " John", "word"),
    ("The largest ocean on Earth is the", " Pacific", " Atlantic", "word"),
    ("The chemical formula for water is", " H", " O", "word"),
    # Numerical pairs
    ("Water freezes at", " 0", " 100", "number"),
    ("The boiling point of water is", " 100", " 212", "number"),
    ("The atomic number of carbon is", " 6", " 12", "number"),
    ("The speed of light is approximately", " 299", " 186", "number"),
]


def dpo_loss(model, ref_model, tok, prompt, chosen, rejected, beta=0.05):
    text_c = prompt + chosen
    text_r = prompt + rejected
    inp_c = tok(text_c, return_tensors='pt').to(DEVICE)
    inp_r = tok(text_r, return_tensors='pt').to(DEVICE)
    prompt_len = tok(prompt, return_tensors='pt')['input_ids'].shape[1]
    lc = model(**inp_c).logits[0, prompt_len-1:-1, :].float().clamp(-100, 100)
    lr_ = model(**inp_r).logits[0, prompt_len-1:-1, :].float().clamp(-100, 100)
    lp_c = F.log_softmax(lc, dim=-1).gather(1, inp_c['input_ids'][0, prompt_len:].unsqueeze(1)).squeeze()
    lp_r = F.log_softmax(lr_, dim=-1).gather(1, inp_r['input_ids'][0, prompt_len:].unsqueeze(1)).squeeze()
    if lp_c.dim() == 0: lp_c = lp_c.unsqueeze(0)
    if lp_r.dim() == 0: lp_r = lp_r.unsqueeze(0)
    with torch.no_grad():
        rlc = ref_model(**inp_c).logits[0, prompt_len-1:-1, :].float().clamp(-100, 100)
        rlr = ref_model(**inp_r).logits[0, prompt_len-1:-1, :].float().clamp(-100, 100)
        rlp_c = F.log_softmax(rlc, dim=-1).gather(1, inp_c['input_ids'][0, prompt_len:].unsqueeze(1)).squeeze()
        rlp_r = F.log_softmax(rlr, dim=-1).gather(1, inp_r['input_ids'][0, prompt_len:].unsqueeze(1)).squeeze()
        if rlp_c.dim() == 0: rlp_c = rlp_c.unsqueeze(0)
        if rlp_r.dim() == 0: rlp_r = rlp_r.unsqueeze(0)
    diff = beta * ((lp_c.sum() - rlp_c.sum()) - (lp_r.sum() - rlp_r.sum()))
    return -F.logsigmoid(diff)


def main():
    print("[P129] Why Do Numerical Tokens Resist DPO?")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ===== H1: Embedding Geometry =====
    print("\n  === H1: Token Embedding Geometry ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    embed = base_model.model.embed_tokens.weight.detach().cpu().float()

    # Collect token IDs by category
    word_chosen_ids = []
    word_rejected_ids = []
    num_chosen_ids = []
    num_rejected_ids = []

    for prompt, chosen, rejected, cat in DPO_PAIRS_CATEGORIZED:
        c_ids = tok.encode(chosen)
        r_ids = tok.encode(rejected)
        c_id = c_ids[-1] if c_ids else 0
        r_id = r_ids[-1] if r_ids else 0
        if cat == 'word':
            word_chosen_ids.append(c_id)
            word_rejected_ids.append(r_id)
        else:
            num_chosen_ids.append(c_id)
            num_rejected_ids.append(r_id)

    # Pairwise cosine similarity within groups
    def pairwise_cos(ids):
        if len(ids) < 2:
            return 0.0
        embs = embed[ids]
        cos_matrix = F.cosine_similarity(embs.unsqueeze(0), embs.unsqueeze(1), dim=-1)
        # Extract upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(len(ids), len(ids)), diagonal=1).bool()
        return cos_matrix[mask].mean().item()

    word_pair_cos = pairwise_cos(word_chosen_ids + word_rejected_ids)
    num_pair_cos = pairwise_cos(num_chosen_ids + num_rejected_ids)

    # Chosen-rejected cosine distance per pair
    word_cr_cos = []
    for prompt, chosen, rejected, cat in DPO_PAIRS_CATEGORIZED:
        if cat != 'word': continue
        c_id = tok.encode(chosen)[-1]
        r_id = tok.encode(rejected)[-1]
        cos = F.cosine_similarity(embed[c_id].unsqueeze(0), embed[r_id].unsqueeze(0)).item()
        word_cr_cos.append(cos)

    num_cr_cos = []
    for prompt, chosen, rejected, cat in DPO_PAIRS_CATEGORIZED:
        if cat != 'number': continue
        c_id = tok.encode(chosen)[-1]
        r_id = tok.encode(rejected)[-1]
        cos = F.cosine_similarity(embed[c_id].unsqueeze(0), embed[r_id].unsqueeze(0)).item()
        num_cr_cos.append(cos)

    print(f"    Word tokens pairwise cos: {word_pair_cos:.4f}")
    print(f"    Number tokens pairwise cos: {num_pair_cos:.4f}")
    print(f"    Word chosen-rejected cos: {np.mean(word_cr_cos):.4f} +/- {np.std(word_cr_cos):.4f}")
    print(f"    Number chosen-rejected cos: {np.mean(num_cr_cos):.4f} +/- {np.std(num_cr_cos):.4f}")

    h1_result = {
        'word_pairwise_cos': word_pair_cos,
        'num_pairwise_cos': num_pair_cos,
        'word_cr_cos_mean': float(np.mean(word_cr_cos)),
        'num_cr_cos_mean': float(np.mean(num_cr_cos)),
        'h1_confirmed': num_pair_cos > word_pair_cos,  # Numbers more clustered?
    }
    print(f"    H1 {'CONFIRMED' if h1_result['h1_confirmed'] else 'REJECTED'}: "
          f"numbers {'more' if h1_result['h1_confirmed'] else 'less'} clustered")

    # ===== H2: Baseline Confidence =====
    print("\n  === H2: Baseline Confidence ===")
    word_confs = []
    num_confs = []
    word_gaps = []
    num_gaps = []

    for prompt, chosen, rejected, cat in DPO_PAIRS_CATEGORIZED:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        c_id = tok.encode(chosen)[-1]
        r_id = tok.encode(rejected)[-1]
        with torch.no_grad():
            logits = base_model(**inp).logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        conf = probs[c_id].item()
        gap = (probs[c_id] - probs[r_id]).item()

        if cat == 'word':
            word_confs.append(conf)
            word_gaps.append(gap)
        else:
            num_confs.append(conf)
            num_gaps.append(gap)

    print(f"    Word baseline confidence: {np.mean(word_confs):.4f} +/- {np.std(word_confs):.4f}")
    print(f"    Number baseline confidence: {np.mean(num_confs):.4f} +/- {np.std(num_confs):.4f}")
    print(f"    Word chosen-rejected gap: {np.mean(word_gaps):.4f} +/- {np.std(word_gaps):.4f}")
    print(f"    Number chosen-rejected gap: {np.mean(num_gaps):.4f} +/- {np.std(num_gaps):.4f}")

    h2_result = {
        'word_conf_mean': float(np.mean(word_confs)),
        'num_conf_mean': float(np.mean(num_confs)),
        'word_gap_mean': float(np.mean(word_gaps)),
        'num_gap_mean': float(np.mean(num_gaps)),
        'h2_confirmed': np.mean(num_confs) < np.mean(word_confs),
    }
    print(f"    H2 {'CONFIRMED' if h2_result['h2_confirmed'] else 'REJECTED'}: "
          f"numbers have {'lower' if h2_result['h2_confirmed'] else 'higher'} baseline confidence")

    del base_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ===== H3: DPO Effect Size by Category =====
    print("\n  === H3: DPO Effect Size ===")
    dpo_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    base_model2 = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

    dpo_model.train()
    for name, param in dpo_model.named_parameters():
        param.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name:
                param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [p for p in dpo_model.parameters() if p.requires_grad], lr=5e-6)
    for epoch in range(5):
        for prompt, chosen, rejected, _ in DPO_PAIRS_CATEGORIZED:
            loss = dpo_loss(dpo_model, ref_model, tok, prompt, chosen, rejected)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in dpo_model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
    dpo_model.eval()

    word_effects = []
    num_effects = []
    for prompt, chosen, rejected, cat in DPO_PAIRS_CATEGORIZED:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        c_id = tok.encode(chosen)[-1]
        r_id = tok.encode(rejected)[-1]
        with torch.no_grad():
            base_logits = base_model2(**inp).logits[0, -1, :].float()
            dpo_logits = dpo_model(**inp).logits[0, -1, :].float()
        base_probs = torch.softmax(base_logits, dim=-1)
        dpo_probs = torch.softmax(dpo_logits, dim=-1)

        chosen_change = (dpo_probs[c_id] - base_probs[c_id]).item()
        rejected_change = (dpo_probs[r_id] - base_probs[r_id]).item()
        effect = chosen_change - rejected_change  # positive = DPO working correctly

        def safe_tok(s):
            return s.encode('ascii', 'replace').decode()

        if cat == 'word':
            word_effects.append(effect)
        else:
            num_effects.append(effect)
        print(f"    [{cat:6s}] {prompt[:30]:30s} "
              f"chosen_delta={chosen_change:+.4f} "
              f"rejected_delta={rejected_change:+.4f} "
              f"effect={effect:+.4f}")

    print(f"\n    Word mean DPO effect: {np.mean(word_effects):+.4f} +/- {np.std(word_effects):.4f}")
    print(f"    Number mean DPO effect: {np.mean(num_effects):+.4f} +/- {np.std(num_effects):.4f}")
    print(f"    Ratio: {np.mean(word_effects)/np.mean(num_effects):.1f}x" if np.mean(num_effects) != 0
          else "    Ratio: inf")

    h3_result = {
        'word_effect_mean': float(np.mean(word_effects)),
        'num_effect_mean': float(np.mean(num_effects)),
        'ratio': float(np.mean(word_effects) / np.mean(num_effects)) if np.mean(num_effects) != 0 else float('inf'),
        'h3_confirmed': abs(np.mean(num_effects)) < abs(np.mean(word_effects)),
    }
    print(f"    H3 {'CONFIRMED' if h3_result['h3_confirmed'] else 'REJECTED'}: "
          f"DPO effect {'weaker' if h3_result['h3_confirmed'] else 'stronger'} on numbers")

    del dpo_model, ref_model, base_model2, optimizer; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save
    out = {
        'phase': '129', 'name': 'Numerical Token Resistance',
        'h1_embedding_geometry': h1_result,
        'h2_baseline_confidence': h2_result,
        'h3_dpo_effect_size': h3_result,
    }
    with open(os.path.join(RESULTS_DIR, 'phase129_numerical.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # H1: Embedding similarity
    ax = axes[0]
    cats = ['Word\npairwise', 'Number\npairwise', 'Word\nchosen-rej', 'Number\nchosen-rej']
    vals = [word_pair_cos, num_pair_cos, np.mean(word_cr_cos), np.mean(num_cr_cos)]
    colors = ['#3498db', '#e74c3c', '#3498db', '#e74c3c']
    bars = ax.bar(cats, vals, color=colors, alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
               f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('H1: Token Embedding Geometry', fontweight='bold')
    ax.set_ylim(0, max(vals)*1.3)

    # H2: Baseline confidence
    ax = axes[1]
    cats2 = ['Word\nconfidence', 'Number\nconfidence', 'Word\ngap', 'Number\ngap']
    vals2 = [np.mean(word_confs), np.mean(num_confs), np.mean(word_gaps), np.mean(num_gaps)]
    bars = ax.bar(cats2, vals2, color=['#3498db', '#e74c3c', '#3498db', '#e74c3c'], alpha=0.8)
    for bar, val in zip(bars, vals2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
               f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
    ax.set_ylabel('Probability')
    ax.set_title('H2: Baseline Confidence', fontweight='bold')

    # H3: DPO effect
    ax = axes[2]
    cats3 = ['Word\nDPO effect', 'Number\nDPO effect']
    vals3 = [np.mean(word_effects), np.mean(num_effects)]
    colors3 = ['#2ecc71' if v > 0 else '#e74c3c' for v in vals3]
    bars = ax.bar(cats3, vals3, color=colors3, alpha=0.8)
    for bar, val in zip(bars, vals3):
        y = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.015
        ax.text(bar.get_x()+bar.get_width()/2, y,
               f'{val:+.4f}', ha='center', fontweight='bold', fontsize=12)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_ylabel('DPO Effect (chosen_delta - rejected_delta)')
    ax.set_title('H3: DPO Effect Size', fontweight='bold')

    fig.suptitle('Phase 129: Why Do Numerical Tokens Resist DPO?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase129_numerical.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    confirmed = [h for h, r in [('H1', h1_result), ('H2', h2_result), ('H3', h3_result)]
                 if r.get(f'{h.lower()}_confirmed')]
    print(f"    Confirmed: {', '.join(confirmed) if confirmed else 'NONE'}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 129] Complete.")

if __name__ == '__main__':
    main()
