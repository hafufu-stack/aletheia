# -*- coding: utf-8 -*-
"""
Phase 141: Multi-Token Rigid-Body Surgery
Can we handle multi-token numbers (e.g. "1538" -> ["15","38"])?

Problem: BPE splits numbers into sub-tokens. Naive dispersion
breaks the Markov transition probabilities between sub-tokens.

Solution: Treat multi-token numbers as "rigid bodies" - maintain
relative distances/angles between sub-tokens while translating
the entire group away from the cluster center.

Test: Facts with multi-token numerical answers:
  "A year has 365 days" -> ["3","65"] or ["36","5"] depending on tokenizer
  "The speed of light is 299792" -> multiple tokens

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

# Multi-token numerical facts for evaluation
# Format: (prompt, full_answer_text, category)
MULTI_TOKEN_FACTS = [
    ("A year has", " 365", "multi"),
    ("The speed of light is approximately", " 299", "multi"),
    ("Water boils at", " 100", "single"),  # control: single-token
    ("Water freezes at", " 0", "single"),   # control: single-token
    ("The atomic number of carbon is", " 6", "single"),
    ("The number of days in February is", " 28", "multi"),
    ("The freezing point of water in Fahrenheit is", " 32", "multi"),
    ("The boiling point of water in Fahrenheit is", " 212", "multi"),
]

# DPO pairs (mix of single and multi-token)
DPO_PAIRS = [
    ("Water freezes at", " 0", " 100"),
    ("The boiling point of water is", " 100", " 212"),
    ("The atomic number of carbon is", " 6", " 12"),
    ("A year has", " 365", " 400"),
    ("The freezing point of water in Fahrenheit is", " 32", " 72"),
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


def naive_disperse(model, tok, strength=1.0):
    """Naive dispersion: push each number token away from center independently."""
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366",
                  " 28", " 32", " 400", " 72"]
    embed = model.model.embed_tokens.weight.data
    ids = []
    for t in num_tokens:
        token_ids = tok.encode(t)
        ids.append(token_ids[-1])
    ids = list(set(ids))  # deduplicate
    vecs = embed[ids].clone()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()


def rigid_body_disperse(model, tok, strength=1.0):
    """
    Rigid-body dispersion: identify multi-token number groups,
    preserve internal geometry, translate as a unit.
    
    Groups: tokens that form multi-digit numbers (e.g. "365" might be "3"+"65")
    Strategy: 
    1. Find all number-related token IDs
    2. Group tokens that appear in the same multi-token number
    3. Compute group centroid
    4. Translate entire group as rigid body (preserving internal distances)
    """
    # All number-related tokens
    all_num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                      " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366",
                      " 28", " 32", " 400", " 72"]
    
    embed = model.model.embed_tokens.weight.data
    
    # Get all unique token IDs
    all_ids = set()
    # Also find multi-token groups by encoding multi-digit numbers
    multi_digit_numbers = ["365", "299", "212", "186", "100", "366", "400"]
    groups = {}  # group_name -> list of token IDs
    
    for num_str in multi_digit_numbers:
        token_ids = tok.encode(" " + num_str)
        # Remove BOS if present
        if token_ids[0] == tok.bos_token_id:
            token_ids = token_ids[1:]
        groups[num_str] = token_ids
        all_ids.update(token_ids)
    
    # Single-digit tokens
    for d in range(10):
        token_ids = tok.encode(f" {d}")
        tid = token_ids[-1]
        if tid not in all_ids:
            groups[str(d)] = [tid]
            all_ids.add(tid)
    
    # Also add 28, 32, 72
    for num_str in ["28", "32", "72"]:
        token_ids = tok.encode(" " + num_str)
        if token_ids[0] == tok.bos_token_id:
            token_ids = token_ids[1:]
        groups[num_str] = token_ids
        all_ids.update(token_ids)
    
    all_ids = list(all_ids)
    vecs = embed[all_ids].clone().float()
    global_center = vecs.mean(dim=0)
    
    # For each group, translate as rigid body
    for group_name, group_ids in groups.items():
        if len(group_ids) == 1:
            # Single token: standard dispersion
            idx = group_ids[0]
            vec = embed[idx].clone().float()
            diff = vec - global_center
            direction = diff / (diff.norm() + 1e-8)
            embed[idx] += (strength * direction * vec.norm()).to(embed.dtype)
        else:
            # Multi-token: compute group centroid, translate entire group
            group_vecs = embed[group_ids].clone().float()
            group_centroid = group_vecs.mean(dim=0)
            
            # Direction: from global center to group centroid
            diff = group_centroid - global_center
            direction = diff / (diff.norm() + 1e-8)
            
            # Translation vector (same for all tokens in group)
            translation = strength * direction * group_centroid.norm()
            
            # Apply same translation to all tokens in group (rigid body)
            for idx in group_ids:
                embed[idx] += translation.to(embed.dtype)
    
    return groups


def measure_group_coherence(model, tok, groups):
    """Measure how well multi-token groups maintained internal structure."""
    embed = model.model.embed_tokens.weight.data
    coherence = {}
    for name, ids in groups.items():
        if len(ids) > 1:
            vecs = embed[ids].float()
            # Internal L2 distances
            dists = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0)).squeeze(0)
            mask = ~torch.eye(len(ids), dtype=bool, device=dists.device)
            if mask.sum() > 0:
                coherence[name] = dists[mask].mean().item()
    return coherence


def evaluate(model, tok, facts, code_mode=False):
    """Evaluate on facts, handling Qwen's space-then-digit pattern.
    
    Qwen tokenizes ' 365' as [space, 3, 6, 5].
    The model first predicts 'space', then '3', then '6', then '5'.
    We check if the model predicts the correct FIRST DIGIT after generating 2 tokens.
    """
    results = []
    space_id = tok.encode(" ")[-1]
    
    for prompt, expected, cat in facts:
        text = f"# {prompt}" if code_mode else prompt
        exp_ids = tok.encode(expected)
        # Find first digit token (skip space tokens)
        digit_ids = [i for i in exp_ids if i != space_id]
        first_digit_id = digit_ids[0] if digit_ids else exp_ids[-1]

        inp = tok(text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        pred_id = logits.argmax().item()
        
        if pred_id == space_id:
            # Model predicted space first (correct start for " 365")
            # Generate one more token to get the actual digit
            inp2_ids = torch.cat([inp['input_ids'], torch.tensor([[pred_id]], device=DEVICE)], dim=1)
            with torch.no_grad():
                logits2 = model(input_ids=inp2_ids).logits[0, -1, :].float()
            pred_digit = logits2.argmax().item()
            correct = (pred_digit == first_digit_id)
            pred_text = tok.decode([pred_id, pred_digit])
        else:
            # Model predicted a non-space token directly
            correct = (pred_id == first_digit_id)
            pred_text = tok.decode([pred_id])
        
        results.append({'cat': cat, 'correct': int(correct), 'expected': expected,
                        'first_digit': tok.decode([first_digit_id]),
                        'pred': pred_text})
    single_c = sum(r['correct'] for r in results if r['cat'] == 'single')
    single_t = max(1, sum(1 for r in results if r['cat'] == 'single'))
    multi_c = sum(r['correct'] for r in results if r['cat'] == 'multi')
    multi_t = max(1, sum(1 for r in results if r['cat'] == 'multi'))
    return single_c / single_t, multi_c / multi_t, results


def main():
    print("[P141] Multi-Token Rigid-Body Surgery")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    n_layers = 24
    boundary = int(n_layers * 0.94)

    # First: show how tokenizer splits multi-digit numbers
    print("\n  Tokenizer analysis:")
    for num in ["365", "299", "212", "100", "28", "32"]:
        ids = tok.encode(" " + num)
        tokens = [tok.decode([i]) for i in ids]
        print(f"    '{num}' -> {ids} = {tokens}")

    configs = {}

    # A: Baseline
    print("\n  === A: Baseline ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    s_acc, m_acc, details = evaluate(model, tok, MULTI_TOKEN_FACTS)
    print(f"    Single-token: {s_acc:.0%}, Multi-token: {m_acc:.0%}")
    for d in details:
        print(f"      {d['expected']:6s} -> pred='{d['pred']}' {'OK' if d['correct'] else 'MISS'}")
    configs['A_baseline'] = {'single': s_acc, 'multi': m_acc}
    del model; gc.collect(); torch.cuda.empty_cache()

    # B: Naive Dispersion + DPO
    print("\n  === B: Naive Dispersion + DPO ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    naive_disperse(model, tok, strength=1.0)
    naive_disperse(ref, tok, strength=1.0)
    # DPO
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-6)
    for epoch in range(5):
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss(model, ref, tok, prompt, chosen, rejected)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
    model.eval()
    s_acc, m_acc, details = evaluate(model, tok, MULTI_TOKEN_FACTS)
    print(f"    Single-token: {s_acc:.0%}, Multi-token: {m_acc:.0%}")
    for d in details:
        print(f"      {d['expected']:6s} -> pred='{d['pred']}' {'OK' if d['correct'] else 'MISS'}")
    configs['B_naive_dpo'] = {'single': s_acc, 'multi': m_acc}
    del model, ref; gc.collect(); torch.cuda.empty_cache()

    # C: Rigid-Body Dispersion + DPO
    print("\n  === C: Rigid-Body Dispersion + DPO ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    
    # Measure pre-surgery group coherence
    pre_groups = rigid_body_disperse(model, tok, strength=0.0)  # dummy to get groups
    # Reload and do actual surgery
    del model; gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    pre_coherence = measure_group_coherence(model, tok, pre_groups)
    
    groups = rigid_body_disperse(model, tok, strength=1.0)
    rigid_body_disperse(ref, tok, strength=1.0)
    
    post_coherence = measure_group_coherence(model, tok, groups)
    print("    Group coherence (internal L2):")
    for name in pre_coherence:
        pre = pre_coherence[name]
        post = post_coherence.get(name, 0)
        change = (post - pre) / (pre + 1e-8) * 100
        print(f"      {name}: {pre:.4f} -> {post:.4f} ({change:+.1f}%)")
    
    # DPO
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-6)
    for epoch in range(5):
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss(model, ref, tok, prompt, chosen, rejected)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
    model.eval()
    s_acc, m_acc, details = evaluate(model, tok, MULTI_TOKEN_FACTS)
    print(f"    Single-token: {s_acc:.0%}, Multi-token: {m_acc:.0%}")
    for d in details:
        print(f"      {d['expected']:6s} -> pred='{d['pred']}' {'OK' if d['correct'] else 'MISS'}")
    configs['C_rigid_dpo'] = {'single': s_acc, 'multi': m_acc}

    # D: Rigid-Body + DPO + Shield+Sword (Ultimate Combo for multi-token)
    print("\n  === D: Rigid-Body + DPO + Shield+Sword ===")
    s_acc, m_acc, details = evaluate(model, tok, MULTI_TOKEN_FACTS, code_mode=True)
    print(f"    Single-token: {s_acc:.0%}, Multi-token: {m_acc:.0%}")
    for d in details:
        print(f"      {d['expected']:6s} -> pred='{d['pred']}' {'OK' if d['correct'] else 'MISS'}")
    configs['D_rigid_ultimate'] = {'single': s_acc, 'multi': m_acc}
    del model, ref; gc.collect(); torch.cuda.empty_cache()

    # Save
    result_data = {
        'phase': '141', 'name': 'Multi-Token Rigid-Body Surgery',
        'configs': configs,
        'pre_coherence': pre_coherence,
        'post_coherence': post_coherence,
    }
    with open(os.path.join(RESULTS_DIR, 'phase141_rigid_body.json'), 'w') as f:
        json.dump(result_data, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = ['Baseline', 'Naive\nDispersion\n+DPO', 'Rigid-Body\nDispersion\n+DPO', 'Rigid-Body\n+DPO\n+Shield&Sword']
    keys = ['A_baseline', 'B_naive_dpo', 'C_rigid_dpo', 'D_rigid_ultimate']
    single_vals = [configs[k]['single'] for k in keys]
    multi_vals = [configs[k]['multi'] for k in keys]
    x = np.arange(len(names))
    w = 0.35

    # Left: Single vs Multi accuracy
    ax = axes[0]
    ax.bar(x-w/2, single_vals, w, label='Single-token nums', color='#3498db', alpha=0.8)
    ax.bar(x+w/2, multi_vals, w, label='Multi-token nums', color='#e74c3c', alpha=0.8)
    for i in range(len(names)):
        ax.text(x[i]-w/2, single_vals[i]+0.02, f'{single_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i]+w/2, multi_vals[i]+0.02, f'{multi_vals[i]:.0%}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Single vs Multi-Token Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.3)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Group coherence comparison
    ax = axes[1]
    if pre_coherence and post_coherence:
        group_names = list(pre_coherence.keys())
        pre_vals = [pre_coherence[g] for g in group_names]
        post_vals = [post_coherence.get(g, 0) for g in group_names]
        x2 = np.arange(len(group_names))
        ax.bar(x2-w/2, pre_vals, w, label='Pre-surgery', color='#95a5a6', alpha=0.8)
        ax.bar(x2+w/2, post_vals, w, label='Post-surgery (rigid)', color='#2ecc71', alpha=0.8)
        ax.set_xticks(x2); ax.set_xticklabels(group_names, fontsize=10)
        ax.set_ylabel('Internal L2 Distance', fontsize=12)
        ax.set_title('Group Coherence Preservation', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No multi-token groups found', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)

    plt.suptitle('Phase 141: Multi-Token Rigid-Body Surgery',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase141_rigid_body.png'), dpi=150,
               bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    naive_m = configs['B_naive_dpo']['multi']
    rigid_m = configs['C_rigid_dpo']['multi']
    rigid_ult = configs['D_rigid_ultimate']['multi']
    if rigid_m > naive_m:
        print(f"  -> RIGID-BODY SUPERIOR: multi-token {naive_m:.0%} -> {rigid_m:.0%}")
    elif rigid_m == naive_m:
        print(f"  -> No difference: naive={naive_m:.0%}, rigid={rigid_m:.0%}")
    else:
        print(f"  -> Naive is better: naive={naive_m:.0%} > rigid={rigid_m:.0%}")
    if rigid_ult > rigid_m:
        print(f"  -> Shield+Sword further improves: {rigid_m:.0%} -> {rigid_ult:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 141] Complete.")

if __name__ == '__main__':
    main()
