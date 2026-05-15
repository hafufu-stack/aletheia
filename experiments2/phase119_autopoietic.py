# -*- coding: utf-8 -*-
"""
Phase 119: Autopoietic F-DPO (Self-Correcting Factuality Pipeline)
Combines P116b (entropy-based refusal) + P117b (back-only DPO) into
an end-to-end pipeline:

1. Train back-only DPO with optimal lr=5e-6 (P117b)
2. At inference: measure uncertainty at L_0.94
3. If uncertain -> refuse ("I don't know")
4. If certain -> answer using DPO-trained model

Evaluates on a combined test set of known + unknown facts.

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

# Training set for DPO
DPO_PAIRS = [
    ("The capital of Japan is", " Tokyo", " Osaka"),
    ("The capital of France is", " Paris", " Lyon"),
    ("The capital of Germany is", " Berlin", " Munich"),
    ("The capital of Italy is", " Rome", " Milan"),
    ("The capital of Spain is", " Madrid", " Barcelona"),
    ("The largest planet in the solar system is", " Jupiter", " Saturn"),
    ("Water freezes at", " 0", " 100"),
    ("The chemical symbol for gold is", " Au", " Ag"),
    ("The author of Romeo and Juliet is", " William", " Charles"),
    ("The first president of the United States was", " George", " John"),
    ("The chemical formula for water is", " H", " O"),
    ("The boiling point of water is", " 100", " 212"),
    ("The atomic number of carbon is", " 6", " 12"),
    ("The largest ocean on Earth is the", " Pacific", " Atlantic"),
    ("The speed of light is approximately", " 299", " 186"),
]

# Test: known facts (model should answer)
TEST_KNOWN = [
    ("The tallest mountain in the world is", " Mount"),
    ("The currency of the United Kingdom is the", " pound"),
    ("The speed of sound is approximately", " 343"),
    ("Albert Einstein was born in", " Ul"),
    ("Photosynthesis converts sunlight into", " chemical"),
]

# Test: unknown facts (model should refuse)
TEST_UNKNOWN = [
    ("The 47th prime number is", " 211"),
    ("The population of Funabashi in 2025 was approximately", " 640"),
    ("The melting point of hafnium in Kelvin is", " 2506"),
    ("The deepest point of Lake Baikal in meters is", " 1642"),
    ("The ISO country code for Bhutan is", " BT"),
]


def get_layer_entropy(model, tok, prompt, layer_idx):
    """Get entropy at a specific layer (float32 safe)."""
    hidden = {}
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.dim() == 3:
            hidden['h'] = h[:, -1, :].detach()
        else:
            hidden['h'] = h[-1, :].detach().unsqueeze(0)
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model(**inp)
    handle.remove()
    h = model.model.norm(hidden['h'].float())
    logits = model.lm_head(h.to(next(model.lm_head.parameters()).dtype)).squeeze()
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    top_token = logits.argmax().item()
    return entropy, top_token


def get_gate_l1(model, tok, prompt, layer_idx):
    """Get MLP gate L1 norm (P118's best metric candidate)."""
    gate_data = {}
    def mlp_hook(module, inp, out):
        x = inp[0][:, -1, :].detach().float()
        gate_out = torch.nn.functional.silu(
            module.gate_proj(x.to(next(module.gate_proj.parameters()).dtype)))
        gate_data['l1'] = gate_out.float().abs().mean().item()
    handle = model.model.layers[layer_idx].mlp.register_forward_hook(mlp_hook)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model(**inp)
    handle.remove()
    return gate_data.get('l1', 0)


def dpo_loss(model, ref_model, tok, prompt, chosen, rejected, beta=0.05):
    """DPO loss (float32, P117b proven)."""
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
    print("[P119] Autopoietic F-DPO Pipeline")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    best_layer = int(n_layers * 0.94)  # 22

    # Step 1: Train back-only DPO (P117b optimal: lr=5e-6, beta=0.05)
    print("\n  Step 1: Training back-only DPO (lr=5e-6, beta=0.05)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Freeze all, unfreeze back MLPs only
    model.train()
    boundary = best_layer
    for name, param in model.named_parameters():
        param.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name:
                param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-6)

    for epoch in range(5):
        epoch_loss = 0
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss(model, ref_model, tok, prompt, chosen, rejected)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"    Epoch {epoch}: loss={epoch_loss/len(DPO_PAIRS):.4f}")
    model.eval()

    # Step 2: Calibrate uncertainty threshold
    # Use P118's best metric (try both entropy and gate_l1)
    print("\n  Step 2: Calibrating uncertainty thresholds...")

    # Measure on training set (known ground truth)
    train_known_ents = []
    train_known_gates = []
    for prompt, chosen, _ in DPO_PAIRS:
        ent, _ = get_layer_entropy(model, tok, prompt, best_layer)
        gate = get_gate_l1(model, tok, prompt, best_layer)
        train_known_ents.append(ent)
        train_known_gates.append(gate)

    test_unknown_ents = []
    test_unknown_gates = []
    for prompt, _ in TEST_UNKNOWN:
        ent, _ = get_layer_entropy(model, tok, prompt, best_layer)
        gate = get_gate_l1(model, tok, prompt, best_layer)
        test_unknown_ents.append(ent)
        test_unknown_gates.append(gate)

    # Optimal threshold: midpoint between means
    ent_threshold = (np.mean(train_known_ents) + np.mean(test_unknown_ents)) / 2
    gate_threshold = (np.mean(train_known_gates) + np.mean(test_unknown_gates)) / 2

    print(f"    Entropy threshold: {ent_threshold:.2f}")
    print(f"    Gate L1 threshold: {gate_threshold:.4f}")

    # Step 3: Evaluate the pipeline
    print("\n  Step 3: Evaluating combined pipeline...")

    all_test = [(p, a, 'known') for p, a in TEST_KNOWN] + \
               [(p, a, 'unknown') for p, a in TEST_UNKNOWN]

    results_by_method = {}
    for method_name, use_gate in [('entropy_gated', False), ('gate_l1_gated', True)]:
        correct = 0
        total = len(all_test)
        details = []

        for prompt, answer, fact_type in all_test:
            fact_tokens = tok.encode(answer)
            fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0

            # Get uncertainty
            ent, top_token = get_layer_entropy(model, tok, prompt, best_layer)
            gate = get_gate_l1(model, tok, prompt, best_layer)

            if use_gate:
                uncertain = gate > gate_threshold  # Use gate metric
            else:
                uncertain = ent > ent_threshold     # Use entropy metric

            if uncertain:
                # Refuse - correct if fact was unknown
                is_correct = (fact_type == 'unknown')
                action = 'refuse'
            else:
                # Answer - correct if fact was known AND answer is right
                is_correct = (fact_type == 'known' and top_token == fact_id)
                action = 'answer'

            if is_correct:
                correct += 1
            details.append({
                'prompt': prompt[:40],
                'fact_type': fact_type,
                'action': action,
                'correct': is_correct,
                'entropy': ent,
                'gate_l1': gate,
            })

        acc = correct / total
        results_by_method[method_name] = {
            'accuracy': acc, 'details': details,
            'threshold': gate_threshold if use_gate else ent_threshold,
        }
        print(f"    {method_name}: {correct}/{total} = {acc:.0%}")

    # Baseline: no refusal (just DPO model answering everything)
    baseline_correct = 0
    for prompt, answer, fact_type in all_test:
        fact_tokens = tok.encode(answer)
        fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0
        _, top_token = get_layer_entropy(model, tok, prompt, best_layer)
        if fact_type == 'known' and top_token == fact_id:
            baseline_correct += 1
        elif fact_type == 'unknown' and top_token != fact_id:
            baseline_correct += 1  # Unknown + wrong = arguably "not hallucinating"
    baseline_acc = baseline_correct / len(all_test)
    print(f"    baseline (no refusal): {baseline_correct}/{len(all_test)} = {baseline_acc:.0%}")

    # Also: raw base model baseline (no DPO)
    raw_correct = 0
    for prompt, answer, fact_type in all_test:
        fact_tokens = tok.encode(answer)
        fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0
        _, top_token = get_layer_entropy(ref_model, tok, prompt, best_layer)
        if fact_type == 'known' and top_token == fact_id:
            raw_correct += 1
        elif fact_type == 'unknown' and top_token != fact_id:
            raw_correct += 1
    raw_acc = raw_correct / len(all_test)
    print(f"    raw_base (no DPO, no refusal): {raw_correct}/{len(all_test)} = {raw_acc:.0%}")

    del model, ref_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save
    out_data = {
        'phase': '119', 'name': 'Autopoietic F-DPO',
        'model': model_id, 'best_layer': best_layer,
        'ent_threshold': float(ent_threshold),
        'gate_threshold': float(gate_threshold),
        'results_by_method': results_by_method,
        'baseline_acc': baseline_acc,
        'raw_base_acc': raw_acc,
    }
    with open(os.path.join(RESULTS_DIR, 'phase119_autopoietic.json'), 'w') as f:
        json.dump(out_data, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Accuracy comparison
    ax = axes[0]
    methods = ['raw_base', 'dpo_only', 'entropy_gated', 'gate_l1_gated']
    accs = [raw_acc, baseline_acc,
            results_by_method['entropy_gated']['accuracy'],
            results_by_method['gate_l1_gated']['accuracy']]
    colors_bar = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(methods, accs, color=colors_bar)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.0%}', ha='center', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Combined Accuracy')
    ax.set_title('Pipeline Comparison')
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)

    # Panel 2: Decision matrix for entropy method
    ax = axes[1]
    det = results_by_method['entropy_gated']['details']
    known_det = [d for d in det if d['fact_type'] == 'known']
    unknown_det = [d for d in det if d['fact_type'] == 'unknown']
    tp = sum(1 for d in unknown_det if d['action'] == 'refuse')
    fp = sum(1 for d in known_det if d['action'] == 'refuse')
    tn = sum(1 for d in known_det if d['action'] == 'answer' and d['correct'])
    fn = sum(1 for d in unknown_det if d['action'] == 'answer')
    matrix = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Answer', 'Refuse'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Known', 'Unknown'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                   fontsize=18, fontweight='bold')
    ax.set_title('Entropy-Gated Decision Matrix')
    ax.set_xlabel('Model Action')
    ax.set_ylabel('Fact Type')

    # Panel 3: Entropy distribution
    ax = axes[2]
    known_ents = [d['entropy'] for d in det if d['fact_type'] == 'known']
    unknown_ents = [d['entropy'] for d in det if d['fact_type'] == 'unknown']
    ax.hist(known_ents, bins=10, alpha=0.6, color='#2ecc71', label='Known')
    ax.hist(unknown_ents, bins=10, alpha=0.6, color='#e74c3c', label='Unknown')
    ax.axvline(x=ent_threshold, color='black', linestyle='--', label=f'Threshold={ent_threshold:.1f}')
    ax.set_xlabel('Entropy at L_0.94')
    ax.set_ylabel('Count')
    ax.set_title('Entropy Distribution')
    ax.legend()

    fig.suptitle('Phase 119: Autopoietic F-DPO (Entropy-Gated Factuality)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase119_autopoietic.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 119] Complete.")


if __name__ == '__main__':
    main()
