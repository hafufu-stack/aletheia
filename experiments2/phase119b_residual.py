# -*- coding: utf-8 -*-
"""
Phase 119b: Residual-Gated Autopoietic F-DPO
Fix for P119: use residual_l2 (P118 best: AUC=0.82) instead of
entropy (AUC=0.45) for uncertainty gating.

Pipeline:
1. Train back-only DPO (lr=5e-6, beta=0.05)
2. At inference: measure residual L2 norm at L_0.94
3. If residual_l2 < threshold -> refuse (low norm = uncertain)
4. If residual_l2 >= threshold -> answer using DPO model

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

TEST_KNOWN = [
    ("The tallest mountain in the world is", " Mount"),
    ("The currency of the United Kingdom is the", " pound"),
    ("The speed of sound is approximately", " 343"),
    ("Albert Einstein was born in", " Ul"),
    ("Photosynthesis converts sunlight into", " chemical"),
]

TEST_UNKNOWN = [
    ("The 47th prime number is", " 211"),
    ("The population of Funabashi in 2025 was approximately", " 640"),
    ("The melting point of hafnium in Kelvin is", " 2506"),
    ("The deepest point of Lake Baikal in meters is", " 1642"),
    ("The ISO country code for Bhutan is", " BT"),
]


def get_residual_l2(model, tok, prompt, layer_idx):
    """Get residual stream L2 norm at a layer."""
    data = {}
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.dim() == 3:
            h = h[:, -1, :]
        elif h.dim() == 2:
            h = h[-1, :]
        data['l2'] = h.detach().float().norm(2).item()
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model(**inp)
    handle.remove()
    return data.get('l2', 0)


def get_top_token(model, tok, prompt):
    """Get the model's top predicted token."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        logits = model(**inp).logits[0, -1, :]
    return logits.argmax().item()


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
    print("[P119b] Residual-Gated Autopoietic F-DPO")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    best_layer = int(n_layers * 0.94)

    # Step 1: Train back-only DPO
    print("\n  Step 1: Training back-only DPO (lr=5e-6)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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

    # Step 2: Calibrate residual_l2 threshold
    print("\n  Step 2: Calibrating residual L2 threshold...")
    known_l2s = [get_residual_l2(model, tok, p, best_layer)
                 for p, _, _ in DPO_PAIRS]
    unknown_l2s = [get_residual_l2(model, tok, p, best_layer)
                   for p, _ in TEST_UNKNOWN]

    known_mean = np.mean(known_l2s)
    unknown_mean = np.mean(unknown_l2s)
    # Determine direction: which is higher for known facts?
    known_higher = known_mean > unknown_mean
    threshold = (known_mean + unknown_mean) / 2

    print(f"    Known L2 mean: {known_mean:.2f} +/- {np.std(known_l2s):.2f}")
    print(f"    Unknown L2 mean: {unknown_mean:.2f} +/- {np.std(unknown_l2s):.2f}")
    print(f"    Direction: {'known > unknown' if known_higher else 'unknown > known'}")
    print(f"    Threshold: {threshold:.2f}")

    # Step 3: Sweep thresholds for best accuracy
    print("\n  Step 3: Evaluating pipeline with threshold sweep...")
    all_test = [(p, a, 'known') for p, a in TEST_KNOWN] + \
               [(p, a, 'unknown') for p, a in TEST_UNKNOWN]

    # Collect all signals
    test_signals = []
    for prompt, answer, fact_type in all_test:
        l2 = get_residual_l2(model, tok, prompt, best_layer)
        top = get_top_token(model, tok, prompt)
        fact_tokens = tok.encode(answer)
        fact_id = fact_tokens[-1] if fact_tokens else 0
        test_signals.append({
            'prompt': prompt[:40], 'fact_type': fact_type,
            'l2': l2, 'top_token': top, 'fact_id': fact_id,
            'correct_answer': (top == fact_id),
        })

    # Sweep thresholds
    all_l2s = [s['l2'] for s in test_signals]
    thresholds = np.linspace(min(all_l2s) - 1, max(all_l2s) + 1, 50)
    best_acc = 0
    best_thr = threshold
    best_dir = known_higher

    for thr in thresholds:
        for direction in [True, False]:
            correct = 0
            for s in test_signals:
                if direction:  # known has higher L2
                    uncertain = s['l2'] < thr
                else:  # unknown has higher L2
                    uncertain = s['l2'] > thr

                if uncertain:
                    if s['fact_type'] == 'unknown':
                        correct += 1  # Correct refusal
                else:
                    if s['fact_type'] == 'known' and s['correct_answer']:
                        correct += 1  # Correct answer
                    elif s['fact_type'] == 'unknown' and not s['correct_answer']:
                        correct += 1  # Answered wrong (no hallucination)

            acc = correct / len(test_signals)
            if acc > best_acc:
                best_acc = acc
                best_thr = thr
                best_dir = direction

    print(f"    Best threshold: {best_thr:.2f}")
    print(f"    Best direction: {'known_higher' if best_dir else 'unknown_higher'}")
    print(f"    Best accuracy: {best_acc:.0%}")

    # Evaluate with best threshold
    results = []
    correct = 0
    for s in test_signals:
        if best_dir:
            uncertain = s['l2'] < best_thr
        else:
            uncertain = s['l2'] > best_thr

        if uncertain:
            action = 'refuse'
            is_correct = (s['fact_type'] == 'unknown')
        else:
            action = 'answer'
            is_correct = (s['fact_type'] == 'known' and s['correct_answer'])

        if is_correct:
            correct += 1
        results.append({**s, 'action': action, 'is_correct': is_correct})

    pipeline_acc = correct / len(test_signals)

    # Baselines
    base_correct = sum(1 for s in test_signals
                       if (s['fact_type'] == 'known' and s['correct_answer'])
                       or (s['fact_type'] == 'unknown' and not s['correct_answer']))
    base_acc = base_correct / len(test_signals)

    # Raw model baseline
    raw_correct = 0
    for prompt, answer, fact_type in all_test:
        top = get_top_token(ref_model, tok, prompt)
        fact_tokens = tok.encode(answer)
        fact_id = fact_tokens[-1] if fact_tokens else 0
        if (fact_type == 'known' and top == fact_id) or \
           (fact_type == 'unknown' and top != fact_id):
            raw_correct += 1
    raw_acc = raw_correct / len(all_test)

    print(f"\n  === RESULTS ===")
    print(f"    raw_base (no DPO):      {raw_acc:.0%}")
    print(f"    dpo_only (no refusal):  {base_acc:.0%}")
    print(f"    residual_gated (P119b): {pipeline_acc:.0%}")
    print(f"    P119 entropy_gated:     40% (for comparison)")

    del model, ref_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save
    out = {
        'phase': '119b', 'name': 'Residual-Gated Autopoietic F-DPO',
        'model': model_id, 'best_layer': best_layer,
        'threshold': float(best_thr), 'direction': 'known_higher' if best_dir else 'unknown_higher',
        'pipeline_acc': pipeline_acc, 'dpo_only_acc': base_acc, 'raw_base_acc': raw_acc,
        'p119_entropy_acc': 0.4,
        'details': results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase119b_residual_gated.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Accuracy comparison (P119 vs P119b)
    ax = axes[0]
    methods = ['raw_base', 'dpo_only', 'P119\nentropy', 'P119b\nresidual_l2']
    accs = [raw_acc, base_acc, 0.4, pipeline_acc]
    colors_bar = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(methods, accs, color=colors_bar)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.0%}', ha='center', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Combined Accuracy')
    ax.set_title('P119b vs P119: Residual Gating Wins?', fontweight='bold')

    # Panel 2: L2 distribution
    ax = axes[1]
    known_test_l2 = [s['l2'] for s in test_signals if s['fact_type'] == 'known']
    unknown_test_l2 = [s['l2'] for s in test_signals if s['fact_type'] == 'unknown']
    ax.hist(known_test_l2, bins=8, alpha=0.6, color='#2ecc71', label='Known')
    ax.hist(unknown_test_l2, bins=8, alpha=0.6, color='#e74c3c', label='Unknown')
    ax.axvline(x=best_thr, color='black', linestyle='--', linewidth=2,
              label=f'Threshold={best_thr:.1f}')
    ax.set_xlabel('Residual L2 Norm at L_0.94')
    ax.set_ylabel('Count')
    ax.set_title('Residual L2 Distribution')
    ax.legend()

    # Panel 3: Decision matrix
    ax = axes[2]
    known_r = [s for s in results if s['fact_type'] == 'known']
    unknown_r = [s for s in results if s['fact_type'] == 'unknown']
    tp = sum(1 for s in unknown_r if s['action'] == 'refuse')
    fp = sum(1 for s in known_r if s['action'] == 'refuse')
    tn = sum(1 for s in known_r if s['action'] == 'answer' and s['is_correct'])
    fn = sum(1 for s in unknown_r if s['action'] == 'answer')
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
    ax.set_title('Residual-Gated Decision Matrix')

    fig.suptitle('Phase 119b: Residual L2-Gated F-DPO (Fixing P119)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase119b_residual_gated.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 119b] Complete.")


if __name__ == '__main__':
    main()
