# -*- coding: utf-8 -*-
"""
Phase 127: Direct Logit Difference (Ground Truth)
Instead of projecting ΔW through unembedding (P125), directly measure
what happens to token probabilities when we switch from base to DPO model.

For each DPO training pair: compute logits_DPO - logits_base at last position
and check if chosen token probability increases.

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

TEST_FACTS = [
    ("The tallest mountain in the world is", " Mount", " K"),
    ("The currency of the United Kingdom is the", " pound", " dollar"),
    ("The speed of sound is approximately", " 343", " 299"),
    ("Albert Einstein was born in", " Ul", " Berlin"),
    ("Photosynthesis converts sunlight into", " chemical", " electrical"),
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
    print("[P127] Direct Logit Difference (Ground Truth)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

    # Train DPO
    print("  Training DPO...")
    dpo_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
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
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss(dpo_model, ref_model, tok, prompt, chosen, rejected)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in dpo_model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
    dpo_model.eval()

    del ref_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Direct logit comparison
    print("\n  Comparing logits (base vs DPO)...")
    all_pairs = DPO_PAIRS + [(p, c, r) for p, c, r in TEST_FACTS]

    results = []
    for prompt, chosen, rejected in all_pairs:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        chosen_ids = tok.encode(chosen)
        rejected_ids = tok.encode(rejected)
        chosen_id = chosen_ids[-1] if chosen_ids else 0
        rejected_id = rejected_ids[-1] if rejected_ids else 0

        with torch.no_grad():
            base_logits = base_model(**inp).logits[0, -1, :].float()
            dpo_logits = dpo_model(**inp).logits[0, -1, :].float()

        base_probs = torch.softmax(base_logits, dim=-1)
        dpo_probs = torch.softmax(dpo_logits, dim=-1)

        logit_diff = dpo_logits - base_logits

        is_train = prompt in [p for p, _, _ in DPO_PAIRS]

        def safe_token(tid):
            try:
                return tok.decode([tid]).encode('ascii', 'replace').decode()
            except Exception:
                return f'<{tid}>'

        result = {
            'prompt': prompt[:40],
            'is_train': is_train,
            'chosen_token': safe_token(chosen_id),
            'rejected_token': safe_token(rejected_id),
            'base_chosen_prob': base_probs[chosen_id].item(),
            'dpo_chosen_prob': dpo_probs[chosen_id].item(),
            'base_rejected_prob': base_probs[rejected_id].item(),
            'dpo_rejected_prob': dpo_probs[rejected_id].item(),
            'chosen_logit_change': logit_diff[chosen_id].item(),
            'rejected_logit_change': logit_diff[rejected_id].item(),
            'chosen_prob_change': (dpo_probs[chosen_id] - base_probs[chosen_id]).item(),
            'rejected_prob_change': (dpo_probs[rejected_id] - base_probs[rejected_id]).item(),
            'base_top': safe_token(base_logits.argmax().item()),
            'dpo_top': safe_token(dpo_logits.argmax().item()),
            'dpo_correct': (dpo_logits.argmax().item() == chosen_id),
            'base_correct': (base_logits.argmax().item() == chosen_id),
        }
        results.append(result)

        direction = 'UP' if result['chosen_prob_change'] > 0 else 'DOWN'
        tag = 'TRAIN' if is_train else 'TEST'
        print(f"    [{tag}] {prompt[:35]:35s} "
              f"chosen={result['chosen_token']:6s} "
              f"{result['base_chosen_prob']:.3f}->{result['dpo_chosen_prob']:.3f} ({direction}) | "
              f"rej={result['rejected_token']:6s} "
              f"{result['base_rejected_prob']:.3f}->{result['dpo_rejected_prob']:.3f}")

    # Summary statistics
    train_results = [r for r in results if r['is_train']]
    test_results = [r for r in results if not r['is_train']]

    train_chosen_up = sum(1 for r in train_results if r['chosen_prob_change'] > 0)
    train_rejected_down = sum(1 for r in train_results if r['rejected_prob_change'] < 0)
    test_chosen_up = sum(1 for r in test_results if r['chosen_prob_change'] > 0)

    print(f"\n  === GROUND TRUTH ===")
    print(f"    Train: chosen prob UP in {train_chosen_up}/{len(train_results)}")
    print(f"    Train: rejected prob DOWN in {train_rejected_down}/{len(train_results)}")
    print(f"    Test:  chosen prob UP in {test_chosen_up}/{len(test_results)}")
    print(f"    Train DPO correct: {sum(r['dpo_correct'] for r in train_results)}/{len(train_results)}")
    print(f"    Test DPO correct:  {sum(r['dpo_correct'] for r in test_results)}/{len(test_results)}")

    del base_model, dpo_model, optimizer; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save
    out = {
        'phase': '127', 'name': 'Direct Logit Difference',
        'results': results,
        'summary': {
            'train_chosen_up': train_chosen_up,
            'train_total': len(train_results),
            'train_rejected_down': train_rejected_down,
            'test_chosen_up': test_chosen_up,
            'test_total': len(test_results),
        }
    }
    with open(os.path.join(RESULTS_DIR, 'phase127_logit_diff.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Chosen prob change (train)
    ax = axes[0]
    prompts_short = [r['prompt'][:20] for r in train_results]
    chosen_changes = [r['chosen_prob_change'] for r in train_results]
    rejected_changes = [r['rejected_prob_change'] for r in train_results]
    x = range(len(train_results))
    ax.bar([i-0.15 for i in x], chosen_changes, 0.3, color='#2ecc71', label='Chosen')
    ax.bar([i+0.15 for i in x], rejected_changes, 0.3, color='#e74c3c', label='Rejected')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(prompts_short, rotation=90, fontsize=6)
    ax.set_ylabel('Prob Change (DPO - Base)')
    ax.set_title('Train Set: Probability Changes', fontweight='bold')
    ax.legend()

    # Panel 2: Test set
    ax = axes[1]
    prompts_short = [r['prompt'][:20] for r in test_results]
    chosen_changes = [r['chosen_prob_change'] for r in test_results]
    rejected_changes = [r['rejected_prob_change'] for r in test_results]
    x = range(len(test_results))
    ax.bar([i-0.15 for i in x], chosen_changes, 0.3, color='#2ecc71', label='Chosen')
    ax.bar([i+0.15 for i in x], rejected_changes, 0.3, color='#e74c3c', label='Rejected')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(prompts_short, rotation=90, fontsize=7)
    ax.set_ylabel('Prob Change (DPO - Base)')
    ax.set_title('Test Set: Generalization', fontweight='bold')
    ax.legend()

    # Panel 3: Summary
    ax = axes[2]
    cats = ['Train\nChosen UP', 'Train\nRejected DOWN', 'Test\nChosen UP']
    vals = [train_chosen_up/len(train_results),
            train_rejected_down/len(train_results),
            test_chosen_up/len(test_results) if test_results else 0]
    bars = ax.bar(cats, vals, color=['#2ecc71', '#e74c3c', '#3498db'])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Fraction')
    ax.set_title('DPO Effect Summary', fontweight='bold')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
               f'{val:.0%}', ha='center', fontweight='bold', fontsize=12)

    fig.suptitle('Phase 127: What Does DPO ACTUALLY Do to Token Probabilities?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase127_logit_diff.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 127] Complete.")

if __name__ == '__main__':
    main()
