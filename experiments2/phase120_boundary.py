# -*- coding: utf-8 -*-
"""
Phase 120: DPO Phase Boundary Law (1.5B Scale Test)
Tests Deep Think's hypothesis: lr* ~ 1/sqrt(N_back)

Runs the same P117b back-only DPO sweep on Qwen2.5-1.5B to find
the critical learning rate where test accuracy collapses.

If the phase boundary shifts from 5e-6 (0.5B) to ~2e-6 (1.5B),
it confirms the inverse-square scaling law.

Model: Qwen2.5-1.5B (GPU, float32, back-only DPO)
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
    ("The tallest mountain in the world is", " Mount"),
    ("The currency of the United Kingdom is the", " pound"),
    ("The speed of sound is approximately", " 343"),
    ("Albert Einstein was born in", " Ul"),
    ("Photosynthesis converts sunlight into", " chemical"),
]

FLUENCY_PROMPTS = [
    "Once upon a time, there was a",
    "The weather today is",
    "In the year 2025,",
    "The most important thing in life is",
    "Scientists have discovered that",
]


def evaluate_accuracy(model, tok, facts):
    correct = 0
    for prompt, answer in facts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        fact_tokens = tok.encode(answer)
        fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
        if logits.argmax().item() == fact_id:
            correct += 1
    return correct / len(facts)


def evaluate_perplexity(model, tok, prompts):
    total_loss = 0
    total_tokens = 0
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        total_loss += out.loss.item() * inp['input_ids'].shape[1]
        total_tokens += inp['input_ids'].shape[1]
    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


def dpo_loss(model, ref_model, tok, prompt, chosen, rejected, beta):
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


def run_sweep(model_id, tok, n_layers, boundary, lr, beta, n_epochs):
    """Run back-only DPO with specific hyperparams."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).eval().to(DEVICE)

    model.eval()
    base_train = evaluate_accuracy(model, tok, [(p, c) for p, c, r in DPO_PAIRS])
    base_test = evaluate_accuracy(model, tok, TEST_FACTS)
    base_ppl = evaluate_perplexity(model, tok, FLUENCY_PROMPTS)

    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name:
                param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    for epoch in range(n_epochs):
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss(model, ref_model, tok, prompt, chosen, rejected, beta)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

    model.eval()
    final_train = evaluate_accuracy(model, tok, [(p, c) for p, c, r in DPO_PAIRS])
    final_test = evaluate_accuracy(model, tok, TEST_FACTS)
    final_ppl = evaluate_perplexity(model, tok, FLUENCY_PROMPTS)

    result = {
        'lr': lr, 'beta': beta,
        'n_trainable': n_trainable, 'n_total': n_total,
        'trainable_pct': n_trainable / n_total,
        'before': {'train': base_train, 'test': base_test, 'ppl': base_ppl},
        'after': {'train': final_train, 'test': final_test, 'ppl': final_ppl},
        'harmonic_mean': 2*final_train*final_test/(final_train+final_test) if (final_train+final_test) > 0 else 0,
    }

    del model, ref_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    print("[P120] DPO Phase Boundary Law (1.5B Scale)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoTokenizer
    model_id = 'Qwen/Qwen2.5-1.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    n_layers = 28
    boundary = int(n_layers * 0.94)  # 26

    # Focused sweep: finer grid around expected phase boundary
    learning_rates = [1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5]
    betas = [0.05, 0.1, 0.5]
    n_epochs = 5

    all_results = {}
    for lr in learning_rates:
        for beta in betas:
            key = f"lr={lr:.0e}_beta={beta}"
            print(f"\n  === {key} ===")
            result = run_sweep(model_id, tok, n_layers, boundary, lr, beta, n_epochs)
            all_results[key] = result
            b = result['before']
            a = result['after']
            print(f"    Trainable: {result['n_trainable']:,}/{result['n_total']:,} ({result['trainable_pct']:.1%})")
            print(f"    Before: train={b['train']:.0%}, test={b['test']:.0%}")
            print(f"    After:  train={a['train']:.0%}, test={a['test']:.0%}, ppl={a['ppl']:.1f}")
            print(f"    HM: {result['harmonic_mean']:.0%}")

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")

    # Find phase boundary
    best_key = max(all_results, key=lambda k: all_results[k]['harmonic_mean'])
    best = all_results[best_key]
    print(f"\n  BEST: {best_key}")
    print(f"    Train: {best['after']['train']:.0%}, Test: {best['after']['test']:.0%}")

    # Compare with 0.5B result
    lr_star_05b = 5e-6
    print(f"\n  Phase boundary comparison:")
    print(f"    0.5B: lr* = {lr_star_05b:.0e}")
    print(f"    1.5B: lr* = {best['lr']:.0e}")

    # Save
    out = {
        'phase': '120', 'name': 'DPO Phase Boundary Law',
        'model': model_id, 'boundary_layer': boundary,
        'best_config': best_key,
        'best_harmonic_mean': best['harmonic_mean'],
        'comparison_05b_lr_star': lr_star_05b,
        'results': all_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase120_boundary.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot: Heatmap comparison with P117b
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Test accuracy heatmap
    ax = axes[0]
    test_matrix = np.zeros((len(learning_rates), len(betas)))
    for i, lr in enumerate(learning_rates):
        for j, beta in enumerate(betas):
            key = f"lr={lr:.0e}_beta={beta}"
            if key in all_results:
                test_matrix[i, j] = all_results[key]['after']['test']
    im = ax.imshow(test_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b}' for b in betas])
    ax.set_yticks(range(len(learning_rates)))
    ax.set_yticklabels([f'{lr:.0e}' for lr in learning_rates])
    ax.set_xlabel('Beta')
    ax.set_ylabel('Learning Rate')
    ax.set_title('1.5B: Test Accuracy (Generalization)')
    for i in range(len(learning_rates)):
        for j in range(len(betas)):
            ax.text(j, i, f'{test_matrix[i,j]:.0%}', ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax)

    # Panel 2: Harmonic mean
    ax = axes[1]
    hm_matrix = np.zeros((len(learning_rates), len(betas)))
    for i, lr in enumerate(learning_rates):
        for j, beta in enumerate(betas):
            key = f"lr={lr:.0e}_beta={beta}"
            if key in all_results:
                hm_matrix[i, j] = all_results[key]['harmonic_mean']
    im2 = ax.imshow(hm_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b}' for b in betas])
    ax.set_yticks(range(len(learning_rates)))
    ax.set_yticklabels([f'{lr:.0e}' for lr in learning_rates])
    ax.set_xlabel('Beta')
    ax.set_ylabel('Learning Rate')
    ax.set_title('1.5B: Harmonic Mean (Train x Test)')
    for i in range(len(learning_rates)):
        for j in range(len(betas)):
            ax.text(j, i, f'{hm_matrix[i,j]:.0%}', ha='center', va='center', fontsize=10)
    plt.colorbar(im2, ax=ax)

    # Panel 3: Phase boundary comparison (0.5B vs 1.5B)
    ax = axes[2]
    # Extract test acc vs lr (averaged over beta)
    lr_test_15b = []
    for lr in learning_rates:
        tests = [all_results[f"lr={lr:.0e}_beta={b}"]['after']['test']
                 for b in betas if f"lr={lr:.0e}_beta={b}" in all_results]
        lr_test_15b.append(np.mean(tests) if tests else 0)
    ax.plot([f'{lr:.0e}' for lr in learning_rates], lr_test_15b, 'o-', color='#e74c3c',
            linewidth=2, markersize=8, label='1.5B (P120)')
    # P117b 0.5B data (hardcoded from previous results)
    lr_05b = [5e-7, 1e-6, 5e-6, 1e-5, 5e-5]
    test_05b = [0.6, 0.6, 0.6, 0.4, 0.4]  # From P117b
    ax.plot([f'{lr:.0e}' for lr in lr_05b], test_05b, 's--', color='#3498db',
            linewidth=2, markersize=8, label='0.5B (P117b)')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Phase Boundary: 0.5B vs 1.5B')
    ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    fig.suptitle('Phase 120: DPO Phase Boundary Law - Does lr* Scale with N?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase120_boundary.png'), dpi=150)
    plt.close()
    print("[Phase 120] Complete.")


if __name__ == '__main__':
    main()
