# -*- coding: utf-8 -*-
"""
Phase 117b: F-DPO Hyperparameter Sweep (Back-Only)
P117 showed back-only DPO (5.3% params) is safest but lost test acc (60->40%).
Sweep learning rate and beta to find optimal factuality-preservation tradeoff.

Key question: Can we improve train acc WITHOUT destroying test generalization?

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
    text_chosen = prompt + chosen
    text_rejected = prompt + rejected
    inp_c = tok(text_chosen, return_tensors='pt').to(DEVICE)
    inp_r = tok(text_rejected, return_tensors='pt').to(DEVICE)
    prompt_ids = tok(prompt, return_tensors='pt')['input_ids']
    prompt_len = prompt_ids.shape[1]

    logits_c = model(**inp_c).logits
    logits_r = model(**inp_r).logits
    lc = logits_c[0, prompt_len-1:-1, :].float().clamp(-100, 100)
    lr = logits_r[0, prompt_len-1:-1, :].float().clamp(-100, 100)
    lp_c = F.log_softmax(lc, dim=-1)
    lp_r = F.log_softmax(lr, dim=-1)

    target_c = inp_c['input_ids'][0, prompt_len:]
    target_r = inp_r['input_ids'][0, prompt_len:]
    plp_c = lp_c.gather(1, target_c.unsqueeze(1)).squeeze()
    plp_r = lp_r.gather(1, target_r.unsqueeze(1)).squeeze()
    if plp_c.dim() == 0: plp_c = plp_c.unsqueeze(0)
    if plp_r.dim() == 0: plp_r = plp_r.unsqueeze(0)
    plp_c = plp_c.sum()
    plp_r = plp_r.sum()

    with torch.no_grad():
        rlc = ref_model(**inp_c).logits[0, prompt_len-1:-1, :].float().clamp(-100, 100)
        rlr = ref_model(**inp_r).logits[0, prompt_len-1:-1, :].float().clamp(-100, 100)
        rlp_c = F.log_softmax(rlc, dim=-1).gather(1, target_c.unsqueeze(1)).squeeze()
        rlp_r = F.log_softmax(rlr, dim=-1).gather(1, target_r.unsqueeze(1)).squeeze()
        if rlp_c.dim() == 0: rlp_c = rlp_c.unsqueeze(0)
        if rlp_r.dim() == 0: rlp_r = rlp_r.unsqueeze(0)
        rlp_c = rlp_c.sum()
        rlp_r = rlp_r.sum()

    logits_diff = beta * ((plp_c - rlp_c) - (plp_r - rlp_r))
    return -F.logsigmoid(logits_diff)


def run_back_only_dpo(model_id, tok, n_layers, boundary, lr, beta, n_epochs):
    """Run back-only DPO with specific hyperparams. Returns results dict."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).eval().to(DEVICE)

    # Baseline
    model.eval()
    base_train = evaluate_accuracy(model, tok, [(p, c) for p, c, r in DPO_PAIRS])
    base_test = evaluate_accuracy(model, tok, TEST_FACTS)
    base_ppl = evaluate_perplexity(model, tok, FLUENCY_PROMPTS)

    # Freeze: only back MLPs trainable
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name:
                param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    history = {'loss': [], 'train_acc': [], 'test_acc': [], 'ppl': []}

    for epoch in range(n_epochs):
        epoch_loss = 0
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss(model, ref_model, tok, prompt, chosen, rejected, beta)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(DPO_PAIRS)
        model.eval()
        train_acc = evaluate_accuracy(model, tok, [(p, c) for p, c, r in DPO_PAIRS])
        test_acc = evaluate_accuracy(model, tok, TEST_FACTS)
        ppl = evaluate_perplexity(model, tok, FLUENCY_PROMPTS)
        history['loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['ppl'].append(ppl)
        model.train()

    model.eval()
    final_train = evaluate_accuracy(model, tok, [(p, c) for p, c, r in DPO_PAIRS])
    final_test = evaluate_accuracy(model, tok, TEST_FACTS)
    final_ppl = evaluate_perplexity(model, tok, FLUENCY_PROMPTS)

    result = {
        'lr': lr, 'beta': beta, 'n_epochs': n_epochs,
        'before': {'train': base_train, 'test': base_test, 'ppl': base_ppl},
        'after': {'train': final_train, 'test': final_test, 'ppl': final_ppl},
        'history': history,
        # Key metric: harmonic mean of train and test
        'harmonic_mean': 2 * final_train * final_test / (final_train + final_test) if (final_train + final_test) > 0 else 0,
    }

    del model, ref_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    print("[P117b] F-DPO Hyperparameter Sweep (Back-Only)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    n_layers = 24
    boundary = int(n_layers * 0.94)  # 22

    # Sweep grid
    learning_rates = [5e-7, 1e-6, 5e-6, 1e-5, 5e-5]
    betas = [0.05, 0.1, 0.3, 0.5, 1.0]
    n_epochs = 5

    all_results = {}
    for lr in learning_rates:
        for beta in betas:
            key = f"lr={lr:.0e}_beta={beta}"
            print(f"\n  === {key} ===")
            result = run_back_only_dpo(model_id, tok, n_layers, boundary, lr, beta, n_epochs)
            all_results[key] = result
            b = result['before']
            a = result['after']
            hm = result['harmonic_mean']
            print(f"    Before: train={b['train']:.0%}, test={b['test']:.0%}")
            print(f"    After:  train={a['train']:.0%}, test={a['test']:.0%}, ppl={a['ppl']:.1f}")
            print(f"    Harmonic mean: {hm:.0%}")

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")

    # Find best
    best_key = max(all_results, key=lambda k: all_results[k]['harmonic_mean'])
    best = all_results[best_key]
    print(f"\n  BEST: {best_key}")
    print(f"    Train: {best['after']['train']:.0%}, Test: {best['after']['test']:.0%}")
    print(f"    Harmonic mean: {best['harmonic_mean']:.0%}")

    # Save
    out = {
        'phase': '117b', 'name': 'F-DPO Hyperparameter Sweep',
        'model': model_id, 'boundary_layer': boundary,
        'best_config': best_key,
        'best_harmonic_mean': best['harmonic_mean'],
        'results': all_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase117b_sweep.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Heatmap of harmonic mean (lr x beta)
    ax = axes[0, 0]
    hm_matrix = np.zeros((len(learning_rates), len(betas)))
    for i, lr in enumerate(learning_rates):
        for j, beta in enumerate(betas):
            key = f"lr={lr:.0e}_beta={beta}"
            if key in all_results:
                hm_matrix[i, j] = all_results[key]['harmonic_mean']
    im = ax.imshow(hm_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b}' for b in betas])
    ax.set_yticks(range(len(learning_rates)))
    ax.set_yticklabels([f'{lr:.0e}' for lr in learning_rates])
    ax.set_xlabel('Beta (DPO)')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Harmonic Mean (Train x Test)')
    for i in range(len(learning_rates)):
        for j in range(len(betas)):
            ax.text(j, i, f'{hm_matrix[i,j]:.0%}', ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   color='white' if hm_matrix[i,j] < 0.3 else 'black')
    plt.colorbar(im, ax=ax)

    # Panel 2: Train acc heatmap
    ax = axes[0, 1]
    train_matrix = np.zeros((len(learning_rates), len(betas)))
    for i, lr in enumerate(learning_rates):
        for j, beta in enumerate(betas):
            key = f"lr={lr:.0e}_beta={beta}"
            if key in all_results:
                train_matrix[i, j] = all_results[key]['after']['train']
    im2 = ax.imshow(train_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b}' for b in betas])
    ax.set_yticks(range(len(learning_rates)))
    ax.set_yticklabels([f'{lr:.0e}' for lr in learning_rates])
    ax.set_xlabel('Beta')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Train Accuracy After DPO')
    for i in range(len(learning_rates)):
        for j in range(len(betas)):
            ax.text(j, i, f'{train_matrix[i,j]:.0%}', ha='center', va='center', fontsize=8)
    plt.colorbar(im2, ax=ax)

    # Panel 3: Test acc heatmap
    ax = axes[1, 0]
    test_matrix = np.zeros((len(learning_rates), len(betas)))
    for i, lr in enumerate(learning_rates):
        for j, beta in enumerate(betas):
            key = f"lr={lr:.0e}_beta={beta}"
            if key in all_results:
                test_matrix[i, j] = all_results[key]['after']['test']
    im3 = ax.imshow(test_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b}' for b in betas])
    ax.set_yticks(range(len(learning_rates)))
    ax.set_yticklabels([f'{lr:.0e}' for lr in learning_rates])
    ax.set_xlabel('Beta')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Test Accuracy After DPO (Generalization)')
    for i in range(len(learning_rates)):
        for j in range(len(betas)):
            ax.text(j, i, f'{test_matrix[i,j]:.0%}', ha='center', va='center', fontsize=8)
    plt.colorbar(im3, ax=ax)

    # Panel 4: PPL heatmap
    ax = axes[1, 1]
    ppl_matrix = np.zeros((len(learning_rates), len(betas)))
    for i, lr in enumerate(learning_rates):
        for j, beta in enumerate(betas):
            key = f"lr={lr:.0e}_beta={beta}"
            if key in all_results:
                ppl_val = all_results[key]['after']['ppl']
                ppl_matrix[i, j] = min(ppl_val, 20) if not np.isnan(ppl_val) else 20
    im4 = ax.imshow(ppl_matrix, cmap='Reds_r', aspect='auto')
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b}' for b in betas])
    ax.set_yticks(range(len(learning_rates)))
    ax.set_yticklabels([f'{lr:.0e}' for lr in learning_rates])
    ax.set_xlabel('Beta')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Perplexity After DPO (Fluency)')
    for i in range(len(learning_rates)):
        for j in range(len(betas)):
            ax.text(j, i, f'{ppl_matrix[i,j]:.1f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im4, ax=ax)

    fig.suptitle(f'Phase 117b: F-DPO Sweep (Back-Only, Best={best_key}, HM={best["harmonic_mean"]:.0%})',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase117b_sweep.png'), dpi=150)
    plt.close()
    print("[Phase 117b] Complete.")


if __name__ == '__main__':
    main()
