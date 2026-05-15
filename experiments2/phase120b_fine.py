# -*- coding: utf-8 -*-
"""
Phase 120b: Fine-Grained lr* Boundary Detection
P120 showed lr*=5e-6 for both 0.5B and 1.5B, but our grid was coarse.
This phase does a fine sweep around the boundary for BOTH models
to detect if there's a subtle shift.

Sweep: lr in [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 1e-5]
Beta fixed at 0.05 (irrelevant per P117b).

Models: Qwen2.5-0.5B (24 layers), Qwen2.5-1.5B (28 layers)
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


def run_single(model_id, tok, n_layers, boundary, lr, beta=0.05, n_epochs=5):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

    base_train = evaluate_accuracy(model, tok, [(p, c) for p, c, r in DPO_PAIRS])
    base_test = evaluate_accuracy(model, tok, TEST_FACTS)

    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name:
                param.requires_grad = True

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

    del model, ref_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'lr': lr, 'base_train': base_train, 'base_test': base_test,
        'final_train': final_train, 'final_test': final_test,
        'test_delta': final_test - base_test,
    }


def main():
    print("[P120b] Fine-Grained lr* Boundary Detection")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoTokenizer

    # Fine sweep around 5e-6
    learning_rates = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 1e-5]

    models = [
        ('Qwen/Qwen2.5-0.5B', '0.5B', 24),
        ('Qwen/Qwen2.5-1.5B', '1.5B', 28),
    ]

    all_results = {}
    for model_id, label, n_layers in models:
        print(f"\n  === {label} ===")
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        boundary = int(n_layers * 0.94)

        model_results = {}
        for lr in learning_rates:
            print(f"    lr={lr:.0e}...", end=" ", flush=True)
            r = run_single(model_id, tok, n_layers, boundary, lr)
            model_results[f"{lr:.0e}"] = r
            print(f"train={r['final_train']:.0%}, test={r['final_test']:.0%}, "
                  f"delta={r['test_delta']:+.0%}")

        # Find exact boundary: highest lr where test doesn't drop
        boundary_lr = learning_rates[0]
        base_test = model_results[f"{learning_rates[0]:.0e}"]['base_test']
        for lr in learning_rates:
            r = model_results[f"{lr:.0e}"]
            if r['final_test'] >= base_test - 0.01:  # Allow 1% tolerance
                boundary_lr = lr
            else:
                break

        all_results[label] = {
            'model': model_id, 'boundary': boundary,
            'lr_star': boundary_lr, 'sweep': model_results,
        }
        print(f"    lr* = {boundary_lr:.0e}")

    # Save
    out = {'phase': '120b', 'name': 'Fine-Grained lr* Boundary', 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase120b_fine_boundary.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for label, data in all_results.items():
        sweep = data['sweep']
        lrs = [float(k) for k in sweep.keys()]
        tests = [sweep[k]['final_test'] for k in sweep.keys()]
        trains = [sweep[k]['final_train'] for k in sweep.keys()]
        base = sweep[list(sweep.keys())[0]]['base_test']

        ax.plot(range(len(lrs)), tests, 'o-', color=colors.get(label, 'gray'),
               linewidth=2, markersize=8, label=f'{label} Test')
        ax.plot(range(len(lrs)), trains, 's--', color=colors.get(label, 'gray'),
               linewidth=1, markersize=5, alpha=0.5, label=f'{label} Train')
        ax.axhline(y=base, color=colors.get(label, 'gray'), linestyle=':',
                  alpha=0.3)
        # Mark lr*
        lr_star_idx = lrs.index(data['lr_star'])
        ax.axvline(x=lr_star_idx, color=colors.get(label, 'gray'),
                  linestyle='--', alpha=0.5)
        ax.annotate(f'lr*={data["lr_star"]:.0e}',
                   xy=(lr_star_idx, tests[lr_star_idx]),
                   xytext=(lr_star_idx + 0.5, tests[lr_star_idx] + 0.05),
                   fontsize=10, fontweight='bold', color=colors.get(label, 'gray'),
                   arrowprops=dict(arrowstyle='->', color=colors.get(label, 'gray')))

    ax.set_xticks(range(len(learning_rates)))
    ax.set_xticklabels([f'{lr:.0e}' for lr in learning_rates], rotation=45, ha='right')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Accuracy')
    ax.set_title('Phase 120b: Fine-Grained DPO Phase Boundary\nWhere Exactly Does Generalization Collapse?',
                fontweight='bold', fontsize=13)
    ax.legend(loc='lower left')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase120b_fine_boundary.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 120b] Complete.")


if __name__ == '__main__':
    main()
