# -*- coding: utf-8 -*-
"""
Phase 124: 1.5B Phase Boundary Extended
P120b showed 1.5B never collapses up to lr=1e-5.
Extend to lr=2e-5, 5e-5, 1e-4 to find where it actually breaks.

Model: Qwen2.5-1.5B (GPU, float32)
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


def evaluate_accuracy(model, tok, facts):
    correct = 0
    for prompt, answer in facts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        fact_tokens = tok.encode(answer)
        fact_id = fact_tokens[-1] if fact_tokens else 0
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
        if logits.argmax().item() == fact_id:
            correct += 1
    return correct / len(facts)


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


def run_single(model_id, tok, n_layers, boundary, lr, n_epochs=5):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

    base_train = evaluate_accuracy(model, tok, [(p, c) for p, c, _ in DPO_PAIRS])
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
            loss = dpo_loss(model, ref_model, tok, prompt, chosen, rejected)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

    model.eval()
    final_train = evaluate_accuracy(model, tok, [(p, c) for p, c, _ in DPO_PAIRS])
    final_test = evaluate_accuracy(model, tok, TEST_FACTS)

    del model, ref_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'lr': lr, 'base_test': base_test, 'final_train': final_train, 'final_test': final_test}


def main():
    print("[P124] 1.5B Phase Boundary Extended")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoTokenizer
    model_id = 'Qwen/Qwen2.5-1.5B'
    n_layers = 28
    boundary = int(n_layers * 0.94)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    learning_rates = [1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4]

    results = {}
    for lr in learning_rates:
        print(f"\n  lr={lr:.0e}...", end=" ", flush=True)
        r = run_single(model_id, tok, n_layers, boundary, lr)
        results[f"{lr:.0e}"] = r
        print(f"train={r['final_train']:.0%}, test={r['final_test']:.0%}")

    # Find boundary
    base_test = results[list(results.keys())[0]]['base_test']
    boundary_lr = learning_rates[0]
    for lr in learning_rates:
        r = results[f"{lr:.0e}"]
        if r['final_test'] >= base_test - 0.01:
            boundary_lr = lr
        else:
            break

    print(f"\n  1.5B lr* = {boundary_lr:.0e}")
    print(f"  0.5B lr* = 8e-06 (from P120b)")
    print(f"  Ratio: {boundary_lr / 8e-6:.1f}x")

    # Save
    out = {
        'phase': '124', 'name': '1.5B Phase Boundary Extended',
        'model': model_id, 'lr_star_1b5': boundary_lr,
        'lr_star_0b5': 8e-6, 'ratio': boundary_lr / 8e-6,
        'results': results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase124_extended.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    lrs = [float(k) for k in results.keys()]
    tests = [results[k]['final_test'] for k in results]
    trains = [results[k]['final_train'] for k in results]

    ax.plot(range(len(lrs)), tests, 'o-', color='#e74c3c', linewidth=2,
           markersize=10, label='1.5B Test')
    ax.plot(range(len(lrs)), trains, 's--', color='#e74c3c', linewidth=1,
           markersize=6, alpha=0.5, label='1.5B Train')

    # Add 0.5B data from P120b
    lr_05b = [1e-5]
    test_05b = [0.4]
    ax.scatter([0], test_05b, color='#3498db', s=200, marker='*', zorder=5,
              label='0.5B Test (collapses here)')

    ax.axhline(y=base_test, color='gray', linestyle=':', alpha=0.5, label=f'Baseline={base_test:.0%}')
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f'{lr:.0e}' for lr in lrs], rotation=45, ha='right')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Accuracy')
    ax.set_title('Phase 124: Where Does 1.5B Actually Collapse?\n(0.5B collapses at 1e-5, marked with star)',
                fontweight='bold', fontsize=13)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    for i, (tr, te) in enumerate(zip(trains, tests)):
        ax.text(i, te + 0.03, f'{te:.0%}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase124_extended.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 124] Complete.")

if __name__ == '__main__':
    main()
