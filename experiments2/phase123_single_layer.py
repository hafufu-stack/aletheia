# -*- coding: utf-8 -*-
"""
Phase 123: Single-Layer DPO (L23 Only)
P122 showed only L23's DPO edit is statistically significant (z=4.84).
Hypothesis: Training ONLY L23's MLP achieves the same factuality gain
as training L22+L23, with less risk of overfitting.

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
        fact_id = fact_tokens[-1] if fact_tokens else 0
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
        if logits.argmax().item() == fact_id:
            correct += 1
    return correct / len(facts)


def evaluate_perplexity(model, tok, prompts):
    total_loss, total_tokens = 0, 0
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        total_loss += out.loss.item() * inp['input_ids'].shape[1]
        total_tokens += inp['input_ids'].shape[1]
    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


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


def train_and_eval(model_id, tok, n_layers, target_layers, lr, label):
    """Train DPO on specific layers and evaluate."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

    base_train = evaluate_accuracy(model, tok, [(p, c) for p, c, _ in DPO_PAIRS])
    base_test = evaluate_accuracy(model, tok, TEST_FACTS)
    base_ppl = evaluate_perplexity(model, tok, FLUENCY_PROMPTS)

    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = False
        for i in target_layers:
            if f"layers.{i}." in name and "mlp" in name:
                param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    for epoch in range(5):
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
    final_ppl = evaluate_perplexity(model, tok, FLUENCY_PROMPTS)

    result = {
        'label': label, 'target_layers': target_layers,
        'n_trainable': n_trainable, 'lr': lr,
        'before': {'train': base_train, 'test': base_test, 'ppl': base_ppl},
        'after': {'train': final_train, 'test': final_test, 'ppl': final_ppl},
    }

    del model, ref_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    print("[P123] Single-Layer DPO (L23 Only)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    configs = [
        ([23], 5e-6, 'L23 only'),
        ([22], 5e-6, 'L22 only'),
        ([22, 23], 5e-6, 'L22+L23 (P117b)'),
        ([21, 22, 23], 5e-6, 'L21-L23'),
        ([23], 1e-5, 'L23 only (2x lr)'),
    ]

    results = []
    for layers, lr, label in configs:
        print(f"\n  === {label} ===")
        r = train_and_eval(model_id, tok, n_layers, layers, lr, label)
        results.append(r)
        b, a = r['before'], r['after']
        print(f"    Params: {r['n_trainable']:,}")
        print(f"    Before: train={b['train']:.0%}, test={b['test']:.0%}, ppl={b['ppl']:.1f}")
        print(f"    After:  train={a['train']:.0%}, test={a['test']:.0%}, ppl={a['ppl']:.1f}")

    # Save
    out = {'phase': '123', 'name': 'Single-Layer DPO', 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase123_single_layer.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = [r['label'] for r in results]
    trains = [r['after']['train'] for r in results]
    tests = [r['after']['test'] for r in results]
    ppls = [r['after']['ppl'] for r in results]
    params = [r['n_trainable'] for r in results]

    ax = axes[0]
    x = range(len(labels))
    bars1 = ax.bar([i-0.2 for i in x], trains, 0.35, color='#3498db', label='Train')
    bars2 = ax.bar([i+0.2 for i in x], tests, 0.35, color='#2ecc71', label='Test')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Test Accuracy')
    ax.legend()
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars1, trains):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
               f'{val:.0%}', ha='center', fontsize=8)
    for bar, val in zip(bars2, tests):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
               f'{val:.0%}', ha='center', fontsize=8)

    ax = axes[1]
    ax.bar(labels, ppls, color='#e74c3c')
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Perplexity')
    ax.set_title('Fluency (Lower = Better)')
    for i, v in enumerate(ppls):
        ax.text(i, v+0.1, f'{v:.1f}', ha='center', fontsize=9)

    ax = axes[2]
    ax.bar(labels, [p/1000 for p in params], color='#9b59b6')
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Trainable Params (K)')
    ax.set_title('Parameter Efficiency')
    for i, p in enumerate(params):
        ax.text(i, p/1000+0.5, f'{p:,}', ha='center', fontsize=7)

    fig.suptitle('Phase 123: How Many Layers Does DPO Need?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase123_single_layer.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 123] Complete.")

if __name__ == '__main__':
    main()
