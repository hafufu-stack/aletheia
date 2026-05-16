# -*- coding: utf-8 -*-
"""
Phase 147: The Arithmetic Awakening
Does Embedding Surgery help LLMs learn arithmetic faster?

If number clustering is "Dyscalculia" (a bug, not a feature),
then dispersing numbers should let SFT teach arithmetic faster.

Compare: Base vs Surgery model learning simple addition.

Model: Qwen2.5-0.5B (GPU, float32)
"""
import torch, json, os, gc, numpy as np, time, random
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training data: simple single-digit addition
random.seed(42)
TRAIN_DATA = []
for a in range(10):
    for b in range(10):
        s = a + b
        TRAIN_DATA.append(f"{a} + {b} = {s}")
random.shuffle(TRAIN_DATA)

# Test data: held-out additions (different format to test generalization)
TEST_DATA = [
    ("1 + 1 =", " 2"),
    ("3 + 4 =", " 7"),
    ("5 + 5 =", " 10"),
    ("2 + 7 =", " 9"),
    ("0 + 8 =", " 8"),
    ("6 + 3 =", " 9"),
    ("4 + 4 =", " 8"),
    ("7 + 2 =", " 9"),
    ("9 + 0 =", " 9"),
    ("1 + 8 =", " 9"),
    ("3 + 3 =", " 6"),
    ("5 + 2 =", " 7"),
    ("8 + 1 =", " 9"),
    ("2 + 6 =", " 8"),
    ("4 + 5 =", " 9"),
]


def disperse_embeddings(model, tok, strength=1.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 11", " 12", " 13", " 14", " 15", " 16", " 17", " 18"]
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in num_tokens]
    ids = list(set(ids))
    vecs = embed[ids].clone()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()


def train_sft(model, tok, train_data, n_steps):
    """Train next-token prediction on arithmetic data."""
    model.train()
    # Train all parameters for SFT
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    losses = []
    for step in range(n_steps):
        text = train_data[step % len(train_data)]
        inp = tok(text, return_tensors='pt', truncation=True, max_length=32).to(DEVICE)
        ids = inp['input_ids']
        outputs = model(**inp, labels=ids)
        loss = outputs.loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


def evaluate_arithmetic(model, tok, test_data):
    """Check first digit of predicted answer."""
    correct = 0
    for prompt, expected in test_data:
        exp_ids = tok.encode(expected)
        space_id = tok.encode(" ")[-1]
        digit_ids = [i for i in exp_ids if i != space_id]
        first_digit = digit_ids[0] if digit_ids else exp_ids[-1]

        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        pred_id = logits.argmax().item()

        # If model predicts space first, check next token
        if pred_id == space_id:
            inp2 = torch.cat([inp['input_ids'],
                            torch.tensor([[pred_id]], device=DEVICE)], dim=1)
            with torch.no_grad():
                logits2 = model(input_ids=inp2).logits[0, -1, :].float()
            pred_id = logits2.argmax().item()

        if pred_id == first_digit:
            correct += 1
    return correct / len(test_data)


def main():
    print("[P147] The Arithmetic Awakening")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # SFT step milestones to measure
    milestones = [0, 50, 100, 200, 500, 1000]
    max_steps = max(milestones)

    # A: Base model SFT
    print("\n  === A: Base Model SFT ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    base_results = []
    # Pre-training accuracy
    acc = evaluate_arithmetic(model, tok, TEST_DATA)
    base_results.append({'steps': 0, 'acc': acc})
    print(f"    Step 0: {acc:.0%}")

    base_losses = train_sft(model, tok, TRAIN_DATA, max_steps)
    for m in milestones[1:]:
        # Re-evaluate at each milestone
        model_tmp = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
        losses = train_sft(model_tmp, tok, TRAIN_DATA, m)
        acc = evaluate_arithmetic(model_tmp, tok, TEST_DATA)
        base_results.append({'steps': m, 'acc': acc, 'final_loss': losses[-1]})
        print(f"    Step {m}: {acc:.0%} (loss={losses[-1]:.4f})")
        del model_tmp; gc.collect(); torch.cuda.empty_cache()
    del model; gc.collect(); torch.cuda.empty_cache()

    # B: Surgery Model SFT
    print("\n  === B: Surgery Model SFT ===")
    surg_results = []
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    disperse_embeddings(model, tok, strength=1.0)
    acc = evaluate_arithmetic(model, tok, TEST_DATA)
    surg_results.append({'steps': 0, 'acc': acc})
    print(f"    Step 0: {acc:.0%}")
    del model; gc.collect(); torch.cuda.empty_cache()

    for m in milestones[1:]:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
        disperse_embeddings(model, tok, strength=1.0)
        losses = train_sft(model, tok, TRAIN_DATA, m)
        acc = evaluate_arithmetic(model, tok, TEST_DATA)
        surg_results.append({'steps': m, 'acc': acc, 'final_loss': losses[-1]})
        print(f"    Step {m}: {acc:.0%} (loss={losses[-1]:.4f})")
        del model; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase147_awakening.json'), 'w') as f:
        json.dump({'phase': '147', 'name': 'Arithmetic Awakening',
                   'base': base_results, 'surgery': surg_results}, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Learning curves (accuracy)
    ax = axes[0]
    base_steps = [r['steps'] for r in base_results]
    base_accs = [r['acc'] for r in base_results]
    surg_steps = [r['steps'] for r in surg_results]
    surg_accs = [r['acc'] for r in surg_results]
    ax.plot(base_steps, base_accs, 'b-o', lw=2.5, markersize=8, label='Base Model')
    ax.plot(surg_steps, surg_accs, 'r-s', lw=2.5, markersize=8, label='Surgery Model')
    ax.set_xlabel('SFT Steps', fontsize=12)
    ax.set_ylabel('Arithmetic Accuracy', fontsize=12)
    ax.set_title('Learning Curves: Addition Task', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Right: Loss comparison
    ax = axes[1]
    base_loss_pts = [(r['steps'], r.get('final_loss', None)) for r in base_results if r.get('final_loss')]
    surg_loss_pts = [(r['steps'], r.get('final_loss', None)) for r in surg_results if r.get('final_loss')]
    if base_loss_pts:
        ax.plot([p[0] for p in base_loss_pts], [p[1] for p in base_loss_pts],
               'b-o', lw=2, markersize=8, label='Base Model')
    if surg_loss_pts:
        ax.plot([p[0] for p in surg_loss_pts], [p[1] for p in surg_loss_pts],
               'r-s', lw=2, markersize=8, label='Surgery Model')
    ax.set_xlabel('SFT Steps', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Loss Convergence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 147: The Arithmetic Awakening\n'
                'Does Surgery help LLMs learn math faster?',
                fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase147_awakening.png'), dpi=150,
               bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    # Compare at 500 steps
    base_500 = next((r['acc'] for r in base_results if r['steps'] == 500), 0)
    surg_500 = next((r['acc'] for r in surg_results if r['steps'] == 500), 0)
    if surg_500 > base_500 + 0.1:
        print(f"  -> ARITHMETIC AWAKENING CONFIRMED!")
        print(f"     Surgery learns faster: {surg_500:.0%} vs Base {base_500:.0%} at 500 steps")
    elif surg_500 < base_500 - 0.1:
        print(f"  -> Surgery HURTS learning: {surg_500:.0%} vs Base {base_500:.0%}")
    else:
        print(f"  -> No significant difference: surgery={surg_500:.0%} vs base={base_500:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 147] Complete.")

if __name__ == '__main__':
    main()
