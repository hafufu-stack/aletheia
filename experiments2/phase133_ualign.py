# -*- coding: utf-8 -*-
"""
Phase 133: The UAlign Blackhole - Teaching "I Don't Know" via DPO
P127 proved DPO is 100% reliable at suppressing rejected tokens.
P129 proved DPO cannot promote numerical facts.

Solution: Use DPO's suppression power to teach abstention.
Train: chosen = "I don't know" / rejected = wrong number
Since DPO reliably suppresses rejected, the model should learn to
AVOID fabricating numerical answers it doesn't know.

This tests the UAlign hypothesis: the safest alignment for unknown
facts is calibrated uncertainty, not forced generation.

Experiment:
1. Baseline: measure numerical fact hallucination rate
2. UAlign DPO: train with "I don't know" as chosen
3. Measure: does the model stop fabricating numbers?
4. Compare: word facts should be unaffected

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

# UAlign training pairs: chosen = abstention, rejected = wrong number
UALIGN_PAIRS = [
    ("Water freezes at", " I", " 50"),      # "I don't know" starts with I
    ("The boiling point of water is", " I", " 500"),
    ("The atomic number of carbon is", " I", " 12"),
    ("The speed of light is approximately", " I", " 186"),
    ("A year has", " I", " 100"),
    ("The atomic number of oxygen is", " I", " 16"),
    ("The number of planets in the solar system is", " I", " 5"),
    ("The melting point of iron is", " I", " 500"),
]

# Normal DPO pairs (word facts, should not be affected)
NORMAL_PAIRS = [
    ("The capital of Japan is", " Tokyo", " Osaka"),
    ("The capital of France is", " Paris", " Lyon"),
    ("The capital of Germany is", " Berlin", " Munich"),
    ("The capital of Italy is", " Rome", " Milan"),
]

# Test set
TEST_PROMPTS = [
    ("Water freezes at", " 0", "number", "train"),
    ("The boiling point of water is", " 100", "number", "train"),
    ("The speed of light is approximately", " 299", "number", "train"),
    ("The number of planets in the solar system is", " 8", "number", "train"),
    ("The melting point of iron is", " 15", "number", "train"),
    ("The capital of Japan is", " Tokyo", "word", "normal"),
    ("The capital of France is", " Paris", "word", "normal"),
    ("The largest planet is", " Jupiter", "word", "test"),
    ("Pi is approximately", " 3", "number", "test"),
    ("A decade has", " 10", "number", "test"),
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


def evaluate(model, tok, prompts):
    """Evaluate: check if output is a number, abstention, or correct."""
    results = []
    for item in prompts:
        prompt, expected, cat, split = item
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        pred_id = logits.argmax().item()
        pred_tok = tok.decode([pred_id])

        # Check if prediction is a number
        is_number = pred_tok.strip().replace('.', '').replace('-', '').isdigit()
        # Check if prediction starts abstention pattern
        is_abstain = pred_tok.strip() in ['I', 'It', 'Unknown', 'N', 'None', 'That']
        # Check if correct
        exp_id = tok.encode(expected)[-1]
        is_correct = (pred_id == exp_id)

        # Probability mass on number tokens (0-9, common numbers)
        num_token_ids = [tok.encode(f" {i}")[-1] for i in range(10)]
        num_prob = sum(probs[tid].item() for tid in num_token_ids)

        results.append({
            'prompt': prompt[:40], 'cat': cat, 'split': split,
            'pred': pred_tok.encode('ascii', 'replace').decode().strip(),
            'expected': expected.strip(),
            'is_correct': is_correct,
            'is_number': is_number,
            'is_abstain': is_abstain,
            'num_prob_mass': num_prob,
            'top_prob': probs[pred_id].item(),
        })
    return results


def main():
    print("[P133] UAlign Blackhole - Teaching 'I Don't Know'")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)  # L22
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # === Baseline ===
    print("\n  === Baseline ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    base_results = evaluate(base_model, tok, TEST_PROMPTS)
    for r in base_results:
        print(f"    [{r['cat']:6s}] {r['prompt']:40s} -> {r['pred']:10s} "
              f"(num={r['is_number']}, correct={r['is_correct']}, "
              f"num_mass={r['num_prob_mass']:.3f})")
    del base_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === UAlign DPO Training ===
    print("\n  === UAlign DPO Training ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

    # Freeze all except last MLP layers
    for name, param in model.named_parameters():
        param.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name:
                param.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"    Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=5e-6)

    # Train on UAlign pairs (abstention) + normal pairs (word facts)
    losses = []
    for epoch in range(10):
        epoch_loss = 0
        # UAlign: teach abstention on numbers
        for prompt, chosen, rejected in UALIGN_PAIRS:
            loss = dpo_loss(model, ref_model, tok, prompt, chosen, rejected)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        # Normal: maintain word fact accuracy
        for prompt, chosen, rejected in NORMAL_PAIRS:
            loss = dpo_loss(model, ref_model, tok, prompt, chosen, rejected)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / (len(UALIGN_PAIRS) + len(NORMAL_PAIRS))
        losses.append(avg)
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}: loss={avg:.4f}")

    model.eval()

    # === Evaluate UAlign model ===
    print("\n  === UAlign Model Evaluation ===")
    ualign_results = evaluate(model, tok, TEST_PROMPTS)
    for r in ualign_results:
        print(f"    [{r['cat']:6s}] {r['prompt']:40s} -> {r['pred']:10s} "
              f"(num={r['is_number']}, abstain={r['is_abstain']}, "
              f"num_mass={r['num_prob_mass']:.3f})")

    del model, ref_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Analysis ===
    print("\n  === Analysis ===")
    # Compare number probability mass before/after
    base_num_mass = [r['num_prob_mass'] for r in base_results if r['cat'] == 'number']
    ualign_num_mass = [r['num_prob_mass'] for r in ualign_results if r['cat'] == 'number']
    base_abstain = sum(1 for r in base_results if r['cat'] == 'number' and r['is_abstain'])
    ualign_abstain = sum(1 for r in ualign_results if r['cat'] == 'number' and r['is_abstain'])
    n_num = sum(1 for r in base_results if r['cat'] == 'number')

    base_word_correct = sum(1 for r in base_results if r['cat'] == 'word' and r['is_correct'])
    ualign_word_correct = sum(1 for r in ualign_results if r['cat'] == 'word' and r['is_correct'])
    n_word = sum(1 for r in base_results if r['cat'] == 'word')

    print(f"    Number prob mass: base={np.mean(base_num_mass):.3f} -> "
          f"ualign={np.mean(ualign_num_mass):.3f} "
          f"({np.mean(ualign_num_mass)-np.mean(base_num_mass):+.3f})")
    print(f"    Abstention rate: base={base_abstain}/{n_num} -> "
          f"ualign={ualign_abstain}/{n_num}")
    print(f"    Word accuracy: base={base_word_correct}/{n_word} -> "
          f"ualign={ualign_word_correct}/{n_word}")

    # Save
    out = {
        'phase': '133', 'name': 'UAlign Blackhole',
        'baseline': [{k: v for k, v in r.items()} for r in base_results],
        'ualign': [{k: v for k, v in r.items()} for r in ualign_results],
        'summary': {
            'base_num_mass_mean': float(np.mean(base_num_mass)),
            'ualign_num_mass_mean': float(np.mean(ualign_num_mass)),
            'base_abstain_rate': base_abstain / n_num if n_num else 0,
            'ualign_abstain_rate': ualign_abstain / n_num if n_num else 0,
            'base_word_acc': base_word_correct / n_word if n_word else 0,
            'ualign_word_acc': ualign_word_correct / n_word if n_word else 0,
            'training_losses': losses,
        },
    }
    with open(os.path.join(RESULTS_DIR, 'phase133_ualign.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Number probability mass before/after
    ax = axes[0]
    prompts = [r['prompt'][:20] for r in base_results if r['cat'] == 'number']
    x = np.arange(len(prompts))
    w = 0.35
    ax.bar(x-w/2, base_num_mass, w, label='Baseline', color='#e74c3c', alpha=0.8)
    ax.bar(x+w/2, ualign_num_mass, w, label='UAlign', color='#2ecc71', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(prompts, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('P(any number token)')
    ax.legend(); ax.set_title('Number Prob Mass', fontweight='bold')

    # Panel 2: Abstention rate
    ax = axes[1]
    cats = ['Num Abstain', 'Num Fabricate', 'Word Correct']
    base_vals = [base_abstain/n_num, 1-base_abstain/n_num, base_word_correct/n_word]
    ualign_vals = [ualign_abstain/n_num, 1-ualign_abstain/n_num, ualign_word_correct/n_word]
    x = np.arange(len(cats))
    ax.bar(x-w/2, base_vals, w, label='Baseline', color='#e74c3c', alpha=0.8)
    ax.bar(x+w/2, ualign_vals, w, label='UAlign', color='#2ecc71', alpha=0.8)
    for i, (bv, uv) in enumerate(zip(base_vals, ualign_vals)):
        ax.text(i-w/2, bv+0.02, f'{bv:.0%}', ha='center', fontsize=9)
        ax.text(i+w/2, uv+0.02, f'{uv:.0%}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel('Rate'); ax.legend()
    ax.set_title('Abstention vs Fabrication', fontweight='bold')
    ax.set_ylim(0, 1.3)

    # Panel 3: Training loss
    ax = axes[2]
    ax.plot(losses, 'b-o', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('UAlign DPO Training Loss', fontweight='bold')

    fig.suptitle("Phase 133: UAlign Blackhole - Can DPO's 100% Suppression Teach Abstention?",
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase133_ualign.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 133] Complete.")

if __name__ == '__main__':
    main()
