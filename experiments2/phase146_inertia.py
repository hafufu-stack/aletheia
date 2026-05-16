# -*- coding: utf-8 -*-
"""
Phase 146: DPO Inertia Law
Does 1.5B need MORE DPO data to overcome inertia?

Hypothesis: 1.5B has more parameters = more "inertia".
0.5B worked with 4 DPO pairs * 5 epochs = 20 gradient steps.
1.5B might need 10x-50x more gradient steps.

Test: Scale DPO data from 5 pairs to 50 pairs to 200 pairs.

Model: Qwen2.5-1.5B (GPU, float16)
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

# Base DPO pairs (same as S31)
BASE_PAIRS = [
    ("Water freezes at", " 0", " 100"),
    ("The boiling point of water is", " 100", " 212"),
    ("The atomic number of carbon is", " 6", " 12"),
    ("The speed of light is approximately", " 299", " 186"),
]

# Extended DPO pairs - paraphrases and augmentations
EXTENDED_PAIRS = [
    ("The freezing point of water is", " 0", " 32"),
    ("Water turns to ice at", " 0", " 100"),
    ("At what temperature does water boil?", " 100", " 212"),
    ("Boiling water temperature is", " 100", " 50"),
    ("Carbon has atomic number", " 6", " 14"),
    ("The element carbon has", " 6", " 8"),
    ("Light speed is about", " 299", " 186"),
    ("The velocity of light equals", " 299", " 300"),
    ("A year has how many days?", " 365", " 360"),
    ("Days in a year:", " 365", " 400"),
    ("The number of continents is", " 7", " 6"),
    ("There are how many continents?", " 7", " 5"),
    ("Pi equals approximately", " 3", " 4"),
    ("The value of pi is about", " 3", " 2"),
    ("One kilometer equals", " 1000", " 100"),
    ("A meter has how many centimeters?", " 100", " 10"),
    ("Hydrogen atomic number is", " 1", " 2"),
    ("The atomic number of oxygen is", " 8", " 16"),
    ("Nitrogen atomic number:", " 7", " 14"),
    ("Helium atomic number:", " 2", " 4"),
    ("How many sides does a triangle have?", " 3", " 4"),
    ("A square has how many sides?", " 4", " 5"),
    ("A hexagon has how many sides?", " 6", " 8"),
    ("An octagon has how many sides?", " 8", " 6"),
    ("How many minutes in an hour?", " 60", " 100"),
    ("Seconds in a minute:", " 60", " 30"),
    ("Hours in a day:", " 24", " 12"),
    ("Days in a week:", " 7", " 5"),
    ("Months in a year:", " 12", " 10"),
    ("Weeks in a year:", " 52", " 48"),
    ("The acceleration due to gravity is about", " 9", " 10"),
    ("Absolute zero is approximately minus", " 273", " 459"),
    ("How many states in the US?", " 50", " 52"),
    ("Chromosomes in a human cell:", " 46", " 48"),
    ("Bones in the human body:", " 206", " 200"),
    ("Teeth in an adult human:", " 32", " 28"),
    ("The melting point of iron is about", " 1538", " 1000"),
    ("How many planets in the solar system?", " 8", " 9"),
    ("Legs on a spider:", " 8", " 6"),
    ("Legs on an insect:", " 6", " 8"),
    ("Eyes on a human:", " 2", " 1"),
    ("Chambers in the heart:", " 4", " 2"),
    ("The pH of pure water is", " 7", " 6"),
    ("Room temperature is approximately", " 20", " 25"),
    ("The number of amino acids is", " 20", " 22"),
    ("Protons in a hydrogen atom:", " 1", " 0"),
]

EVAL_SET = [
    ("Water freezes at", " 0"),
    ("The boiling point of water is", " 100"),
    ("The atomic number of carbon is", " 6"),
    ("The speed of light is approximately", " 299"),
    ("A year has", " 365"),
    ("Pi is approximately", " 3"),
    ("The number of continents is", " 7"),
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


def disperse_embeddings(model, tok, strength=1.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366"]
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in num_tokens]
    ids = list(set(ids))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)


def train_and_eval(model, ref, tok, dpo_pairs, epochs=5):
    n_layers = 28
    boundary = int(n_layers * 0.94)
    for name, p in model.named_parameters():
        p.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable: return 0.0, 0
    opt = torch.optim.AdamW(trainable, lr=2e-5)
    total_steps = 0
    for epoch in range(epochs):
        for prompt, chosen, rejected in dpo_pairs:
            loss = dpo_loss(model, ref, tok, prompt, chosen, rejected)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            total_steps += 1
    model.eval()
    correct = 0
    for prompt, expected in EVAL_SET:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        exp_ids = tok.encode(expected)
        exp_id = exp_ids[-1]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if logits.argmax().item() == exp_id: correct += 1
    return correct / len(EVAL_SET), total_steps


def main():
    print("[P146] DPO Inertia Law")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-1.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Data size conditions
    conditions = [
        ('4 pairs x 5ep', BASE_PAIRS, 5),
        ('4 pairs x 20ep', BASE_PAIRS, 20),
        ('4 pairs x 50ep', BASE_PAIRS, 50),
        ('50 pairs x 5ep', BASE_PAIRS + EXTENDED_PAIRS[:46], 5),
        ('50 pairs x 10ep', BASE_PAIRS + EXTENDED_PAIRS[:46], 10),
    ]

    results = []
    for name, pairs, epochs in conditions:
        print(f"\n  === {name} ({len(pairs)} pairs, {epochs} epochs) ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float16).to(DEVICE)
        ref = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float16).eval().to(DEVICE)
        # Apply surgery
        disperse_embeddings(model, tok, strength=1.0)
        disperse_embeddings(ref, tok, strength=1.0)
        acc, steps = train_and_eval(model, ref, tok, pairs, epochs)
        print(f"    Steps: {steps}, Accuracy: {acc:.0%}")
        results.append({'name': name, 'n_pairs': len(pairs), 'epochs': epochs,
                       'steps': steps, 'acc': acc})
        del model, ref; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase146_inertia.json'), 'w') as f:
        json.dump({'phase': '146', 'name': 'DPO Inertia Law',
                   'results': results}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    steps_vals = [r['steps'] for r in results]
    acc_vals = [r['acc'] for r in results]
    labels = [r['name'] for r in results]
    ax.plot(steps_vals, acc_vals, 'b-o', lw=2.5, markersize=12)
    for i, (s, a, l) in enumerate(zip(steps_vals, acc_vals, labels)):
        ax.annotate(f'{l}\n{a:.0%}', (s, a), fontsize=9, ha='center',
                   va='bottom' if a > 0 else 'top',
                   xytext=(0, 10 if a > 0 else -15), textcoords='offset points')
    ax.axhline(y=0, color='red', ls='--', alpha=0.5, label='P139 result (0%)')
    # Mark 0.5B reference
    ax.axhline(y=1.0, color='green', ls=':', alpha=0.5, label='0.5B Ultimate (100%)')
    ax.axvline(x=20, color='orange', ls=':', alpha=0.5, label='0.5B: 4 pairs x 5ep = 20 steps')
    ax.set_xlabel('Total DPO Gradient Steps', fontsize=12)
    ax.set_ylabel('Number Accuracy', fontsize=12)
    ax.set_title('Phase 146: DPO Inertia Law\nDoes 1.5B need more DPO data?',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase146_inertia.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    if any(r['acc'] > 0 for r in results):
        best = max(results, key=lambda r: r['acc'])
        print(f"  -> INERTIA LAW CONFIRMED: DPO works with {best['steps']} steps ({best['acc']:.0%})")
    else:
        print(f"  -> 1.5B is immune even with {max(r['steps'] for r in results)} DPO steps")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 146] Complete.")

if __name__ == '__main__':
    main()
