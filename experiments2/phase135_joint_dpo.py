# -*- coding: utf-8 -*-
"""
Phase 135: Autopoietic Joint-DPO (Truth + Silence Simultaneously)
P134 showed sequential DPO then UAlign causes overwriting.
Solution: Mix DPO pairs and UAlign pairs in a SINGLE batch,
optimizing a joint loss so both behaviors emerge simultaneously.

Loss = L_dpo(known facts) + lambda * L_ualign(unknown facts)

With Embedding Surgery pre-applied (P130b proven).

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

# DPO pairs (known facts)
DPO_PAIRS = [
    ("The capital of Japan is", " Tokyo", " Osaka"),
    ("The capital of France is", " Paris", " Lyon"),
    ("The capital of Germany is", " Berlin", " Munich"),
    ("The capital of Italy is", " Rome", " Milan"),
    ("Water freezes at", " 0", " 100"),
    ("The boiling point of water is", " 100", " 212"),
    ("The atomic number of carbon is", " 6", " 12"),
]

# UAlign pairs (uncertain -> abstain)
UALIGN_PAIRS = [
    ("The melting point of iron is", " I", " 500"),
    ("The atomic number of nitrogen is", " I", " 14"),
    ("The distance to the moon in km is", " I", " 500"),
    ("The population of Tokyo in millions is", " I", " 50"),
]

# Test set
TEST_SET = [
    ("The capital of the United Kingdom is", " London", "word", "known"),
    ("The largest planet is", " Jupiter", "word", "known"),
    ("Water freezes at", " 0", "number", "trained"),
    ("The boiling point of water is", " 100", "number", "trained"),
    ("The atomic number of carbon is", " 6", "number", "trained"),
    ("The melting point of iron is", " 15", "number", "uncertain"),
    ("The population of Tokyo in millions is", " 14", "number", "uncertain"),
    ("A year has", " 365", "number", "novel"),
    ("Pi is approximately", " 3", "number", "novel"),
    ("The number of continents is", " 7", "number", "novel"),
    ("The largest ocean is the", " Pacific", "word", "novel"),
    ("The chemical symbol for gold is", " Au", "word", "novel"),
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
    vecs = embed[ids].clone()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += strength * direction * embed[idx].norm()


def evaluate_full(model, tok, test_set):
    results = []
    for prompt, expected, cat, split in test_set:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        exp_id = tok.encode(expected)[-1]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        pred_id = logits.argmax().item()
        pred_tok = tok.decode([pred_id])
        is_correct = (pred_id == exp_id)
        is_abstain = pred_tok.strip() in ['I', 'It', 'Unknown', 'N', 'None']
        results.append({
            'prompt': prompt[:45], 'cat': cat, 'split': split,
            'correct': is_correct, 'is_abstain': is_abstain,
            'pred': pred_tok.encode('ascii', 'replace').decode().strip(),
        })
    return results


def main():
    print("[P135] Autopoietic Joint-DPO")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    lambdas = [0.0, 0.3, 0.5, 1.0, 2.0]
    all_results = {}

    for lam in lambdas:
        print(f"\n  === lambda = {lam} ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
        ref = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
        disperse_embeddings(model, tok, strength=1.0)
        disperse_embeddings(ref, tok, strength=1.0)

        for name, p in model.named_parameters():
            p.requires_grad = False
            for i in range(boundary, n_layers):
                if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=5e-6)

        losses = []
        for epoch in range(10):
            epoch_loss = 0
            # Joint batch: interleave DPO and UAlign
            all_pairs = [(p, c, r, 'dpo') for p, c, r in DPO_PAIRS] + \
                        [(p, c, r, 'ualign') for p, c, r in UALIGN_PAIRS]
            import random as rnd
            rnd.shuffle(all_pairs)
            for prompt, chosen, rejected, pair_type in all_pairs:
                loss = dpo_loss(model, ref, tok, prompt, chosen, rejected)
                if pair_type == 'ualign':
                    loss = loss * lam
                if loss.item() > 0:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(all_pairs))

        model.eval()
        r = evaluate_full(model, tok, TEST_SET)

        # Summarize
        n_correct = sum(x['correct'] for x in r)
        n_abstain = sum(x['is_abstain'] for x in r)
        n_total = len(r)
        word_correct = sum(x['correct'] for x in r if x['cat'] == 'word')
        word_total = sum(1 for x in r if x['cat'] == 'word')
        num_correct = sum(x['correct'] for x in r if x['cat'] == 'number')
        num_abstain = sum(x['is_abstain'] for x in r if x['cat'] == 'number')
        num_total = sum(1 for x in r if x['cat'] == 'number')
        uncertain_abstain = sum(x['is_abstain'] for x in r if x['split'] == 'uncertain')
        uncertain_total = sum(1 for x in r if x['split'] == 'uncertain')

        print(f"    Total: {n_correct}/{n_total} correct, {n_abstain}/{n_total} abstain")
        print(f"    Word: {word_correct}/{word_total}, Num: {num_correct}/{num_total}, "
              f"Uncertain abstain: {uncertain_abstain}/{uncertain_total}")

        all_results[lam] = {
            'total_correct': n_correct/n_total, 'total_abstain': n_abstain/n_total,
            'word_acc': word_correct/word_total if word_total else 0,
            'num_correct': num_correct/num_total if num_total else 0,
            'num_abstain': num_abstain/num_total if num_total else 0,
            'uncertain_abstain': uncertain_abstain/uncertain_total if uncertain_total else 0,
            'losses': losses, 'details': r,
        }
        del model, ref; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save
    out = {'phase': '135', 'name': 'Autopoietic Joint-DPO'}
    for lam, v in all_results.items():
        out[f'lambda_{lam}'] = {k: val for k, val in v.items() if k != 'details'}
    with open(os.path.join(RESULTS_DIR, 'phase135_joint_dpo.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    lams = sorted(all_results.keys())

    ax = axes[0]
    ax.plot(lams, [all_results[l]['word_acc'] for l in lams], 'b-o', label='Word Acc', lw=2)
    ax.plot(lams, [all_results[l]['num_correct'] for l in lams], 'r-s', label='Num Correct', lw=2)
    ax.plot(lams, [all_results[l]['num_abstain'] for l in lams], 'g-^', label='Num Abstain', lw=2)
    ax.set_xlabel('UAlign Weight (lambda)'); ax.set_ylabel('Rate')
    ax.legend(); ax.set_title('Joint Loss: Accuracy vs Lambda', fontweight='bold')
    ax.set_ylim(-0.05, 1.1)

    ax = axes[1]
    ax.plot(lams, [all_results[l]['uncertain_abstain'] for l in lams], 'g-o', lw=2)
    ax.set_xlabel('Lambda'); ax.set_ylabel('Uncertain Abstention Rate')
    ax.set_title('Uncertain Number Abstention', fontweight='bold')
    ax.set_ylim(-0.05, 1.1)

    ax = axes[2]
    for lam in lams:
        if 'losses' in all_results[lam]:
            ax.plot(all_results[lam]['losses'], label=f'lam={lam}', lw=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.set_title('Training Loss', fontweight='bold')

    fig.suptitle('Phase 135: Joint DPO+UAlign - Can We Learn Truth AND Silence Simultaneously?',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase135_joint_dpo.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 135] Complete.")

if __name__ == '__main__':
    main()
