# -*- coding: utf-8 -*-
"""
Phase 130b: Embedding Dispersion Surgery
P129 proved: numerical tokens have 9x higher embedding clustering (cos=0.73).
P130 proved: Knowledge LoRA can't fix this through FFN injection.

Root cause hypothesis: DPO fails on numbers because embeddings are too close.
If we SURGICALLY SPREAD the numerical token embeddings apart, does DPO
suddenly start working?

Experiment:
1. Measure baseline embedding clustering for number tokens
2. Apply orthogonal dispersion: rotate numerical embeddings to be more separated
3. Re-run DPO on the modified model
4. Check if numerical fact accuracy improves

This is a direct causal test of P129's geometric hypothesis.

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

TRAIN_PAIRS = [
    ("The capital of Japan is", " Tokyo", " Osaka", "word"),
    ("The capital of France is", " Paris", " Lyon", "word"),
    ("The capital of Germany is", " Berlin", " Munich", "word"),
    ("The capital of Italy is", " Rome", " Milan", "word"),
    ("The capital of Spain is", " Madrid", " Barcelona", "word"),
    ("Water freezes at", " 0", " 100", "number"),
    ("The boiling point of water is", " 100", " 212", "number"),
    ("The atomic number of carbon is", " 6", " 12", "number"),
    ("The speed of light is approximately", " 299", " 186", "number"),
]

TEST_PAIRS = [
    ("The capital of the United Kingdom is", " London", " Manchester", "word"),
    ("The largest planet is", " Jupiter", " Saturn", "word"),
    ("The number of planets in the solar system is", " 8", " 9", "number"),
    ("A year has", " 365", " 366", "number"),
    ("The atomic number of oxygen is", " 8", " 16", "number"),
]


def pairwise_cos(embed, ids):
    if len(ids) < 2:
        return 0.0
    embs = embed[ids]
    cos = F.cosine_similarity(embs.unsqueeze(0), embs.unsqueeze(1), dim=-1)
    mask = torch.triu(torch.ones(len(ids), len(ids)), diagonal=1).bool()
    return cos[mask].mean().item()


def disperse_embeddings(model, tok, target_tokens, strength=0.5):
    """Spread numerical token embeddings apart using orthogonal perturbation."""
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in target_tokens]

    # Get current embeddings for target tokens
    vecs = embed[ids].clone()
    center = vecs.mean(dim=0)

    # Compute directions away from center
    diffs = vecs - center.unsqueeze(0)

    # Apply Gram-Schmidt-like orthogonalization to maximize separation
    # Then scale and add perturbation
    for i, idx in enumerate(ids):
        direction = diffs[i]
        direction = direction / (direction.norm() + 1e-8)
        # Add perturbation in the direction away from cluster center
        perturbation = strength * direction * embed[idx].norm()
        embed[idx] += perturbation

    return ids


def run_dpo(model, ref_model, tok, pairs, lr=5e-6, epochs=5, n_layers=24, boundary=22):
    """Standard DPO training on specified pairs."""
    for name, param in model.named_parameters():
        param.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name:
                param.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    for epoch in range(epochs):
        for prompt, chosen, rejected, _ in pairs:
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
            diff = 0.05 * ((lp_c.sum() - rlp_c.sum()) - (lp_r.sum() - rlp_r.sum()))
            loss = -F.logsigmoid(diff)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
    model.eval()
    return model


def evaluate(model, tok, pairs):
    results = {'word_correct': 0, 'word_total': 0, 'num_correct': 0, 'num_total': 0,
               'details': []}
    for prompt, chosen, rejected, cat in pairs:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        c_id = tok.encode(chosen)[-1]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        pred = logits.argmax().item()
        correct = (pred == c_id)
        if cat == 'word':
            results['word_total'] += 1; results['word_correct'] += int(correct)
        else:
            results['num_total'] += 1; results['num_correct'] += int(correct)
        results['details'].append({
            'prompt': prompt[:40], 'cat': cat, 'correct': correct,
            'pred': tok.decode([pred]).encode('ascii', 'replace').decode().strip(),
        })
    results['word_acc'] = results['word_correct']/results['word_total'] if results['word_total'] else 0
    results['num_acc'] = results['num_correct']/results['num_total'] if results['num_total'] else 0
    return results


def main():
    print("[P130b] Embedding Dispersion Surgery")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Numerical tokens to disperse
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366"]

    all_results = {}
    strengths = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

    for strength in strengths:
        print(f"\n  === Dispersion Strength: {strength} ===")

        # Fresh model for each strength
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

        # Measure initial clustering
        embed = model.model.embed_tokens.weight.detach().cpu().float()
        num_ids = [tok.encode(t)[-1] for t in num_tokens]
        word_ids = [tok.encode(t)[-1] for t in [" Tokyo", " Paris", " Berlin", " Rome", " Madrid",
                                                  " Osaka", " Lyon", " Munich", " Milan", " Barcelona"]]
        cos_before = pairwise_cos(embed, num_ids)

        # Apply dispersion
        if strength > 0:
            disperse_embeddings(model, tok, num_tokens, strength)
            # Also disperse in ref_model to keep DPO reference consistent
            disperse_embeddings(ref_model, tok, num_tokens, strength)

        embed_after = model.model.embed_tokens.weight.detach().cpu().float()
        cos_after = pairwise_cos(embed_after, num_ids)
        word_cos = pairwise_cos(embed_after, word_ids)

        print(f"    Num clustering: {cos_before:.4f} -> {cos_after:.4f}")
        print(f"    Word clustering: {word_cos:.4f}")

        # Pre-DPO baseline
        model.eval()
        pre_dpo = evaluate(model, tok, TRAIN_PAIRS)
        print(f"    Pre-DPO: word={pre_dpo['word_acc']:.0%} num={pre_dpo['num_acc']:.0%}")

        # Run DPO
        model.train()
        model = run_dpo(model, ref_model, tok, TRAIN_PAIRS, lr=5e-6, epochs=5,
                       n_layers=n_layers, boundary=boundary)

        # Post-DPO evaluation
        post_train = evaluate(model, tok, TRAIN_PAIRS)
        post_test = evaluate(model, tok, TEST_PAIRS)
        print(f"    Post-DPO train: word={post_train['word_acc']:.0%} num={post_train['num_acc']:.0%}")
        print(f"    Post-DPO test:  word={post_test['word_acc']:.0%} num={post_test['num_acc']:.0%}")

        all_results[strength] = {
            'cos_before': cos_before, 'cos_after': cos_after, 'word_cos': word_cos,
            'pre_dpo': {'word': pre_dpo['word_acc'], 'num': pre_dpo['num_acc']},
            'post_train': {'word': post_train['word_acc'], 'num': post_train['num_acc']},
            'post_test': {'word': post_test['word_acc'], 'num': post_test['num_acc']},
        }

        del model, ref_model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save
    out = {'phase': '130b', 'name': 'Embedding Dispersion Surgery',
           'results': {str(k): v for k, v in all_results.items()}}
    with open(os.path.join(RESULTS_DIR, 'phase130b_embedding_surgery.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Embedding clustering vs dispersion strength
    ax = axes[0]
    ss = sorted(all_results.keys())
    cos_vals = [all_results[s]['cos_after'] for s in ss]
    word_cos_vals = [all_results[s]['word_cos'] for s in ss]
    ax.plot(ss, cos_vals, 'r-o', label='Number tokens', linewidth=2)
    ax.plot(ss, word_cos_vals, 'b-s', label='Word tokens', linewidth=2)
    ax.set_xlabel('Dispersion Strength')
    ax.set_ylabel('Pairwise Cosine Similarity')
    ax.legend(); ax.set_title('Embedding Clustering', fontweight='bold')

    # Panel 2: DPO accuracy on numbers vs dispersion
    ax = axes[1]
    pre_nums = [all_results[s]['pre_dpo']['num'] for s in ss]
    post_nums = [all_results[s]['post_train']['num'] for s in ss]
    ax.plot(ss, pre_nums, 'gray', marker='o', linestyle='--', label='Pre-DPO (num)', linewidth=2)
    ax.plot(ss, post_nums, 'r-s', label='Post-DPO (num)', linewidth=2)
    ax.set_xlabel('Dispersion Strength')
    ax.set_ylabel('Numerical Accuracy')
    ax.legend(); ax.set_title('DPO Effectiveness on Numbers', fontweight='bold')
    ax.set_ylim(-0.05, 1.1)

    # Panel 3: Word accuracy (safety check)
    ax = axes[2]
    pre_words = [all_results[s]['pre_dpo']['word'] for s in ss]
    post_words = [all_results[s]['post_train']['word'] for s in ss]
    ax.plot(ss, pre_words, 'gray', marker='o', linestyle='--', label='Pre-DPO (word)', linewidth=2)
    ax.plot(ss, post_words, 'b-s', label='Post-DPO (word)', linewidth=2)
    ax.set_xlabel('Dispersion Strength')
    ax.set_ylabel('Word Accuracy')
    ax.legend(); ax.set_title('Word Accuracy (Safety)', fontweight='bold')
    ax.set_ylim(-0.05, 1.1)

    fig.suptitle('Phase 130b: Can Spreading Numerical Embeddings Apart Make DPO Work on Numbers?',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase130b_embedding_surgery.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    baseline_num = all_results[0.0]['post_train']['num']
    best_strength = max(ss, key=lambda s: all_results[s]['post_train']['num'])
    best_num = all_results[best_strength]['post_train']['num']
    print(f"    DPO baseline num acc: {baseline_num:.0%}")
    print(f"    Best dispersed num acc: {best_num:.0%} (strength={best_strength})")
    print(f"    Improvement: {best_num - baseline_num:+.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 130b] Complete.")

if __name__ == '__main__':
    main()
