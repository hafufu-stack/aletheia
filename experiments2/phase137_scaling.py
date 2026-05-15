# -*- coding: utf-8 -*-
"""
Phase 137: Cosmological Dispersion Scaling
Does P130b's embedding surgery work at 1.5B scale?
Measure the critical cosine similarity cos* where DPO starts working.

Deep Think predicts: larger models need LESS dispersion (higher cos*)
because higher-dimensional spaces can distinguish closer vectors.

Model: Qwen2.5-1.5B (GPU, float16 for memory)
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
    ("The capital of Japan is", " Tokyo", " Osaka", "word"),
    ("The capital of France is", " Paris", " Lyon", "word"),
    ("Water freezes at", " 0", " 100", "number"),
    ("The boiling point of water is", " 100", " 212", "number"),
    ("The atomic number of carbon is", " 6", " 12", "number"),
    ("The speed of light is approximately", " 299", " 186", "number"),
]


def dpo_loss_fp32(model, ref_model, tok, prompt, chosen, rejected, beta=0.05):
    text_c = prompt + chosen
    text_r = prompt + rejected
    inp_c = tok(text_c, return_tensors='pt').to(DEVICE)
    inp_r = tok(text_r, return_tensors='pt').to(DEVICE)
    plen = tok(prompt, return_tensors='pt')['input_ids'].shape[1]
    # Cast to float32 for DPO loss stability
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


def measure_clustering(model, tok):
    """Measure numerical token embedding cosine similarity."""
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in num_tokens]
    vecs = embed[ids].float()
    vecs_norm = F.normalize(vecs, dim=-1)
    cos_matrix = vecs_norm @ vecs_norm.T
    mask = ~torch.eye(len(ids), dtype=bool, device=cos_matrix.device)
    return cos_matrix[mask].mean().item()


def disperse_embeddings(model, tok, strength=1.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365"]
    embed = model.model.embed_tokens.weight.data
    ids = [tok.encode(t)[-1] for t in num_tokens]
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)


def main():
    print("[P137] Cosmological Dispersion Scaling (1.5B)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    models_to_test = [
        ('Qwen/Qwen2.5-0.5B', 24, torch.float32),
        ('Qwen/Qwen2.5-1.5B', 28, torch.float16),
    ]
    strengths = [0.0, 0.1, 0.3, 0.5, 1.0]
    all_results = {}

    for model_id, n_layers, dtype in models_to_test:
        short_name = model_id.split('/')[-1]
        print(f"\n  ====== {short_name} (dtype={dtype}) ======")
        model_results = {}
        boundary = int(n_layers * 0.94)

        for strength in strengths:
            print(f"\n    --- strength = {strength} ---")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, local_files_only=True, torch_dtype=dtype).to(DEVICE)
                ref = AutoModelForCausalLM.from_pretrained(
                    model_id, local_files_only=True, torch_dtype=dtype).eval().to(DEVICE)
                tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                if tok.pad_token is None: tok.pad_token = tok.eos_token

                # Measure pre-surgery clustering
                cos_before = measure_clustering(model, tok)

                # Apply surgery
                if strength > 0:
                    disperse_embeddings(model, tok, strength)
                    disperse_embeddings(ref, tok, strength)

                cos_after = measure_clustering(model, tok)
                print(f"      Cos: {cos_before:.4f} -> {cos_after:.4f}")

                # Evaluate pre-DPO
                pre_results = {}
                for prompt, chosen, rejected, cat in DPO_PAIRS:
                    inp = tok(prompt, return_tensors='pt').to(DEVICE)
                    exp_id = tok.encode(chosen)[-1]
                    with torch.no_grad():
                        logits = model(**inp).logits[0, -1, :].float()
                    pred_id = logits.argmax().item()
                    if cat not in pre_results: pre_results[cat] = {'correct': 0, 'total': 0}
                    pre_results[cat]['total'] += 1
                    pre_results[cat]['correct'] += int(pred_id == exp_id)

                pre_word = pre_results.get('word', {}).get('correct', 0) / max(1, pre_results.get('word', {}).get('total', 1))
                pre_num = pre_results.get('number', {}).get('correct', 0) / max(1, pre_results.get('number', {}).get('total', 1))

                # DPO training
                for name, p in model.named_parameters():
                    p.requires_grad = False
                    for i in range(boundary, n_layers):
                        if f"layers.{i}." in name and "mlp" in name: p.requires_grad = True
                trainable = [p for p in model.parameters() if p.requires_grad]

                # Use higher precision for DPO loss even with fp16 model
                opt = torch.optim.AdamW(trainable, lr=5e-6)
                for epoch in range(5):
                    for prompt, chosen, rejected, _ in DPO_PAIRS:
                        loss = dpo_loss_fp32(model, ref, tok, prompt, chosen, rejected)
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        opt.step()

                # Evaluate post-DPO
                model.eval()
                post_results = {}
                for prompt, chosen, rejected, cat in DPO_PAIRS:
                    inp = tok(prompt, return_tensors='pt').to(DEVICE)
                    exp_id = tok.encode(chosen)[-1]
                    with torch.no_grad():
                        logits = model(**inp).logits[0, -1, :].float()
                    pred_id = logits.argmax().item()
                    if cat not in post_results: post_results[cat] = {'correct': 0, 'total': 0}
                    post_results[cat]['total'] += 1
                    post_results[cat]['correct'] += int(pred_id == exp_id)

                post_word = post_results.get('word', {}).get('correct', 0) / max(1, post_results.get('word', {}).get('total', 1))
                post_num = post_results.get('number', {}).get('correct', 0) / max(1, post_results.get('number', {}).get('total', 1))

                print(f"      Pre-DPO: word={pre_word:.0%} num={pre_num:.0%}")
                print(f"      Post-DPO: word={post_word:.0%} num={post_num:.0%}")

                model_results[strength] = {
                    'cos_before': cos_before, 'cos_after': cos_after,
                    'pre_word': pre_word, 'pre_num': pre_num,
                    'post_word': post_word, 'post_num': post_num,
                }

            except Exception as e:
                print(f"      ERROR: {e}")
                model_results[strength] = {'error': str(e)}
            finally:
                del model, ref
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        all_results[short_name] = model_results

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase137_scaling.json'), 'w') as f:
        json.dump({'phase': '137', 'name': 'Dispersion Scaling', 'results': all_results},
                 f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'Qwen2.5-0.5B': '#e74c3c', 'Qwen2.5-1.5B': '#3498db'}
    for model_name, results in all_results.items():
        ss = sorted([s for s in results.keys() if 'error' not in results.get(s, {})])
        cos_vals = [results[s].get('cos_after', 0) for s in ss]
        num_vals = [results[s].get('post_num', 0) for s in ss]
        c = colors.get(model_name, '#95a5a6')
        axes[0].plot(ss, cos_vals, '-o', label=f'{model_name} cos', color=c, lw=2)
        axes[1].plot(ss, num_vals, '-s', label=f'{model_name} num_acc', color=c, lw=2)

    axes[0].set_xlabel('Dispersion Strength'); axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Embedding Cosine After Surgery', fontweight='bold')
    axes[0].legend()
    axes[1].set_xlabel('Dispersion Strength'); axes[1].set_ylabel('Number Accuracy (post-DPO)')
    axes[1].set_title('DPO Number Accuracy', fontweight='bold')
    axes[1].legend()

    fig.suptitle('Phase 137: Dispersion Scaling Law (0.5B vs 1.5B)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase137_scaling.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 137] Complete.")

if __name__ == '__main__':
    main()
