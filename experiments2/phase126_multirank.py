# -*- coding: utf-8 -*-
"""
Phase 126: Multi-Rank DPO Decomposition
P125 showed rank-1 SVD promotes WRONG tokens. But rank-1 explains only ~25%.
Do higher ranks cancel it out? What's the NET effect across all ranks?

Method:
1. Decompose DeltaW into top-k SVD components
2. For each rank, project through unembedding to get per-token logit change
3. Sum across all ranks to get the TOTAL logit change
4. Check if total effect promotes correct or wrong tokens

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
    ("The largest ocean on Earth is the", " Pacific", " Atlantic"),
    ("The speed of light is approximately", " 299", " 186"),
]


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


def main():
    print("[P126] Multi-Rank DPO Decomposition")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Collect chosen/rejected token IDs
    chosen_ids = set()
    rejected_ids = set()
    for _, chosen, rejected in DPO_PAIRS:
        chosen_ids.update(tok.encode(chosen))
        rejected_ids.update(tok.encode(rejected))

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    base_weights = {}
    for i in range(boundary, n_layers):
        key = f"L{i}.down_proj"
        w = base_model.model.layers[i].mlp.down_proj.weight.detach().cpu().float()
        base_weights[key] = w.clone()
    unembed = base_model.lm_head.weight.detach().cpu().float()
    del base_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Train DPO
    print("  Training DPO...")
    dpo_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    dpo_model.train()
    for name, param in dpo_model.named_parameters():
        param.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name:
                param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [p for p in dpo_model.parameters() if p.requires_grad], lr=5e-6)
    for epoch in range(5):
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss(dpo_model, ref_model, tok, prompt, chosen, rejected)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in dpo_model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
    dpo_model.eval()

    # Analyze per-rank logit changes
    print("  Analyzing multi-rank decomposition...")
    all_results = {}

    for i in range(boundary, n_layers):
        key = f"L{i}.down_proj"
        w_dpo = dpo_model.model.layers[i].mlp.down_proj.weight.detach().cpu().float()
        delta_w = w_dpo - base_weights[key]

        U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
        n_ranks = min(20, len(S))

        # Cumulative logit change by rank
        cumulative_logit = torch.zeros(unembed.shape[0])
        rank_data = []

        for rank in range(n_ranks):
            # Rank-k component: s_k * u_k * v_k^T
            # Output space direction: u_k (hidden dim)
            direction = U[:, rank] * S[rank]
            # Project through unembedding
            logit_change = unembed @ direction
            cumulative_logit += logit_change

            # Check chosen/rejected token logit changes
            chosen_mean = np.mean([logit_change[tid].item() for tid in chosen_ids if tid < len(logit_change)])
            rejected_mean = np.mean([logit_change[tid].item() for tid in rejected_ids if tid < len(logit_change)])
            cum_chosen = np.mean([cumulative_logit[tid].item() for tid in chosen_ids if tid < len(cumulative_logit)])
            cum_rejected = np.mean([cumulative_logit[tid].item() for tid in rejected_ids if tid < len(cumulative_logit)])

            rank_data.append({
                'rank': rank,
                'sv': S[rank].item(),
                'sv_fraction': (S[rank]**2 / (S**2).sum()).item(),
                'chosen_logit': chosen_mean,
                'rejected_logit': rejected_mean,
                'cum_chosen_logit': cum_chosen,
                'cum_rejected_logit': cum_rejected,
                'correct_direction': chosen_mean > rejected_mean,
                'cum_correct': cum_chosen > cum_rejected,
            })

        # Full DeltaW (all ranks)
        full_logit = unembed @ delta_w @ torch.ones(delta_w.shape[1])  # simplified
        # Actually, the full effect depends on input, let's use column sum as proxy
        # Better: just check the cumulative at max rank
        final_cum_chosen = rank_data[-1]['cum_chosen_logit']
        final_cum_rejected = rank_data[-1]['cum_rejected_logit']

        # Find the rank where cumulative flips to correct
        flip_rank = None
        for rd in rank_data:
            if rd['cum_correct']:
                flip_rank = rd['rank']
                break

        all_results[f"L{i}"] = {
            'ranks': rank_data,
            'flip_rank': flip_rank,
            'final_correct': rank_data[-1]['cum_correct'],
        }

        print(f"\n  L{i}:")
        print(f"    Rank-0: chosen={rank_data[0]['chosen_logit']:.4f}, "
              f"rejected={rank_data[0]['rejected_logit']:.4f} "
              f"({'CORRECT' if rank_data[0]['correct_direction'] else 'WRONG'})")
        print(f"    Cumulative@20: chosen={final_cum_chosen:.4f}, "
              f"rejected={final_cum_rejected:.4f} "
              f"({'CORRECT' if rank_data[-1]['cum_correct'] else 'WRONG'})")
        if flip_rank is not None:
            print(f"    Flips to correct at rank {flip_rank}")
        else:
            print(f"    NEVER flips to correct in top-20!")

    del dpo_model, ref_model, optimizer; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save
    out = {'phase': '126', 'name': 'Multi-Rank Decomposition', 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase126_multirank.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    layers = sorted(all_results.keys())

    for idx, layer in enumerate(layers):
        data = all_results[layer]
        ranks = [r['rank'] for r in data['ranks']]

        # Per-rank logit change
        ax = axes[0, idx]
        chosen_vals = [r['chosen_logit'] for r in data['ranks']]
        rejected_vals = [r['rejected_logit'] for r in data['ranks']]
        ax.bar([r-0.15 for r in ranks], chosen_vals, 0.3, color='#2ecc71', label='Chosen', alpha=0.8)
        ax.bar([r+0.15 for r in ranks], rejected_vals, 0.3, color='#e74c3c', label='Rejected', alpha=0.8)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_xlabel('SVD Rank')
        ax.set_ylabel('Mean Logit Change')
        ax.set_title(f'{layer}: Per-Rank Logit Change', fontweight='bold')
        ax.legend(fontsize=8)

        # Cumulative logit change
        ax = axes[1, idx]
        cum_chosen = [r['cum_chosen_logit'] for r in data['ranks']]
        cum_rejected = [r['cum_rejected_logit'] for r in data['ranks']]
        ax.plot(ranks, cum_chosen, 'o-', color='#2ecc71', label='Chosen (cumulative)', linewidth=2)
        ax.plot(ranks, cum_rejected, 's-', color='#e74c3c', label='Rejected (cumulative)', linewidth=2)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        if data['flip_rank'] is not None:
            ax.axvline(x=data['flip_rank'], color='gold', linewidth=2, linestyle='--',
                      label=f'Flip at rank {data["flip_rank"]}')
        ax.set_xlabel('SVD Rank (cumulative)')
        ax.set_ylabel('Cumulative Logit Change')
        ax.set_title(f'{layer}: Cumulative Effect', fontweight='bold')
        ax.legend(fontsize=8)

    fig.suptitle('Phase 126: Multi-Rank Decomposition of DPO\n'
                'Does the paradox resolve at higher ranks?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase126_multirank.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 126] Complete.")

if __name__ == '__main__':
    main()
