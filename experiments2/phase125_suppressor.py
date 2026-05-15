# -*- coding: utf-8 -*-
"""
Phase 125: Suppressor Token Identification
P121 showed DPO's DeltaW principal direction is nearly orthogonal to
fact vectors. But WHAT direction IS it? What tokens does it promote/suppress?

Method: Project DeltaW's principal direction through the unembedding matrix
to see which tokens are most affected by DPO's weight change.

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
    print("[P125] Suppressor Token Identification")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Save base weights
    print("  Step 1: Loading base model weights...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    base_weights = {}
    for i in range(boundary, n_layers):
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"L{i}.{proj}"
            w = getattr(base_model.model.layers[i].mlp, proj).weight.detach().cpu().float()
            base_weights[key] = w.clone()

    # Get unembedding matrix (lm_head)
    unembed = base_model.lm_head.weight.detach().cpu().float()  # (vocab, hidden)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train DPO
    print("  Step 2: Training DPO...")
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in dpo_model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
    dpo_model.eval()

    # Extract DeltaW and project through unembedding
    print("  Step 3: Analyzing suppressor tokens...")
    all_results = {}

    for i in range(boundary, n_layers):
        proj = 'down_proj'  # Focus on down_proj (hidden -> hidden)
        key = f"L{i}.{proj}"
        w_dpo = getattr(dpo_model.model.layers[i].mlp, proj).weight.detach().cpu().float()
        delta_w = w_dpo - base_weights[key]

        # SVD of DeltaW
        U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
        principal = U[:, 0]  # Top singular vector in output (hidden) space

        # Project through unembedding: which tokens are affected?
        # logit_change = unembed @ principal (shape: vocab)
        logit_change = unembed @ principal

        # Top promoted tokens (positive logit change)
        top_k = 20
        promoted_indices = logit_change.topk(top_k).indices.tolist()
        promoted_values = logit_change.topk(top_k).values.tolist()
        promoted_tokens = []
        for idx, val in zip(promoted_indices, promoted_values):
            try:
                token_str = tok.decode([idx])
            except Exception:
                token_str = f"<{idx}>"
            promoted_tokens.append({'token': token_str, 'id': idx, 'value': val})

        # Top suppressed tokens (negative logit change)
        suppressed_indices = (-logit_change).topk(top_k).indices.tolist()
        suppressed_values = logit_change[suppressed_indices].tolist()
        suppressed_tokens = []
        for idx, val in zip(suppressed_indices, suppressed_values):
            try:
                token_str = tok.decode([idx])
            except Exception:
                token_str = f"<{idx}>"
            suppressed_tokens.append({'token': token_str, 'id': idx, 'value': val})

        # Check: are the DPO training answers in promoted set?
        chosen_tokens = set()
        for _, chosen, _ in DPO_PAIRS:
            ids = tok.encode(chosen)
            chosen_tokens.update(ids)

        rejected_tokens = set()
        for _, _, rejected in DPO_PAIRS:
            ids = tok.encode(rejected)
            rejected_tokens.update(ids)

        promoted_is_chosen = sum(1 for t in promoted_tokens if t['id'] in chosen_tokens)
        suppressed_is_rejected = sum(1 for t in suppressed_tokens if t['id'] in rejected_tokens)

        layer_result = {
            'promoted': promoted_tokens,
            'suppressed': suppressed_tokens,
            'promoted_overlap_chosen': promoted_is_chosen,
            'suppressed_overlap_rejected': suppressed_is_rejected,
            'delta_norm': delta_w.norm().item(),
            'top_sv': S[0].item(),
        }
        all_results[f"L{i}"] = layer_result

        # Safe print (avoid cp932 issues)
        def safe_repr(s):
            return repr(s.encode('ascii', 'replace').decode())
        print(f"\n  L{i} (|dW|={delta_w.norm().item():.4f}):")
        print(f"    TOP PROMOTED:   {', '.join(safe_repr(t['token']) for t in promoted_tokens[:10])}")
        print(f"    TOP SUPPRESSED: {', '.join(safe_repr(t['token']) for t in suppressed_tokens[:10])}")
        print(f"    Promoted in chosen set:   {promoted_is_chosen}/{top_k}")
        print(f"    Suppressed in rejected set: {suppressed_is_rejected}/{top_k}")

    del dpo_model, ref_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save
    out = {'phase': '125', 'name': 'Suppressor Token ID', 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase125_suppressor.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    layers = sorted(all_results.keys())

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        data = all_results[layer]

        # Show top 10 promoted and suppressed
        promoted = data['promoted'][:10]
        suppressed = data['suppressed'][:10]

        tokens = [t['token'][:8] for t in suppressed[::-1]] + ['---'] + [t['token'][:8] for t in promoted]
        values = [t['value'] for t in suppressed[::-1]] + [0] + [t['value'] for t in promoted]
        colors_v = ['#e74c3c' if v < 0 else '#2ecc71' for v in values]

        y_pos = range(len(tokens))
        ax.barh(y_pos, values, color=colors_v)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_xlabel('Logit Change (via Unembedding)')
        ax.set_title(f'{layer}: Tokens Promoted/Suppressed by DPO\n'
                    f'Chosen overlap: {data["promoted_overlap_chosen"]}/20, '
                    f'Rejected overlap: {data["suppressed_overlap_rejected"]}/20',
                    fontweight='bold', fontsize=10)

    fig.suptitle('Phase 125: What Tokens Does DPO Promote and Suppress?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase125_suppressor.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 125] Complete.")

if __name__ == '__main__':
    main()
