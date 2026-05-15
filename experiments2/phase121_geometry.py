# -*- coding: utf-8 -*-
"""
Phase 121: The Geometry of Exorcism
Analyzes WHAT back-only DPO physically does to weight matrices.

Hypothesis: Delta_W (DPO - Base) is anti-aligned with the
"Suppressor direction" and orthogonal to the "Fact direction".
i.e., DPO doesn't teach new facts, it subtracts Grammar Police vectors.

Method:
1. Train back-only DPO (lr=5e-6, optimal)
2. Extract Delta_W for each back layer's MLP
3. Compute "fact vectors" via Logit Lens at each layer
4. Compute cosine similarity between Delta_W principal components
   and the fact/suppressor vectors.

Model: Qwen2.5-0.5B (GPU, float32)
"""
import torch, json, os, gc, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
    """DPO loss (float32)."""
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


def get_fact_direction(model, tok, prompt, answer, layer_idx):
    """Get the 'fact direction' = hidden state at layer when answering correctly."""
    hidden = {}
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.dim() == 3:
            hidden['h'] = h[:, -1, :].detach().float()
        else:
            hidden['h'] = h[-1, :].detach().float().unsqueeze(0)
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model(**inp)
    handle.remove()
    return hidden['h'].squeeze().cpu()


def main():
    print("[P121] The Geometry of Exorcism")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)  # 22
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Step 1: Save base model weights for back layers
    print("  Step 1: Saving base model back-layer MLP weights...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).to(DEVICE)

    base_weights = {}
    for i in range(boundary, n_layers):
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"layers.{i}.mlp.{proj_name}"
            w = getattr(base_model.model.layers[i].mlp, proj_name).weight.detach().cpu().float()
            base_weights[key] = w.clone()

    # Get fact directions at each back layer (using base model)
    print("  Step 2: Computing fact directions...")
    fact_directions = {}
    for i in range(boundary, n_layers):
        layer_dirs = []
        for prompt, chosen, _ in DPO_PAIRS[:5]:  # Subset for speed
            d = get_fact_direction(base_model, tok, prompt, chosen, i)
            layer_dirs.append(d)
        # Average fact direction
        fact_dir = torch.stack(layer_dirs).mean(dim=0)
        fact_dir = fact_dir / fact_dir.norm()
        fact_directions[i] = fact_dir

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3: Train DPO model
    print("  Step 3: Training back-only DPO (lr=5e-6)...")
    dpo_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).eval().to(DEVICE)

    dpo_model.train()
    for name, param in dpo_model.named_parameters():
        param.requires_grad = False
        for i in range(boundary, n_layers):
            if f"layers.{i}." in name and "mlp" in name:
                param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in dpo_model.parameters() if p.requires_grad], lr=5e-6)

    for epoch in range(5):
        epoch_loss = 0
        for prompt, chosen, rejected in DPO_PAIRS:
            loss = dpo_loss(dpo_model, ref_model, tok, prompt, chosen, rejected)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in dpo_model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"    Epoch {epoch}: loss={epoch_loss/len(DPO_PAIRS):.4f}")

    dpo_model.eval()

    # Step 4: Extract Delta_W and analyze geometry
    print("  Step 4: Analyzing weight geometry...")
    geometry_results = {}

    for i in range(boundary, n_layers):
        layer_result = {}
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"layers.{i}.mlp.{proj_name}"
            w_base = base_weights[key]
            w_dpo = getattr(dpo_model.model.layers[i].mlp, proj_name).weight.detach().cpu().float()
            delta_w = w_dpo - w_base

            # Frobenius norm of change
            delta_norm = delta_w.norm().item()
            base_norm = w_base.norm().item()
            relative_change = delta_norm / base_norm if base_norm > 0 else 0

            # SVD of Delta_W to get principal direction
            try:
                U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
                principal_direction = U[:, 0]  # Top singular vector (output space)
                top_singular_value = S[0].item()
                # Rank-1 fraction: how much of the change is explained by top SV?
                rank1_fraction = (S[0]**2 / (S**2).sum()).item() if S.sum() > 0 else 0
            except Exception:
                principal_direction = torch.zeros(delta_w.shape[0])
                top_singular_value = 0
                rank1_fraction = 0

            # Cosine with fact direction
            fact_dir = fact_directions[i]
            if proj_name == 'down_proj':
                # down_proj maps from intermediate to hidden, so output dim = hidden
                cos_fact = F.cosine_similarity(
                    principal_direction[:len(fact_dir)].unsqueeze(0),
                    fact_dir[:len(principal_direction)].unsqueeze(0)
                ).item() if len(principal_direction) == len(fact_dir) else 0
            else:
                cos_fact = 0  # gate/up project to intermediate dim, can't directly compare

            layer_result[proj_name] = {
                'delta_norm': delta_norm,
                'base_norm': base_norm,
                'relative_change': relative_change,
                'top_singular_value': top_singular_value,
                'rank1_fraction': rank1_fraction,
                'cos_with_fact': cos_fact,
            }
            print(f"    L{i}.{proj_name}: |dW|={delta_norm:.4f} ({relative_change:.2%} of base), "
                  f"rank1={rank1_fraction:.1%}, cos(fact)={cos_fact:.3f}")

        geometry_results[f"L{i}"] = layer_result

    # Get DPO directions at each back layer
    print("  Step 5: Computing DPO fact directions (post-training)...")
    dpo_fact_dirs = {}
    for i in range(boundary, n_layers):
        layer_dirs = []
        for prompt, chosen, _ in DPO_PAIRS[:5]:
            d = get_fact_direction(dpo_model, tok, prompt, chosen, i)
            layer_dirs.append(d)
        dpo_dir = torch.stack(layer_dirs).mean(dim=0)
        dpo_dir = dpo_dir / dpo_dir.norm()
        dpo_fact_dirs[i] = dpo_dir

        cos_base_dpo = F.cosine_similarity(
            fact_directions[i].unsqueeze(0), dpo_dir.unsqueeze(0)).item()
        print(f"    L{i}: cos(base_fact, dpo_fact) = {cos_base_dpo:.4f}")
        geometry_results[f"L{i}"]['cos_base_vs_dpo_fact'] = cos_base_dpo

    del dpo_model, ref_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save
    out = {
        'phase': '121', 'name': 'Geometry of Exorcism',
        'model': model_id, 'boundary': boundary,
        'geometry': geometry_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase121_geometry.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    layers = sorted([k for k in geometry_results.keys()])
    projs = ['gate_proj', 'up_proj', 'down_proj']

    # Panel 1: Relative weight change
    ax = axes[0, 0]
    for proj in projs:
        vals = [geometry_results[l][proj]['relative_change'] for l in layers]
        ax.bar([f'{l}\n{proj[:4]}' for l in layers], vals, alpha=0.7, label=proj)
    ax.set_ylabel('|Delta_W| / |W_base|')
    ax.set_title('Relative Weight Change by Layer & Projection')
    ax.legend(fontsize=8)

    # Panel 2: Rank-1 fraction (how "surgical" is the DPO edit?)
    ax = axes[0, 1]
    for proj in projs:
        vals = [geometry_results[l][proj]['rank1_fraction'] for l in layers]
        ax.bar([f'{l}\n{proj[:4]}' for l in layers], vals, alpha=0.7, label=proj)
    ax.set_ylabel('Top-1 SVD Fraction')
    ax.set_title('Rank-1 Concentration (Higher = More Surgical)')
    ax.legend(fontsize=8)

    # Panel 3: Cosine with fact direction (down_proj only)
    ax = axes[1, 0]
    cos_vals = [geometry_results[l]['down_proj']['cos_with_fact'] for l in layers]
    colors_cos = ['#e74c3c' if v < 0 else '#2ecc71' for v in cos_vals]
    ax.bar(layers, cos_vals, color=colors_cos)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Delta_W Principal Direction vs Fact Direction\n(down_proj only)')
    for i, v in enumerate(cos_vals):
        ax.text(i, v + 0.02 * (1 if v >= 0 else -1), f'{v:.3f}',
               ha='center', fontsize=10, fontweight='bold')

    # Panel 4: Fact direction stability (base vs DPO)
    ax = axes[1, 1]
    cos_base_dpo = [geometry_results[l].get('cos_base_vs_dpo_fact', 0) for l in layers]
    ax.bar(layers, cos_base_dpo, color='#3498db')
    ax.set_ylabel('Cosine(Base Fact Dir, DPO Fact Dir)')
    ax.set_title('Fact Direction Stability After DPO\n(1.0 = unchanged, 0.0 = rotated)')
    ax.set_ylim(-0.1, 1.1)
    for i, v in enumerate(cos_base_dpo):
        ax.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('Phase 121: Geometry of Exorcism - What Does DPO Physically Do?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase121_geometry.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 121] Complete.")


if __name__ == '__main__':
    main()
