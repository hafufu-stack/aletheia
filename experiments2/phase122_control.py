# -*- coding: utf-8 -*-
"""
Phase 122: Random Perturbation Control for P121
Tests whether P121's cos(DeltaW, fact)~0.05 is meaningful or trivial.

In d=896 dimensions, random unit vectors have expected cosine ~ 0
with std ~ 1/sqrt(d) ~ 0.033. So cos=0.05 is ~1.5 sigma (weak),
but cos=0.115 (L23) is ~3.5 sigma (significant).

Method:
1. Load base model, compute fact directions (same as P121)
2. Generate 100 random perturbations with same Frobenius norm as DPO's DeltaW
3. Compute cosine similarity for each random perturbation
4. Compare DPO's cos vs random distribution: compute z-score and p-value

Model: Qwen2.5-0.5B (GPU, float32)
"""
import torch, json, os, gc, numpy as np, time
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

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

N_RANDOM = 100  # Number of random perturbations


def get_fact_direction(model, tok, prompt, layer_idx):
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
    print("[P122] Random Perturbation Control for P121")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)  # 22
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Step 1: Compute fact directions
    print("  Step 1: Computing fact directions...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)

    fact_directions = {}
    for i in range(boundary, n_layers):
        dirs = [get_fact_direction(base_model, tok, p, i) for p, _, _ in DPO_PAIRS[:5]]
        fd = torch.stack(dirs).mean(dim=0)
        fact_directions[i] = fd / fd.norm()

    # Save base weights
    base_weights = {}
    for i in range(boundary, n_layers):
        key = f"layers.{i}.mlp.down_proj"
        w = base_model.model.layers[i].mlp.down_proj.weight.detach().cpu().float()
        base_weights[key] = w.clone()

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Train DPO and get actual DeltaW
    print("  Step 2: Training DPO for actual DeltaW...")
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

    # Extract DPO DeltaW for down_proj
    dpo_deltas = {}
    for i in range(boundary, n_layers):
        key = f"layers.{i}.mlp.down_proj"
        w_dpo = dpo_model.model.layers[i].mlp.down_proj.weight.detach().cpu().float()
        delta = w_dpo - base_weights[key]
        dpo_deltas[i] = delta

    del dpo_model, ref_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3: Generate random perturbations and compare
    print(f"  Step 3: Generating {N_RANDOM} random perturbations per layer...")
    all_results = {}

    for i in range(boundary, n_layers):
        delta_w = dpo_deltas[i]
        delta_norm = delta_w.norm().item()
        fact_dir = fact_directions[i]

        # DPO's actual cosine (SVD principal direction)
        U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
        dpo_principal = U[:, 0]
        dpo_cos = F.cosine_similarity(
            dpo_principal[:len(fact_dir)].unsqueeze(0),
            fact_dir[:len(dpo_principal)].unsqueeze(0)
        ).item() if len(dpo_principal) == len(fact_dir) else 0

        # Random perturbations with same Frobenius norm
        random_cosines = []
        for _ in range(N_RANDOM):
            # Generate random matrix, scale to same norm
            rand_delta = torch.randn_like(delta_w)
            rand_delta = rand_delta * (delta_norm / rand_delta.norm())
            # SVD of random perturbation
            Ur, Sr, Vhr = torch.linalg.svd(rand_delta, full_matrices=False)
            rand_principal = Ur[:, 0]
            rand_cos = F.cosine_similarity(
                rand_principal[:len(fact_dir)].unsqueeze(0),
                fact_dir[:len(rand_principal)].unsqueeze(0)
            ).item() if len(rand_principal) == len(fact_dir) else 0
            random_cosines.append(abs(rand_cos))  # Use absolute value

        dpo_abs_cos = abs(dpo_cos)
        rand_mean = np.mean(random_cosines)
        rand_std = np.std(random_cosines)
        z_score = (dpo_abs_cos - rand_mean) / rand_std if rand_std > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)

        print(f"    L{i}: DPO |cos|={dpo_abs_cos:.4f}, "
              f"Random |cos|={rand_mean:.4f} +/- {rand_std:.4f}, "
              f"z={z_score:.2f}, p={p_value:.4f}")

        all_results[f"L{i}"] = {
            'dpo_cos': dpo_cos,
            'dpo_abs_cos': dpo_abs_cos,
            'random_mean': rand_mean,
            'random_std': rand_std,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'random_cosines': random_cosines,
        }

    # Save
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != 'random_cosines'}
                 for k, v in all_results.items()}
    out = {'phase': '122', 'name': 'Random Perturbation Control', 'results': save_data}
    with open(os.path.join(RESULTS_DIR, 'phase122_control.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    layers = sorted(all_results.keys())

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        data = all_results[layer]
        rand_cos = data['random_cosines']

        ax.hist(rand_cos, bins=20, alpha=0.7, color='#3498db', label='Random', density=True)
        ax.axvline(x=data['dpo_abs_cos'], color='#e74c3c', linewidth=3,
                  label=f'DPO |cos|={data["dpo_abs_cos"]:.4f}')
        ax.axvline(x=data['random_mean'], color='#95a5a6', linewidth=1, linestyle='--',
                  label=f'Random mean={data["random_mean"]:.4f}')

        sig_text = 'SIGNIFICANT' if data['significant'] else 'NOT significant'
        ax.set_title(f'{layer}: z={data["z_score"]:.2f}, p={data["p_value"]:.4f}\n{sig_text}',
                    fontweight='bold',
                    color='#e74c3c' if data['significant'] else '#95a5a6')
        ax.set_xlabel('|cos(principal direction, fact direction)|')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)

    fig.suptitle('Phase 122: Is DPO\'s Weight Change Direction Meaningful?\n'
                '(vs Random Perturbation of Same Magnitude)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase122_control.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 122] Complete.")


if __name__ == '__main__':
    main()
