# -*- coding: utf-8 -*-
"""
Phase 128: Hidden State Movement Analysis
Measures HOW hidden states at L23 move after DPO training.
For each prompt, compute h_DPO - h_base at L23 and analyze:
1. Direction of movement (toward chosen or rejected token embedding?)
2. Magnitude of movement
3. Consistency across prompts

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


def get_hidden(model, tok, prompt, layer_idx):
    """Get hidden state at layer_idx for the last token."""
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
    print("[P128] Hidden State Movement Analysis")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    n_layers = 24
    boundary = int(n_layers * 0.94)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Get token embeddings for chosen/rejected
    print("  Getting token embeddings...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)

    embed_weight = base_model.model.embed_tokens.weight.detach().cpu().float()

    # Get base hidden states at L22 and L23
    print("  Getting base hidden states...")
    base_hiddens = {}
    for layer_idx in [22, 23]:
        for prompt, chosen, rejected in DPO_PAIRS:
            key = f"{prompt[:30]}_L{layer_idx}"
            base_hiddens[key] = get_hidden(base_model, tok, prompt, layer_idx)

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

    del ref_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Get DPO hidden states and compare
    print("\n  Analyzing hidden state movements...")
    results = []

    for layer_idx in [22, 23]:
        print(f"\n  --- Layer {layer_idx} ---")
        for prompt, chosen, rejected in DPO_PAIRS:
            key = f"{prompt[:30]}_L{layer_idx}"
            h_base = base_hiddens[key]
            h_dpo = get_hidden(dpo_model, tok, prompt, layer_idx)

            # Movement vector
            delta_h = h_dpo - h_base
            move_magnitude = delta_h.norm().item()

            # Get token embeddings
            chosen_ids = tok.encode(chosen)
            rejected_ids = tok.encode(rejected)
            chosen_id = chosen_ids[-1] if chosen_ids else 0
            rejected_id = rejected_ids[-1] if rejected_ids else 0

            chosen_emb = embed_weight[chosen_id]
            rejected_emb = embed_weight[rejected_id]

            # Direction analysis
            cos_chosen = F.cosine_similarity(
                delta_h.unsqueeze(0), chosen_emb.unsqueeze(0)).item()
            cos_rejected = F.cosine_similarity(
                delta_h.unsqueeze(0), rejected_emb.unsqueeze(0)).item()

            # Direction toward chosen vs rejected
            toward_chosen = cos_chosen > cos_rejected

            def safe_tok(s):
                return s.encode('ascii', 'replace').decode()

            result = {
                'layer': layer_idx, 'prompt': prompt[:35],
                'chosen': safe_tok(chosen), 'rejected': safe_tok(rejected),
                'move_magnitude': move_magnitude,
                'cos_chosen': cos_chosen, 'cos_rejected': cos_rejected,
                'toward_chosen': toward_chosen,
            }
            results.append(result)

            direction = 'TOWARD_CHOSEN' if toward_chosen else 'TOWARD_REJECTED'
            print(f"    {prompt[:30]:30s} {direction:17s} "
                  f"|dh|={move_magnitude:.4f} "
                  f"cos_c={cos_chosen:+.3f} cos_r={cos_rejected:+.3f}")

    # Summary
    for layer_idx in [22, 23]:
        lr = [r for r in results if r['layer'] == layer_idx]
        toward = sum(1 for r in lr if r['toward_chosen'])
        print(f"\n  L{layer_idx}: Toward chosen: {toward}/{len(lr)} ({toward/len(lr):.0%})")
        mean_mag = np.mean([r['move_magnitude'] for r in lr])
        print(f"  L{layer_idx}: Mean |dh| = {mean_mag:.4f}")

    del dpo_model, optimizer; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save
    out = {'phase': '128', 'name': 'Hidden State Movement', 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase128_hidden_movement.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, layer_idx in enumerate([22, 23]):
        ax = axes[idx]
        lr = [r for r in results if r['layer'] == layer_idx]
        prompts_short = [r['prompt'][:15] for r in lr]
        cos_c = [r['cos_chosen'] for r in lr]
        cos_r = [r['cos_rejected'] for r in lr]
        x = range(len(lr))
        ax.bar([i-0.15 for i in x], cos_c, 0.3, color='#2ecc71', label='cos(dh, chosen)')
        ax.bar([i+0.15 for i in x], cos_r, 0.3, color='#e74c3c', label='cos(dh, rejected)')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(prompts_short, rotation=90, fontsize=6)
        ax.set_ylabel('Cosine Similarity')
        toward = sum(1 for r in lr if r['toward_chosen'])
        ax.set_title(f'L{layer_idx}: Hidden State Direction\n'
                    f'Toward chosen: {toward}/{len(lr)}',
                    fontweight='bold')
        ax.legend(fontsize=8)

    # Panel 3: Movement magnitude
    ax = axes[2]
    for layer_idx in [22, 23]:
        lr = [r for r in results if r['layer'] == layer_idx]
        mags = [r['move_magnitude'] for r in lr]
        ax.hist(mags, bins=10, alpha=0.6, label=f'L{layer_idx}')
    ax.set_xlabel('|h_DPO - h_base|')
    ax.set_ylabel('Count')
    ax.set_title('Movement Magnitude Distribution', fontweight='bold')
    ax.legend()

    fig.suptitle('Phase 128: Where Do Hidden States Move After DPO?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase128_hidden_movement.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 128] Complete.")

if __name__ == '__main__':
    main()
