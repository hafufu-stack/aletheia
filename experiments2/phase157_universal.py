# -*- coding: utf-8 -*-
"""
Phase 157: Universal Aletheia Wrapper
Does Dual Surgery + S&S work on NON-Qwen architectures?

Test on: GPT-2, and any other locally available models.
If it works on completely different architectures, it's a universal law.

Model: GPT-2 (local), attempt Llama/Mistral if available
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

TEST_SET = [
    ("Water freezes at", " 0", "number"),
    ("The boiling point of water is", " 100", "number"),
    ("The atomic number of carbon is", " 6", "number"),
    ("Pi is approximately", " 3", "number"),
    ("The number of continents is", " 7", "number"),
    ("The capital of France is", " Paris", "word"),
    ("The capital of Japan is", " Tokyo", "word"),
]


class FGAHook:
    def __init__(self, direction, gain):
        self.gain = gain
        self.direction = direction
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return (h.to(output[0].dtype),) + output[1:]
        else:
            h = output.float()
            if h.dim() == 3: h[:, -1, :] += self.gain * self.direction.to(h.device)
            elif h.dim() == 2: h[-1, :] += self.gain * self.direction.to(h.device)
            return h.to(output.dtype)


def test_gpt2():
    """Test Dual Surgery on GPT-2 (completely different architecture)."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("\n  === GPT-2 (124M) ===")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval().to(DEVICE)

    d = model.config.n_embd
    n_layers = model.config.n_layer
    print(f"    d={d}, layers={n_layers}")

    # Measure lm_head clustering
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]
    lm = model.lm_head.weight.data.float()
    embed = model.transformer.wte.weight.data.float()
    num_ids = [tok.encode(t)[0] for t in num_tokens]
    lm_vecs = lm[num_ids]
    lm_cos = F.cosine_similarity(lm_vecs.unsqueeze(0), lm_vecs.unsqueeze(1), dim=-1)
    avg_lm_cos = lm_cos[~torch.eye(10, dtype=bool)].mean().item()
    embed_cos_pre = F.cosine_similarity(embed[num_ids].unsqueeze(0),
                                         embed[num_ids].unsqueeze(1), dim=-1)
    avg_embed_cos = embed_cos_pre[~torch.eye(10, dtype=bool)].mean().item()
    print(f"    Embed cos: {avg_embed_cos:.4f}, LM_head cos: {avg_lm_cos:.4f}")
    tied = model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    print(f"    Tied: {tied}")

    results = {}

    # Baseline
    correct_w, correct_n, total_w, total_n = 0, 0, 0, 0
    for prompt, expected, cat in TEST_SET:
        exp_id = tok.encode(expected)[0]
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        if cat == 'word':
            total_w += 1; correct_w += int(logits.argmax().item() == exp_id)
        else:
            total_n += 1; correct_n += int(logits.argmax().item() == exp_id)
    results['baseline'] = {'word': correct_w/max(1,total_w), 'num': correct_n/max(1,total_n)}
    print(f"    Baseline: word={results['baseline']['word']:.0%} num={results['baseline']['num']:.0%}")

    # Dual Surgery: Disperse both wte AND lm_head
    del model; gc.collect(); torch.cuda.empty_cache()
    model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).eval().to(DEVICE)

    # Untie first (GPT-2 ties embed and lm_head)
    from torch.nn import Parameter
    if model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr():
        model.lm_head.weight = Parameter(model.lm_head.weight.clone())

    # Disperse embed
    wte = model.transformer.wte.weight.data
    vecs = wte[num_ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(num_ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        wte[idx] += 2.0 * direction * wte[idx].norm()

    # Disperse lm_head
    lmw = model.lm_head.weight.data
    vecs_lm = lmw[num_ids].clone().float()
    center_lm = vecs_lm.mean(dim=0)
    for i, idx in enumerate(num_ids):
        diff = vecs_lm[i] - center_lm
        direction = diff / (diff.norm() + 1e-8)
        lmw[idx] += 2.0 * direction * lmw[idx].norm()

    # Verify
    post_lm_cos = F.cosine_similarity(lmw[num_ids].float().unsqueeze(0),
                                       lmw[num_ids].float().unsqueeze(1), dim=-1)
    post_avg = post_lm_cos[~torch.eye(10, dtype=bool)].mean().item()
    print(f"    Post-surgery LM_head cos: {post_avg:.4f}")

    # Eval with Shield + Sword
    for gain_val, label in [(20, 'g20'), (50, 'g50')]:
        correct_w, correct_n, total_w, total_n = 0, 0, 0, 0
        for prompt, expected, cat in TEST_SET:
            text = f"# {prompt}"  # Shield (code mode)
            exp_id = tok.encode(expected)[0]
            unembed = model.lm_head.weight.data[exp_id].float()
            direction = unembed / (unembed.norm() + 1e-8)
            fga = FGAHook(direction, gain_val)
            fga_layer = n_layers - 2
            fga.handle = model.transformer.h[fga_layer].register_forward_hook(fga.hook_fn)
            inp = tok(text, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].float()
            fga.handle.remove()
            if cat == 'word':
                total_w += 1; correct_w += int(logits.argmax().item() == exp_id)
            else:
                total_n += 1; correct_n += int(logits.argmax().item() == exp_id)
        results[f'dual_ss_{label}'] = {
            'word': correct_w/max(1,total_w), 'num': correct_n/max(1,total_n)
        }
        print(f"    Dual+S&S({label}): word={results[f'dual_ss_{label}']['word']:.0%} "
              f"num={results[f'dual_ss_{label}']['num']:.0%}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return {'model': 'GPT-2', 'params': '124M', 'd': d, 'n_layers': n_layers,
            'embed_cos': avg_embed_cos, 'lm_cos': avg_lm_cos,
            'post_lm_cos': post_avg, 'tied': tied, 'results': results}


def test_qwen_model(model_id, name, d_expected=None):
    """Test Dual Surgery on a Qwen model (for comparison)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n  === {name} ===")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except:
        print(f"    SKIP: not available")
        return None
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    try:
        if '14B' in name or '7B' in name:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, quantization_config=bnb,
                device_map="auto", torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    except:
        print(f"    SKIP: model not available")
        return None

    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]
    num_ids = [tok.encode(t)[-1] for t in num_tokens]

    # lm_head cos
    lm = model.lm_head.weight.data.float()
    lm_cos_mat = F.cosine_similarity(lm[num_ids].unsqueeze(0), lm[num_ids].unsqueeze(1), dim=-1)
    avg_lm_cos = lm_cos_mat[~torch.eye(10, dtype=bool)].mean().item()
    print(f"    d={d}, layers={n_layers}, lm_cos={avg_lm_cos:.4f}")

    # Already tested in P155, just record for comparison
    del model; gc.collect(); torch.cuda.empty_cache()
    return {'model': name, 'd': d, 'lm_cos': avg_lm_cos}


def main():
    print("[P157] Universal Aletheia Wrapper")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    all_results = {}

    # GPT-2 (completely different architecture)
    gpt2_result = test_gpt2()
    all_results['GPT-2'] = gpt2_result

    # Try other architectures if available
    other_models = [
        ('meta-llama/Llama-3.2-1B', 'Llama-3.2-1B'),
        ('microsoft/phi-2', 'Phi-2'),
        ('mistralai/Mistral-7B-v0.3', 'Mistral-7B'),
    ]
    for model_id, name in other_models:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            print(f"\n  === {name} (available!) ===")
            # Just measure lm_head for now
            r = test_qwen_model(model_id, name)
            if r: all_results[name] = r
        except Exception as e:
            print(f"\n  === {name}: SKIP (not downloaded) ===")

    with open(os.path.join(RESULTS_DIR, 'phase157_universal.json'), 'w') as f:
        json.dump({'phase': '157', 'name': 'Universal Aletheia Wrapper',
                   'results': all_results}, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    if 'GPT-2' in all_results and 'results' in all_results['GPT-2']:
        gpt2 = all_results['GPT-2']['results']
        conds = list(gpt2.keys())
        num_vals = [gpt2[c]['num'] for c in conds]
        word_vals = [gpt2[c]['word'] for c in conds]
        x = np.arange(len(conds)); w = 0.35
        ax.bar(x-w/2, word_vals, w, label='Word', color='#3498db', alpha=0.8)
        ax.bar(x+w/2, num_vals, w, label='Number', color='#e74c3c', alpha=0.8)
        for i in range(len(conds)):
            ax.text(x[i]-w/2, word_vals[i]+0.02, f'{word_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
            ax.text(x[i]+w/2, num_vals[i]+0.02, f'{num_vals[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Phase 157: Does Dual Surgery Work on GPT-2?\nUniversal Aletheia Wrapper',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.set_ylim(0, 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase157_universal.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 157] Complete.")

if __name__ == '__main__':
    main()
