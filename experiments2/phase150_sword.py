# -*- coding: utf-8 -*-
"""
Phase 150: Anatomy of the Sword
WHY does Surgery make FGA work? Visualize the logit focusing effect.

P144: FGA alone = 0%, Surgery + FGA = 100%.
Hypothesis: Without surgery, FGA energy "smears" across clustered
number tokens. Surgery creates geometric separation that lets FGA
focus on exactly one target.

Model: Qwen2.5-1.5B (GPU, float16, inference only)
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

# Test prompts with known answers
TEST_PROMPTS = [
    ("Water freezes at", " 0"),
    ("The boiling point of water is", " 100"),
    ("The atomic number of carbon is", " 6"),
]

# Number tokens to track in logit distribution
TRACKED_NUMS = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                " 10", " 100", " 12", " 299", " 365"]


class FGAHook:
    def __init__(self, model, target_token_id, gain):
        self.gain = gain
        unembed = model.lm_head.weight.data[target_token_id].float()
        self.direction = unembed / (unembed.norm() + 1e-8)
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

    def register(self, model, layer_idx):
        self.handle = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle: self.handle.remove()


def disperse_embeddings(model, tok, strength=2.0):
    num_tokens = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
                  " 10", " 100", " 12", " 186", " 212", " 299", " 365", " 366"]
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in num_tokens))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)


def get_logit_profile(model, tok, prompt, target, fga_gain=20, code_mode=True):
    """Get softmax probabilities for tracked number tokens."""
    n_layers = model.config.num_hidden_layers
    fga_layer = n_layers - 4
    text = f"# {prompt}" if code_mode else prompt
    target_id = tok.encode(target)[-1]

    hook = FGAHook(model, target_id, fga_gain)
    hook.register(model, fga_layer)

    inp = tok(text, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        logits = model(**inp).logits[0, -1, :].float()
    hook.remove()

    probs = F.softmax(logits, dim=-1)
    tracked_ids = [tok.encode(t)[-1] for t in TRACKED_NUMS]
    profile = {}
    for t, tid in zip(TRACKED_NUMS, tracked_ids):
        profile[t.strip()] = {
            'logit': logits[tid].item(),
            'prob': probs[tid].item()
        }

    # Also get the top prediction
    pred_id = logits.argmax().item()
    pred_text = tok.decode([pred_id])

    # Concentration metric: prob of target / sum of probs of all tracked nums
    target_prob = probs[target_id].item()
    total_num_prob = sum(probs[tid].item() for tid in tracked_ids)
    concentration = target_prob / (total_num_prob + 1e-10)

    return profile, pred_text, concentration


def main():
    print("[P150] Anatomy of the Sword")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-1.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    all_profiles = {}

    for prompt, target in TEST_PROMPTS:
        print(f"\n  === '{prompt}' -> '{target}' ===")
        profiles = {}

        # A: No surgery, FGA g=20
        print("    A: No surgery + FGA")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float16).eval().to(DEVICE)
        prof, pred, conc = get_logit_profile(model, tok, prompt, target, fga_gain=20)
        print(f"       pred='{pred}' concentration={conc:.2%}")
        profiles['no_surgery'] = {'profile': prof, 'pred': pred, 'concentration': conc}
        del model; gc.collect(); torch.cuda.empty_cache()

        # B: Surgery s=2, FGA g=20
        print("    B: Surgery(s=2) + FGA")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float16).eval().to(DEVICE)
        disperse_embeddings(model, tok, strength=2.0)
        prof, pred, conc = get_logit_profile(model, tok, prompt, target, fga_gain=20)
        print(f"       pred='{pred}' concentration={conc:.2%}")
        profiles['surgery'] = {'profile': prof, 'pred': pred, 'concentration': conc}

        # L2 distances for reference
        embed = model.model.embed_tokens.weight.data
        num_ids = [tok.encode(t)[-1] for t in TRACKED_NUMS]
        vecs = embed[num_ids].float()
        cos = F.cosine_similarity(vecs.unsqueeze(0), vecs.unsqueeze(1), dim=-1)
        avg_cos = cos[~torch.eye(len(num_ids), dtype=bool)].mean().item()
        print(f"       Avg num cosine similarity: {avg_cos:.4f}")
        del model; gc.collect(); torch.cuda.empty_cache()

        all_profiles[prompt] = profiles

    # Save
    with open(os.path.join(RESULTS_DIR, 'phase150_sword.json'), 'w') as f:
        json.dump({'phase': '150', 'name': 'Anatomy of the Sword',
                   'profiles': all_profiles}, f, indent=2, default=str)

    # Plot: logit distribution for each prompt (no-surgery vs surgery)
    n_prompts = len(TEST_PROMPTS)
    fig, axes = plt.subplots(n_prompts, 2, figsize=(16, 5*n_prompts))
    if n_prompts == 1:
        axes = axes.reshape(1, -1)

    for i, (prompt, target) in enumerate(TEST_PROMPTS):
        profiles = all_profiles[prompt]
        nums = list(profiles['no_surgery']['profile'].keys())
        target_clean = target.strip()

        for j, (cond_name, cond_label) in enumerate([
            ('no_surgery', 'Without Surgery + FGA'),
            ('surgery', 'With Surgery(s=2) + FGA')
        ]):
            ax = axes[i, j]
            probs = [profiles[cond_name]['profile'][n]['prob'] for n in nums]
            colors = ['#e74c3c' if n == target_clean else '#3498db' for n in nums]
            bars = ax.bar(nums, probs, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
            conc = profiles[cond_name]['concentration']
            pred = profiles[cond_name]['pred']
            ax.set_title(f'{cond_label}\npred="{pred}", focus={conc:.1%}',
                        fontsize=11, fontweight='bold')
            ax.set_ylabel('Probability', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            # Highlight target
            ax.axvline(x=nums.index(target_clean), color='red', ls='--', alpha=0.3)

        axes[i, 0].set_ylabel(f'"{prompt}"\n\nProbability', fontsize=10)

    plt.suptitle('Phase 150: Anatomy of the Sword\nHow Surgery Focuses FGA Energy',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase150_sword.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === VERDICT ===")
    for prompt, target in TEST_PROMPTS:
        p = all_profiles[prompt]
        no_conc = p['no_surgery']['concentration']
        s_conc = p['surgery']['concentration']
        ratio = s_conc / (no_conc + 1e-10)
        print(f"    '{prompt[:30]}': focus {no_conc:.1%} -> {s_conc:.1%} ({ratio:.1f}x)")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 150] Complete.")

if __name__ == '__main__':
    main()
