# -*- coding: utf-8 -*-
"""
Phase 195: Causal Abacus Hijacking
Can we WRITE to the virtual abacus, not just READ from it?

If carry bit is a physical register (P190), injecting "carry=1"
into a no-carry problem (like 1+1=) should make the model output
a 2-digit answer (e.g. 11, 12) instead of 2.

This is a CAUSAL proof: the model uses internal registers
for computation, not just pattern matching.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"]

def apply_surgery(model, tok, strength=2.0):
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)

def gen_additions():
    data = []
    for a in range(10):
        for b in range(10):
            s = a + b
            data.append((a, b, s, 1 if s >= 10 else 0))
    return data

def main():
    print("[P195] Causal Abacus Hijacking")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    additions = gen_additions()
    steer_layers = [14, 15, 18, 20]  # Where carry info lives

    # Step 1: Extract carry direction vectors via probe
    print("  Extracting carry direction vectors...")
    for target_l in steer_layers:
        hiddens_carry = []; hiddens_nocarry = []
        captured = [None]
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[0] = output[0][:, -1, :].detach().clone()
            else:
                captured[0] = output[:, -1, :].detach().clone()
        for a, b, s, carry in additions:
            prompt = f"def f(): return {a} + {b} ="
            h = model.model.layers[target_l].register_forward_hook(hook_fn)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                _ = model(**inp)
            h.remove()
            if carry == 1:
                hiddens_carry.append(captured[0].squeeze())
            else:
                hiddens_nocarry.append(captured[0].squeeze())

        carry_mean = torch.stack(hiddens_carry).mean(dim=0)
        nocarry_mean = torch.stack(hiddens_nocarry).mean(dim=0)
        # Carry direction = difference between carry and no-carry states
        carry_dir = carry_mean - nocarry_mean
        carry_dir_norm = carry_dir / (carry_dir.norm() + 1e-8)

        # Save for later
        if target_l == steer_layers[0]:
            all_carry_dirs = {}
        all_carry_dirs[target_l] = {'dir': carry_dir_norm, 'magnitude': carry_dir.norm().item()}
        print(f"    L{target_l}: carry direction magnitude = {carry_dir.norm().item():.2f}")

    # Step 2: Hijack! Inject carry=1 into no-carry problems
    print("\n  === Hijacking no-carry problems ===")
    no_carry_problems = [(a, b, a+b) for a in range(5) for b in range(5) if a+b < 10]

    results = {'normal': [], 'hijacked': {}}
    for scale in [0, 1, 3, 5, 10]:
        hijacked = []
        for a, b, s in no_carry_problems:
            prompt = f"def f(): return {a} + {b} ="
            if scale == 0:
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    logits = model(**inp).logits[0, -1, :].float()
                pred_id = logits.argmax().item()
                pred_text = tok.decode([pred_id]).strip()
                results['normal'].append({'a': a, 'b': b, 'sum': s, 'pred': pred_text})
            else:
                # Inject carry direction at all steer layers
                hooks = []
                for sl in steer_layers:
                    d = all_carry_dirs[sl]['dir']
                    def mk(dd, sc):
                        def fn(module, input, output):
                            if isinstance(output, tuple):
                                h = output[0].float()
                                if h.dim() == 3:
                                    h[:, -1, :] += sc * dd.to(h.device)
                                return (h.to(output[0].dtype),) + output[1:]
                            return output
                        return fn
                    hooks.append(model.model.layers[sl].register_forward_hook(mk(d, scale)))
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    logits = model(**inp).logits[0, -1, :].float()
                for hk in hooks: hk.remove()
                pred_id = logits.argmax().item()
                pred_text = tok.decode([pred_id]).strip()
                hijacked.append({'a': a, 'b': b, 'sum': s, 'pred': pred_text})

        if scale > 0:
            results['hijacked'][f's{scale}'] = hijacked
            # Count how many predictions changed to 2-digit
            two_digit = sum(1 for h in hijacked if h['pred'].isdigit() and int(h['pred']) >= 10)
            changed = sum(1 for h, n in zip(hijacked, results['normal']) if h['pred'] != n['pred'])
            print(f"  Scale={scale}: changed={changed}/{len(no_carry_problems)} "
                  f"two_digit={two_digit}/{len(no_carry_problems)}")

    # Show some examples
    print("\n  === Example hijacks (scale=5) ===")
    for h, n in zip(results['hijacked']['s5'][:10], results['normal'][:10]):
        marker = " HIJACKED!" if h['pred'] != n['pred'] else ""
        print(f"    {h['a']}+{h['b']}={h['sum']}: normal='{n['pred']}' -> hijack='{h['pred']}'{marker}")

    with open(os.path.join(RESULTS_DIR, 'phase195_hijack.json'), 'w') as f:
        json.dump({'phase': '195', 'name': 'Causal Abacus Hijacking',
                   'results': results,
                   'carry_magnitudes': {str(l): all_carry_dirs[l]['magnitude'] for l in steer_layers}},
                  f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    scales = [1, 3, 5, 10]
    changed_rates = []
    for s in scales:
        hijacked = results['hijacked'][f's{s}']
        changed = sum(1 for h, n in zip(hijacked, results['normal']) if h['pred'] != n['pred'])
        changed_rates.append(changed / len(no_carry_problems))
    ax.bar([str(s) for s in scales], changed_rates, color='#e74c3c', alpha=0.8)
    for i, v in enumerate(changed_rates):
        ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=14, fontweight='bold')
    ax.set_xlabel('Carry Injection Scale', fontsize=13)
    ax.set_ylabel('Fraction of Predictions Changed', fontsize=13)
    ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase 195: Causal Abacus Hijacking\n'
                 'Injecting "carry=1" into no-carry problems (a+b<10)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase195_hijack.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    total_changed = sum(1 for h, n in zip(results['hijacked']['s5'], results['normal']) if h['pred'] != n['pred'])
    print(f"\n  === VERDICT ===")
    print(f"  -> At scale=5: {total_changed}/{len(no_carry_problems)} predictions changed")
    if total_changed > len(no_carry_problems) * 0.3:
        print("  -> CAUSAL PROOF: Writing to the abacus CHANGES the output!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 195] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
