# -*- coding: utf-8 -*-
"""
Phase 209: Full State Replacement (The Write Breakthrough Attempt)
Previous write attempts (P195, P201, P202) all used ADDITIVE steering.
What if we REPLACE the entire hidden state at the target layer?

Replace h[L20] for "3+4=" with h[L20] from a known "7+8=" problem.
If the output changes from 7 to 15, we achieved a full register transplant.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time
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

def main():
    print("[P209] Full State Replacement")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    # Step 1: Collect "donor" hidden states from known problems
    print("  Collecting donor states...")
    donors = {}  # sum -> {layer: hidden_state}
    for a in range(10):
        for b in range(10):
            s = a + b
            if s not in donors:
                donors[s] = {}
            prompt = f"def f(): return {a} + {b} ="
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                outputs = model(**inp, output_hidden_states=True)
            # Store hidden states from key layers
            for l in [16, 18, 20, 22]:
                h = outputs.hidden_states[l + 1][:, -1, :].detach().clone()
                if l not in donors[s]:
                    donors[s][l] = []
                donors[s][l].append(h)

    # Average donor states per sum value
    for s in donors:
        for l in donors[s]:
            donors[s][l] = torch.stack(donors[s][l]).mean(dim=0)

    # Step 2: Transplant test
    # Take problem "3+4=7" but replace its L20 state with donor from sum=15
    test_cases = [
        (3, 4, 7, 15),   # 3+4=7, transplant sum=15
        (2, 3, 5, 12),   # 2+3=5, transplant sum=12
        (1, 2, 3, 18),   # 1+2=3, transplant sum=18
        (4, 1, 5, 9),    # 4+1=5, transplant sum=9
        (5, 2, 7, 14),   # 5+2=7, transplant sum=14
        (1, 1, 2, 16),   # 1+1=2, transplant sum=16
    ]

    results = []
    for transplant_layer in [16, 18, 20, 22]:
        print(f"\n  === Transplant at L{transplant_layer} ===")
        layer_results = []
        for a, b, true_sum, target_sum in test_cases:
            prompt = f"def f(): return {a} + {b} ="
            # Baseline
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                base_logits = model(**inp).logits[0, -1, :].float()
            base_pred = tok.decode([base_logits.argmax().item()]).strip()

            # Transplant
            donor_state = donors[target_sum][transplant_layer]
            def replace_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h[:, -1, :] = donor_state.squeeze().to(h.dtype).to(h.device)
                    return (h,) + output[1:]
                h = output.clone()
                h[:, -1, :] = donor_state.squeeze().to(h.dtype).to(h.device)
                return h
            handle = model.model.layers[transplant_layer].register_forward_hook(replace_fn)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                trans_logits = model(**inp).logits[0, -1, :].float()
            handle.remove()
            trans_pred = tok.decode([trans_logits.argmax().item()]).strip()

            # Check: does output match donor's sum?
            target_tok = str(target_sum)
            # For 2-digit, first digit
            if target_sum >= 10:
                target_first = str(target_sum)[0]
                matches_target = trans_pred == target_first or trans_pred == target_tok
            else:
                matches_target = trans_pred == target_tok

            changed = trans_pred != base_pred
            layer_results.append({
                'a': a, 'b': b, 'true_sum': true_sum, 'target_sum': target_sum,
                'base_pred': base_pred, 'trans_pred': trans_pred,
                'changed': changed, 'matches_target': matches_target
            })
            marker = " TRANSPLANTED!" if matches_target else (" changed" if changed else "")
            print(f"    {a}+{b}={true_sum} [target={target_sum}]: "
                  f"base='{base_pred}' trans='{trans_pred}'{marker}")

        n_changed = sum(1 for r in layer_results if r['changed'])
        n_target = sum(1 for r in layer_results if r['matches_target'])
        results.append({
            'layer': transplant_layer,
            'changed': n_changed, 'matched': n_target,
            'total': len(test_cases), 'details': layer_results
        })
        print(f"  L{transplant_layer}: {n_changed}/{len(test_cases)} changed, "
              f"{n_target}/{len(test_cases)} matched target")

    with open(os.path.join(RESULTS_DIR, 'phase209_replacement.json'), 'w') as f:
        json.dump({'phase': '209', 'name': 'Full State Replacement',
                   'results': results}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    layers = [r['layer'] for r in results]
    changed_pct = [r['changed']/r['total'] for r in results]
    matched_pct = [r['matched']/r['total'] for r in results]
    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x-w/2, changed_pct, w, label='Output Changed', color='#e67e22', alpha=0.8)
    ax.bar(x+w/2, matched_pct, w, label='Matched Target Sum', color='#27ae60', alpha=0.8)
    for i in range(len(layers)):
        ax.text(x[i]-w/2, changed_pct[i]+0.02, f'{changed_pct[i]:.0%}', ha='center', fontsize=11, fontweight='bold')
        ax.text(x[i]+w/2, matched_pct[i]+0.02, f'{matched_pct[i]:.0%}', ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'L{l}' for l in layers], fontsize=12)
    ax.set_ylabel('Rate', fontsize=12); ax.set_ylim(0, 1.2)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase 209: Full State Replacement\n'
                 'Can we transplant hidden states to change the output?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase209_replacement.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    best = max(results, key=lambda r: r['matched'])
    print(f"\n  === VERDICT ===")
    print(f"  -> Best transplant: L{best['layer']} ({best['matched']}/{best['total']} matched)")
    if best['matched'] > 0:
        print("  -> WRITE SUCCESS! Full state replacement works!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 209] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
