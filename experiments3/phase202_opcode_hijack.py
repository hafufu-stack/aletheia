# -*- coding: utf-8 -*-
"""
Phase 202: OPCODE Hijacking
P200: OPCODE is 100% readable from L2.
Can we REWRITE it? Change + to * at L2, making 3+4 output 12.

Causal proof that the OPCODE register controls computation.

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
    print("[P202] OPCODE Hijacking")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    hijack_layers = [2, 4, 6]  # Early layers where OPCODE lives

    # Step 1: Extract OPCODE direction vectors
    print("  Extracting OPCODE direction vectors...")
    ops = {'+': [], '-': [], '*': []}
    captured = [None]
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured[0] = output[0][:, -1, :].detach().clone()
        else:
            captured[0] = output[:, -1, :].detach().clone()

    for a in range(2, 8):
        for b in range(1, a):
            for op in ['+', '-', '*']:
                prompt = f"def f(): return {a} {op} {b} ="
                h = model.model.layers[2].register_forward_hook(hook_fn)
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad(): _ = model(**inp)
                h.remove()
                ops[op].append(captured[0].squeeze())

    op_means = {}
    for op in ['+', '-', '*']:
        op_means[op] = torch.stack(ops[op]).mean(dim=0)

    # Direction: + -> * (subtract plus, add multiply)
    plus_to_mul = op_means['*'] - op_means['+']
    plus_to_mul_norm = plus_to_mul / (plus_to_mul.norm() + 1e-8)
    plus_to_sub = op_means['-'] - op_means['+']
    plus_to_sub_norm = plus_to_sub / (plus_to_sub.norm() + 1e-8)

    print(f"    + -> * direction magnitude: {plus_to_mul.norm().item():.2f}")
    print(f"    + -> - direction magnitude: {plus_to_sub.norm().item():.2f}")

    # Step 2: Test hijacking
    test_cases = [
        (3, 4, '+'),  # 3+4=7, want 3*4=12
        (5, 3, '+'),  # 5+3=8, want 5*3=15
        (6, 2, '+'),  # 6+2=8, want 6*2=12
        (4, 3, '+'),  # 4+3=7, want 4*3=12
        (7, 2, '+'),  # 7+2=9, want 7*2=14
        (5, 4, '+'),  # 5+4=9, want 5*4=20
    ]

    results = []
    print("\n  === OPCODE Hijacking: + -> * ===")
    for scale in [0, 3, 5, 10, 20]:
        scale_results = []
        for a, b, original_op in test_cases:
            prompt = f"def f(): return {a} {original_op} {b} ="
            true_sum = a + b
            target_product = a * b

            if scale == 0:
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    logits = model(**inp).logits[0, -1, :].float()
            else:
                hooks = []
                for hl in hijack_layers:
                    def mk(dd, sc):
                        def fn(module, input, output):
                            if isinstance(output, tuple):
                                h_out = output[0].float()
                                if h_out.dim() == 3:
                                    h_out[:, -1, :] += sc * dd.to(h_out.device)
                                return (h_out.to(output[0].dtype),) + output[1:]
                            return output
                        return fn
                    hooks.append(model.model.layers[hl].register_forward_hook(mk(plus_to_mul_norm, scale)))
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    logits = model(**inp).logits[0, -1, :].float()
                for hk in hooks: hk.remove()

            pred_id = logits.argmax().item()
            pred_text = tok.decode([pred_id]).strip()
            # Check if output matches multiplication result
            matches_product = (pred_text == str(target_product))
            matches_sum = (pred_text == str(true_sum))
            scale_results.append({
                'a': a, 'b': b, 'op': original_op,
                'true_sum': true_sum, 'target_product': target_product,
                'pred': pred_text, 'matches_product': matches_product,
                'matches_sum': matches_sum
            })

        pct_product = sum(1 for r in scale_results if r['matches_product']) / len(test_cases)
        pct_sum = sum(1 for r in scale_results if r['matches_sum']) / len(test_cases)
        label = "baseline" if scale == 0 else f"s{scale}"
        results.append({'scale': scale, 'label': label,
                        'pct_product': pct_product, 'pct_sum': pct_sum,
                        'details': scale_results})
        print(f"  Scale={scale:2d}: sum_match={pct_sum:.0%} product_match={pct_product:.0%}")
        if scale > 0:
            for r in scale_results:
                marker = " HIJACKED!" if r['matches_product'] else ""
                print(f"    {r['a']}{r['op']}{r['b']}: "
                      f"sum={r['true_sum']} product={r['target_product']} "
                      f"pred='{r['pred']}'{marker}")

    with open(os.path.join(RESULTS_DIR, 'phase202_opcode_hijack.json'), 'w') as f:
        json.dump({'phase': '202', 'name': 'OPCODE Hijacking',
                   'results': results}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    scales_list = [r['scale'] for r in results]
    prod_vals = [r['pct_product'] for r in results]
    sum_vals = [r['pct_sum'] for r in results]
    ax.plot(scales_list, sum_vals, 'b-o', lw=2.5, markersize=8, label='Outputs SUM (a+b)')
    ax.plot(scales_list, prod_vals, 'r-s', lw=2.5, markersize=8, label='Outputs PRODUCT (a*b)')
    ax.set_xlabel('OPCODE Hijack Strength', fontsize=13)
    ax.set_ylabel('Match Rate', fontsize=13)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_title('Phase 202: OPCODE Hijacking\n'
                 'Can we change "+" to "*" by rewriting the instruction register?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase202_opcode_hijack.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    best_hijack = max(results[1:], key=lambda r: r['pct_product'])
    print(f"\n  === VERDICT ===")
    print(f"  -> Baseline sum match: {results[0]['pct_sum']:.0%}")
    print(f"  -> Best product match: scale={best_hijack['scale']} ({best_hijack['pct_product']:.0%})")
    if best_hijack['pct_product'] > 0:
        print("  -> OPCODE HIJACKED! + was rewritten to * !")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 202] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
