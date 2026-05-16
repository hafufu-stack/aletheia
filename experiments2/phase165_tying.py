# -*- coding: utf-8 -*-
"""
Phase 165: The Weight Tying Revelation
Verify: is 0.5B's embed_tokens the SAME tensor as lm_head?
If yes, all our "embed-only surgery" experiments were secretly
Dual Surgery from the beginning!

This is crucial for scientific integrity.

Model: All available Qwen2.5 + GPT-2 (diagnostic only)
"""
import torch, json, os, gc, time
import torch.nn.functional as F

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def check_tying(model_id, name, arch='qwen'):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        if arch == 'gpt2':
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
            model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
            embed = model.transformer.wte.weight
            lm = model.lm_head.weight
        else:
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            if '14B' in name:
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, local_files_only=True, quantization_config=bnb,
                    device_map="auto", torch_dtype=torch.float16)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, local_files_only=True, torch_dtype=torch.float32)
            embed = model.model.embed_tokens.weight
            lm = model.lm_head.weight
    except Exception as e:
        print(f"    SKIP {name}: {e}")
        return None

    # Check if same tensor (data_ptr)
    same_ptr = embed.data_ptr() == lm.data_ptr()

    # Check shapes
    same_shape = embed.shape == lm.shape

    # Check numerical equality (even if different pointers)
    if same_shape:
        max_diff = (embed.float() - lm.float()).abs().max().item()
        allclose = max_diff < 1e-6
    else:
        max_diff = float('inf')
        allclose = False

    # Config check
    tie_config = getattr(model.config, 'tie_word_embeddings', 'unknown')

    result = {
        'name': name,
        'same_pointer': same_ptr,
        'same_shape': same_shape,
        'max_diff': max_diff,
        'allclose': allclose,
        'tie_config': str(tie_config),
        'embed_shape': list(embed.shape),
        'lm_shape': list(lm.shape),
    }

    status = "TIED (same tensor)" if same_ptr else (
        "CLONED (identical values)" if allclose else "INDEPENDENT")

    print(f"    {name}: {status}")
    print(f"      same_ptr={same_ptr}, max_diff={max_diff:.2e}, "
          f"config.tie={tie_config}")
    print(f"      embed={list(embed.shape)}, lm={list(lm.shape)}")

    # Additional: if tied, demonstrate that modifying embed also modifies lm_head
    if same_ptr and arch != 'gpt2':
        # Save original value
        test_idx = 0
        original = embed.data[test_idx, 0].item()
        embed.data[test_idx, 0] += 999.0
        lm_changed = lm.data[test_idx, 0].item()
        embed.data[test_idx, 0] -= 999.0  # restore
        result['mutation_propagates'] = abs(lm_changed - original - 999.0) < 0.01
        print(f"      Mutation test: embed[0,0]+999 -> lm[0,0] changed? "
              f"{'YES' if result['mutation_propagates'] else 'NO'}")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def main():
    print("[P165] The Weight Tying Revelation")
    start_time = time.time()

    models = [
        ('Qwen/Qwen2.5-0.5B', '0.5B', 'qwen'),
        ('Qwen/Qwen2.5-1.5B', '1.5B', 'qwen'),
        ('Qwen/Qwen2.5-14B', '14B', 'qwen'),
        ('gpt2', 'GPT-2', 'gpt2'),
    ]

    results = []
    for model_id, name, arch in models:
        print(f"\n  === {name} ===")
        r = check_tying(model_id, name, arch)
        if r: results.append(r)

    with open(os.path.join(RESULTS_DIR, 'phase165_tying.json'), 'w') as f:
        json.dump({'phase': '165', 'name': 'Weight Tying Revelation',
                   'results': results}, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\n  === IMPLICATIONS ===")
    for r in results:
        tied = r['same_pointer'] or r['allclose']
        print(f"    {r['name']:5s}: {'TIED' if tied else 'INDEPENDENT'}")
        if tied:
            print(f"      -> 'Embed-only surgery' on this model is actually DUAL SURGERY!")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 165] Complete.")

if __name__ == '__main__':
    main()
