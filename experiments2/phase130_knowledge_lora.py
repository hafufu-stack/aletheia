# -*- coding: utf-8 -*-
"""
Phase 130: Knowledge / Skill LoRA Decoupling
V5 proved DPO effect on numbers = 0.0000 (P129).
DPO edits the "Skill" layers (L22-L23 Attention/MLP), but numerical knowledge
must be injected into "Knowledge" layers (shallow FFNs where facts are stored).

Experiment:
1. Knowledge LoRA: rank-8 adapter on FFN layers L1-L10 (knowledge injection)
2. Skill LoRA: rank-8 adapter on Attention layers L22-L23 (format control)
3. Combined: Both LoRA modules simultaneously
4. Baseline: DPO on L22-L23 MLP (P117b approach, known to fail on numbers)

If Knowledge LoRA succeeds where DPO failed, it proves the KAT architecture
hypothesis: "facts must be written into FFNs, not adjusted via preference."

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

# Training pairs: mix of word and numerical facts
TRAIN_PAIRS = [
    ("The capital of Japan is", " Tokyo", " Osaka", "word"),
    ("The capital of France is", " Paris", " Lyon", "word"),
    ("The capital of Germany is", " Berlin", " Munich", "word"),
    ("The capital of Italy is", " Rome", " Milan", "word"),
    ("The capital of Spain is", " Madrid", " Barcelona", "word"),
    ("Water freezes at", " 0", " 100", "number"),
    ("The boiling point of water is", " 100", " 212", "number"),
    ("The atomic number of carbon is", " 6", " 12", "number"),
    ("The speed of light is approximately", " 299", " 186", "number"),
]

# Test pairs (unseen)
TEST_PAIRS = [
    ("The capital of the United Kingdom is", " London", " Manchester", "word"),
    ("The largest planet is", " Jupiter", " Saturn", "word"),
    ("The number of planets in the solar system is", " 8", " 9", "number"),
    ("A year has", " 365", " 366", "number"),
    ("The atomic number of oxygen is", " 8", " 16", "number"),
]


class LoRALayer(torch.nn.Module):
    """Lightweight LoRA adapter for a linear layer."""
    def __init__(self, original_layer, rank=8, alpha=16.0):
        super().__init__()
        self.original = original_layer
        in_f = original_layer.in_features
        out_f = original_layer.out_features
        self.lora_A = torch.nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_f))
        self.scale = alpha / rank

    def forward(self, x):
        orig = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return orig + lora_out


def attach_knowledge_lora(model, layers_range, rank=8):
    """Attach LoRA to FFN (gate/up/down proj) in specified layers."""
    adapters = []
    for i in layers_range:
        layer = model.model.layers[i]
        # Wrap gate_proj
        lora = LoRALayer(layer.mlp.gate_proj, rank=rank).to(DEVICE)
        layer.mlp.gate_proj = lora
        adapters.extend([lora.lora_A, lora.lora_B])
        # Wrap up_proj
        lora2 = LoRALayer(layer.mlp.up_proj, rank=rank).to(DEVICE)
        layer.mlp.up_proj = lora2
        adapters.extend([lora2.lora_A, lora2.lora_B])
    return adapters


def attach_skill_lora(model, layers_range, rank=8):
    """Attach LoRA to Attention (q/k/v proj) in specified layers."""
    adapters = []
    for i in layers_range:
        layer = model.model.layers[i]
        for proj_name in ['q_proj', 'k_proj', 'v_proj']:
            orig = getattr(layer.self_attn, proj_name)
            lora = LoRALayer(orig, rank=rank).to(DEVICE)
            setattr(layer.self_attn, proj_name, lora)
            adapters.extend([lora.lora_A, lora.lora_B])
    return adapters


def train_supervised(model, tok, pairs, adapter_params, epochs=20, lr=1e-4):
    """Supervised training: maximize P(chosen | prompt)."""
    optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=0.01)
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for prompt, chosen, _, _ in pairs:
            text = prompt + chosen
            inp = tok(text, return_tensors='pt').to(DEVICE)
            prompt_len = tok(prompt, return_tensors='pt')['input_ids'].shape[1]
            logits = model(**inp).logits[0, prompt_len-1:-1, :]
            targets = inp['input_ids'][0, prompt_len:]
            loss = F.cross_entropy(logits.float(), targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(pairs))
    return losses


def evaluate(model, tok, pairs, label=""):
    """Evaluate accuracy and per-category performance."""
    results = {'word_correct': 0, 'word_total': 0,
               'num_correct': 0, 'num_total': 0, 'details': []}
    for prompt, chosen, rejected, cat in pairs:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        c_id = tok.encode(chosen)[-1]
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :].float()
        pred_id = logits.argmax().item()
        correct = (pred_id == c_id)
        if cat == 'word':
            results['word_total'] += 1
            results['word_correct'] += int(correct)
        else:
            results['num_total'] += 1
            results['num_correct'] += int(correct)
        results['details'].append({
            'prompt': prompt[:40], 'cat': cat,
            'correct': correct,
            'pred': tok.decode([pred_id]).encode('ascii', 'replace').decode(),
            'expected': chosen.strip(),
        })
    total = results['word_total'] + results['num_total']
    total_c = results['word_correct'] + results['num_correct']
    results['total_acc'] = total_c / total if total > 0 else 0
    results['word_acc'] = results['word_correct'] / results['word_total'] if results['word_total'] else 0
    results['num_acc'] = results['num_correct'] / results['num_total'] if results['num_total'] else 0
    if label:
        print(f"    {label}: total={results['total_acc']:.0%} "
              f"word={results['word_acc']:.0%} num={results['num_acc']:.0%}")
    return results


def main():
    print("[P130] Knowledge / Skill LoRA Decoupling")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_results = {}

    # === Config 1: Baseline (no LoRA) ===
    print("\n  === Config 0: Baseline ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    base_train = evaluate(model, tok, TRAIN_PAIRS, "Train")
    base_test = evaluate(model, tok, TEST_PAIRS, "Test")
    all_results['baseline'] = {'train': base_train, 'test': base_test}
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Config 1: Knowledge LoRA only (FFN L1-L10) ===
    print("\n  === Config 1: Knowledge LoRA (FFN L1-L10) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    k_params = attach_knowledge_lora(model, range(1, 11), rank=8)
    losses_k = train_supervised(model, tok, TRAIN_PAIRS, k_params, epochs=20, lr=1e-4)
    model.eval()
    k_train = evaluate(model, tok, TRAIN_PAIRS, "Train")
    k_test = evaluate(model, tok, TEST_PAIRS, "Test")
    all_results['knowledge_lora'] = {
        'train': k_train, 'test': k_test,
        'final_loss': losses_k[-1], 'losses': losses_k,
        'n_params': sum(p.numel() for p in k_params),
    }
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Config 2: Skill LoRA only (Attn L22-L23) ===
    print("\n  === Config 2: Skill LoRA (Attn L22-L23) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    s_params = attach_skill_lora(model, range(22, 24), rank=8)
    losses_s = train_supervised(model, tok, TRAIN_PAIRS, s_params, epochs=20, lr=1e-4)
    model.eval()
    s_train = evaluate(model, tok, TRAIN_PAIRS, "Train")
    s_test = evaluate(model, tok, TEST_PAIRS, "Test")
    all_results['skill_lora'] = {
        'train': s_train, 'test': s_test,
        'final_loss': losses_s[-1], 'losses': losses_s,
        'n_params': sum(p.numel() for p in s_params),
    }
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Config 3: Combined (Knowledge FFN L1-L10 + Skill Attn L22-L23) ===
    print("\n  === Config 3: Combined Knowledge+Skill LoRA ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    k_params2 = attach_knowledge_lora(model, range(1, 11), rank=8)
    s_params2 = attach_skill_lora(model, range(22, 24), rank=8)
    combined_params = k_params2 + s_params2
    losses_c = train_supervised(model, tok, TRAIN_PAIRS, combined_params, epochs=20, lr=1e-4)
    model.eval()
    c_train = evaluate(model, tok, TRAIN_PAIRS, "Train")
    c_test = evaluate(model, tok, TEST_PAIRS, "Test")
    all_results['combined_lora'] = {
        'train': c_train, 'test': c_test,
        'final_loss': losses_c[-1], 'losses': losses_c,
        'n_params': sum(p.numel() for p in combined_params),
    }
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Config 4: DPO baseline (L22-L23 MLP, same as P117b) ===
    print("\n  === Config 4: DPO Baseline (L22-L23 MLP) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).eval().to(DEVICE)
    for name, param in model.named_parameters():
        param.requires_grad = False
        for i in range(22, 24):
            if f"layers.{i}." in name and "mlp" in name:
                param.requires_grad = True
    dpo_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(dpo_params, lr=5e-6)
    for epoch in range(5):
        for prompt, chosen, rejected, _ in TRAIN_PAIRS:
            text_c = prompt + chosen
            text_r = prompt + rejected
            inp_c = tok(text_c, return_tensors='pt').to(DEVICE)
            inp_r = tok(text_r, return_tensors='pt').to(DEVICE)
            plen = tok(prompt, return_tensors='pt')['input_ids'].shape[1]
            lc = model(**inp_c).logits[0, plen-1:-1, :].float().clamp(-100, 100)
            lr_ = model(**inp_r).logits[0, plen-1:-1, :].float().clamp(-100, 100)
            lp_c = F.log_softmax(lc, dim=-1).gather(1, inp_c['input_ids'][0, plen:].unsqueeze(1)).squeeze()
            lp_r = F.log_softmax(lr_, dim=-1).gather(1, inp_r['input_ids'][0, plen:].unsqueeze(1)).squeeze()
            if lp_c.dim() == 0: lp_c = lp_c.unsqueeze(0)
            if lp_r.dim() == 0: lp_r = lp_r.unsqueeze(0)
            with torch.no_grad():
                rlc = ref_model(**inp_c).logits[0, plen-1:-1, :].float().clamp(-100, 100)
                rlr = ref_model(**inp_r).logits[0, plen-1:-1, :].float().clamp(-100, 100)
                rlp_c = F.log_softmax(rlc, dim=-1).gather(1, inp_c['input_ids'][0, plen:].unsqueeze(1)).squeeze()
                rlp_r = F.log_softmax(rlr, dim=-1).gather(1, inp_r['input_ids'][0, plen:].unsqueeze(1)).squeeze()
                if rlp_c.dim() == 0: rlp_c = rlp_c.unsqueeze(0)
                if rlp_r.dim() == 0: rlp_r = rlp_r.unsqueeze(0)
            diff = 0.05 * ((lp_c.sum() - rlp_c.sum()) - (lp_r.sum() - rlp_r.sum()))
            loss = -F.logsigmoid(diff)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dpo_params, 1.0)
            optimizer.step()
    model.eval()
    d_train = evaluate(model, tok, TRAIN_PAIRS, "Train")
    d_test = evaluate(model, tok, TEST_PAIRS, "Test")
    all_results['dpo_baseline'] = {'train': d_train, 'test': d_test}
    del model, ref_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save
    out = {'phase': '130', 'name': 'Knowledge/Skill LoRA Decoupling'}
    for k, v in all_results.items():
        out[k] = {sub: {kk: vv for kk, vv in sv.items() if kk != 'details'}
                  if isinstance(sv, dict) and 'details' in sv else sv
                  for sub, sv in v.items()} if isinstance(v, dict) else v
    with open(os.path.join(RESULTS_DIR, 'phase130_knowledge_lora.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Accuracy comparison
    ax = axes[0]
    configs = ['Baseline', 'DPO\n(L22-23 MLP)', 'Knowledge\nLoRA\n(FFN L1-10)',
               'Skill LoRA\n(Attn L22-23)', 'Combined\nK+S LoRA']
    keys = ['baseline', 'dpo_baseline', 'knowledge_lora', 'skill_lora', 'combined_lora']
    word_accs = [all_results[k]['train']['word_acc'] for k in keys]
    num_accs = [all_results[k]['train']['num_acc'] for k in keys]
    x = np.arange(len(configs))
    w = 0.35
    b1 = ax.bar(x - w/2, word_accs, w, label='Word Facts', color='#3498db', alpha=0.8)
    b2 = ax.bar(x + w/2, num_accs, w, label='Numerical Facts', color='#e74c3c', alpha=0.8)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.02, f'{h:.0%}',
                   ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=8)
    ax.set_ylabel('Train Accuracy'); ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Train Accuracy by Config', fontweight='bold')

    # Panel 2: Test accuracy
    ax = axes[1]
    word_test = [all_results[k]['test']['word_acc'] for k in keys]
    num_test = [all_results[k]['test']['num_acc'] for k in keys]
    b1 = ax.bar(x - w/2, word_test, w, label='Word Facts', color='#3498db', alpha=0.8)
    b2 = ax.bar(x + w/2, num_test, w, label='Numerical Facts', color='#e74c3c', alpha=0.8)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.02, f'{h:.0%}',
                   ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=8)
    ax.set_ylabel('Test Accuracy'); ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Test Accuracy by Config', fontweight='bold')

    # Panel 3: Training loss curves
    ax = axes[2]
    for k, label, color in [('knowledge_lora', 'Knowledge LoRA', '#e74c3c'),
                             ('skill_lora', 'Skill LoRA', '#3498db'),
                             ('combined_lora', 'Combined K+S', '#2ecc71')]:
        if 'losses' in all_results[k]:
            ax.plot(all_results[k]['losses'], label=label, color=color, linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.set_title('Training Loss', fontweight='bold')

    fig.suptitle('Phase 130: Knowledge/Skill LoRA Decoupling - Can FFN Injection Bypass Numerical Immunity?',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase130_knowledge_lora.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === SUMMARY ===")
    for k in keys:
        r = all_results[k]
        print(f"    {k:20s}: train_word={r['train']['word_acc']:.0%} "
              f"train_num={r['train']['num_acc']:.0%} "
              f"test_word={r['test']['word_acc']:.0%} "
              f"test_num={r['test']['num_acc']:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 130] Complete.")

if __name__ == '__main__':
    main()
