# -*- coding: utf-8 -*-
"""
Phase 117: Targeted F-DPO (MLP Exorcism)
Fine-tune ONLY the back-half MLPs with factuality DPO.

Theory (from P108 MLP Autopsy):
  Front 94% layers store facts. Back 6% contain "Grammar Police"
  (suppressor MLPs) that distort fact retrieval for fluency.
  By freezing front 94% and DPO-training only the back 6%,
  we can re-align suppressors from "truth killers" to "truth guards".

Method:
  1. Generate fact/hallucination pairs from base model
  2. Freeze all layers except back-half MLPs
  3. DPO training with chosen=correct, rejected=hallucination
  4. Measure accuracy recovery

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, copy
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# DPO pairs: (prompt, chosen_completion, rejected_completion)
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

# Test facts (not in DPO training)
TEST_FACTS = [
    ("The tallest mountain in the world is", " Mount"),
    ("The currency of the United Kingdom is the", " pound"),
    ("The speed of sound is approximately", " 343"),
    ("Albert Einstein was born in", " Ul"),
    ("Photosynthesis converts sunlight into", " chemical"),
]

FLUENCY_PROMPTS = [
    "Once upon a time, there was a",
    "The weather today is",
    "In the year 2025,",
    "The most important thing in life is",
    "Scientists have discovered that",
]


def evaluate_accuracy(model, tok, facts):
    correct = 0
    for prompt, answer in facts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        fact_tokens = tok.encode(answer)
        fact_id = fact_tokens[-1] if len(fact_tokens) > 0 else 0
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
        if logits.argmax().item() == fact_id:
            correct += 1
    return correct / len(facts)


def evaluate_perplexity(model, tok, prompts):
    total_loss = 0
    total_tokens = 0
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        total_loss += out.loss.item() * inp['input_ids'].shape[1]
        total_tokens += inp['input_ids'].shape[1]
    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


def get_log_prob(model, tok, prompt, completion):
    """Get log probability of completion given prompt."""
    text = prompt + completion
    inp = tok(text, return_tensors='pt').to(DEVICE)
    prompt_ids = tok(prompt, return_tensors='pt')['input_ids']
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        logits = model(**inp).logits

    # Log probs for completion tokens only
    log_probs = F.log_softmax(logits[0, prompt_len-1:-1, :], dim=-1)
    target_ids = inp['input_ids'][0, prompt_len:]
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze()

    if token_log_probs.dim() == 0:
        return token_log_probs.item()
    return token_log_probs.sum().item()


def dpo_loss(model, ref_model, tok, prompt, chosen, rejected, beta=0.1):
    """Compute DPO loss for a single pair."""
    # Policy log probs
    text_chosen = prompt + chosen
    text_rejected = prompt + rejected

    inp_c = tok(text_chosen, return_tensors='pt').to(DEVICE)
    inp_r = tok(text_rejected, return_tensors='pt').to(DEVICE)
    prompt_ids = tok(prompt, return_tensors='pt')['input_ids']
    prompt_len = prompt_ids.shape[1]

    # Forward pass (policy)
    logits_c = model(**inp_c).logits
    logits_r = model(**inp_r).logits

    # Policy log probs for completion
    # Clamp logits to prevent inf/-inf before log_softmax
    logits_c_clamped = logits_c[0, prompt_len-1:-1, :].float().clamp(-100, 100)
    logits_r_clamped = logits_r[0, prompt_len-1:-1, :].float().clamp(-100, 100)
    lp_c = F.log_softmax(logits_c_clamped, dim=-1)
    lp_r = F.log_softmax(logits_r_clamped, dim=-1)

    target_c = inp_c['input_ids'][0, prompt_len:]
    target_r = inp_r['input_ids'][0, prompt_len:]

    policy_logp_c = lp_c.gather(1, target_c.unsqueeze(1)).squeeze()
    policy_logp_r = lp_r.gather(1, target_r.unsqueeze(1)).squeeze()
    if policy_logp_c.dim() == 0:
        policy_logp_c = policy_logp_c.unsqueeze(0)
    if policy_logp_r.dim() == 0:
        policy_logp_r = policy_logp_r.unsqueeze(0)
    policy_logp_c = policy_logp_c.sum()
    policy_logp_r = policy_logp_r.sum()

    # Reference log probs
    with torch.no_grad():
        ref_logits_c = ref_model(**inp_c).logits
        ref_logits_r = ref_model(**inp_r).logits
        ref_c_clamped = ref_logits_c[0, prompt_len-1:-1, :].float().clamp(-100, 100)
        ref_r_clamped = ref_logits_r[0, prompt_len-1:-1, :].float().clamp(-100, 100)
        ref_lp_c = F.log_softmax(ref_c_clamped, dim=-1)
        ref_lp_r = F.log_softmax(ref_r_clamped, dim=-1)
        ref_logp_c = ref_lp_c.gather(1, target_c.unsqueeze(1)).squeeze()
        ref_logp_r = ref_lp_r.gather(1, target_r.unsqueeze(1)).squeeze()
        if ref_logp_c.dim() == 0:
            ref_logp_c = ref_logp_c.unsqueeze(0)
        if ref_logp_r.dim() == 0:
            ref_logp_r = ref_logp_r.unsqueeze(0)
        ref_logp_c = ref_logp_c.sum()
        ref_logp_r = ref_logp_r.sum()

    # DPO loss
    logits_diff = beta * ((policy_logp_c - ref_logp_c) - (policy_logp_r - ref_logp_r))
    loss = -F.logsigmoid(logits_diff)
    return loss


def main():
    print("[P117] Targeted F-DPO (MLP Exorcism)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    n_layers = 24
    boundary = int(n_layers * 0.94)  # Layer 22
    print(f"  Boundary (0.94): L{boundary}")
    print(f"  Frozen: L0-L{boundary-1} (Knowledge)")
    print(f"  Trainable: L{boundary}-L{n_layers-1} (Skill/Suppressor)")

    # Conditions to test
    conditions = [
        ('full_model_dpo', None),        # DPO on all parameters
        ('back_only_dpo', boundary),      # DPO on back MLPs only (the exorcism)
        ('front_only_dpo', -boundary),    # DPO on front MLPs only (control)
    ]

    all_results = {}

    for cond_name, freeze_boundary in conditions:
        print(f"\n  === Condition: {cond_name} ===")

        # Load fresh model + reference
        # NOTE: float32 required for DPO stability (fp16 causes NaN in v1)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32
        ).to(DEVICE)
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float32
        ).eval().to(DEVICE)

        # Baseline before DPO
        model.eval()
        base_train = evaluate_accuracy(model, tok,
            [(p, c) for p, c, r in DPO_PAIRS])
        base_test = evaluate_accuracy(model, tok, TEST_FACTS)
        base_ppl = evaluate_perplexity(model, tok, FLUENCY_PROMPTS)
        print(f"    Before: train={base_train:.0%}, test={base_test:.0%}, ppl={base_ppl:.1f}")

        # Freeze layers
        model.train()
        trainable_count = 0
        total_count = 0
        for name, param in model.named_parameters():
            total_count += param.numel()
            should_train = True

            if freeze_boundary is not None:
                should_train = False  # Default freeze
                for i in range(n_layers):
                    if f"layers.{i}." in name and "mlp" in name:
                        if freeze_boundary > 0:
                            # back_only: train layers >= boundary
                            if i >= freeze_boundary:
                                should_train = True
                        else:
                            # front_only: train layers < abs(boundary)
                            if i < abs(freeze_boundary):
                                should_train = True

            param.requires_grad = should_train
            if should_train:
                trainable_count += param.numel()

        print(f"    Trainable: {trainable_count:,}/{total_count:,} ({100*trainable_count/total_count:.1f}%)")

        # DPO Training
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-5)

        n_epochs = 5
        history = {'loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(n_epochs):
            epoch_loss = 0
            for prompt, chosen, rejected in DPO_PAIRS:
                loss = dpo_loss(model, ref_model, tok, prompt, chosen, rejected, beta=0.1)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(DPO_PAIRS)
            history['loss'].append(avg_loss)

            # Eval
            model.eval()
            train_acc = evaluate_accuracy(model, tok,
                [(p, c) for p, c, r in DPO_PAIRS])
            test_acc = evaluate_accuracy(model, tok, TEST_FACTS)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            print(f"    Epoch {epoch}: loss={avg_loss:.4f}, train={train_acc:.0%}, test={test_acc:.0%}")
            model.train()

        # Final eval
        model.eval()
        final_train = evaluate_accuracy(model, tok,
            [(p, c) for p, c, r in DPO_PAIRS])
        final_test = evaluate_accuracy(model, tok, TEST_FACTS)
        final_ppl = evaluate_perplexity(model, tok, FLUENCY_PROMPTS)
        print(f"    After: train={final_train:.0%}, test={final_test:.0%}, ppl={final_ppl:.1f}")

        all_results[cond_name] = {
            'freeze_boundary': freeze_boundary,
            'trainable_params': trainable_count,
            'total_params': total_count,
            'trainable_pct': 100*trainable_count/total_count,
            'before': {'train': base_train, 'test': base_test, 'ppl': base_ppl},
            'after': {'train': final_train, 'test': final_test, 'ppl': final_ppl},
            'history': history,
        }

        del model, ref_model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")

    # Save
    out = {'phase': 117, 'name': 'Targeted F-DPO (MLP Exorcism)',
           'model': model_id, 'boundary_layer': boundary, 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase117_fdpo.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Before vs After accuracy
    ax = axes[0, 0]
    conds = list(all_results.keys())
    x = np.arange(len(conds))
    before_train = [all_results[c]['before']['train'] for c in conds]
    after_train = [all_results[c]['after']['train'] for c in conds]
    before_test = [all_results[c]['before']['test'] for c in conds]
    after_test = [all_results[c]['after']['test'] for c in conds]
    w = 0.2
    ax.bar(x - 1.5*w, before_train, w, label='Before (train)', color='#bdc3c7')
    ax.bar(x - 0.5*w, after_train, w, label='After (train)', color='#2ecc71')
    ax.bar(x + 0.5*w, before_test, w, label='Before (test)', color='#95a5a6')
    ax.bar(x + 1.5*w, after_test, w, label='After (test)', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('F-DPO: Before vs After')
    ax.legend(fontsize=7)

    # Panel 2: Perplexity change
    ax = axes[0, 1]
    before_ppl = [all_results[c]['before']['ppl'] for c in conds]
    after_ppl = [all_results[c]['after']['ppl'] for c in conds]
    ax.bar(x - 0.2, before_ppl, 0.4, label='Before', color='#bdc3c7')
    ax.bar(x + 0.2, after_ppl, 0.4, label='After', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('Perplexity')
    ax.set_title('Fluency Impact')
    ax.legend()

    # Panel 3: Training loss curves
    ax = axes[1, 0]
    for cond in conds:
        ax.plot(all_results[cond]['history']['loss'], 'o-', label=cond, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('DPO Loss')
    ax.set_title('DPO Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Accuracy trajectories
    ax = axes[1, 1]
    for cond in conds:
        ax.plot(all_results[cond]['history']['train_acc'], 'o-', label=f"{cond} (train)", linewidth=2)
        ax.plot(all_results[cond]['history']['test_acc'], 's--', label=f"{cond} (test)", linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy During DPO Training')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Phase 117: Targeted F-DPO - MLP Exorcism (Qwen2.5-0.5B)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase117_fdpo.png'), dpi=150)
    plt.close()
    print("[Phase 117] Complete.")


if __name__ == '__main__':
    main()
