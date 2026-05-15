# -*- coding: utf-8 -*-
"""
Phase 115: KAT-LoRA Orthogonalization
Separate Knowledge (front) and Skill (back) via orthogonal LoRA.

Theory (from P1, P108):
  P1: Fact and Skill subspaces separated by only 1.2 degrees.
  P108: Facts live in front-half MLPs, grammar in back-half MLPs.
  -> Attach Knowledge LoRA to front (<0.94), Skill LoRA to back (>0.94)
  -> Orthogonalization penalty forces 90-degree separation.

Model: Qwen2.5-0.5B (GPU, fast training)
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

# Training facts (prompt, completion)
TRAIN_FACTS = [
    ("The capital of Japan is", " Tokyo"),
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("The chemical symbol for gold is", " Au"),
    ("The author of Romeo and Juliet is", " William"),
    ("The first president of the United States was", " George"),
]

# Test facts (held out)
TEST_FACTS = [
    ("The chemical formula for water is", " H"),
    ("The boiling point of water is", " 100"),
    ("The atomic number of carbon is", " 6"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("The speed of sound is approximately", " 343"),
    ("The tallest mountain in the world is", " Mount"),
    ("The currency of the United Kingdom is the", " pound"),
    ("The speed of light is approximately", " 299"),
    ("Albert Einstein was born in", " Ul"),
    ("Photosynthesis converts sunlight into", " chemical"),
]

# Fluency test (should NOT degrade)
FLUENCY_PROMPTS = [
    "Once upon a time, there was a",
    "The weather today is",
    "In the year 2025,",
    "The most important thing in life is",
    "Scientists have discovered that",
]


def evaluate_accuracy(model, tok, facts):
    """Evaluate factual accuracy."""
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
    """Evaluate perplexity on fluency prompts."""
    total_loss = 0
    total_tokens = 0
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        total_loss += out.loss.item() * inp['input_ids'].shape[1]
        total_tokens += inp['input_ids'].shape[1]
    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


def compute_orthogonality(knowledge_params, skill_params):
    """Compute cosine similarity between Knowledge and Skill LoRA weight spaces."""
    k_flat = torch.cat([p.flatten() for p in knowledge_params if p.requires_grad])
    s_flat = torch.cat([p.flatten() for p in skill_params if p.requires_grad])
    min_len = min(len(k_flat), len(s_flat))
    if min_len == 0:
        return 0.0
    k_flat = k_flat[:min_len]
    s_flat = s_flat[:min_len]
    cos = torch.nn.functional.cosine_similarity(k_flat.unsqueeze(0), s_flat.unsqueeze(0))
    return cos.item()


def main():
    print("[P115] KAT-LoRA Orthogonalization")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    model_id = 'Qwen/Qwen2.5-0.5B'
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float16
    ).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    n_layers = 24
    boundary = int(n_layers * 0.94)  # Layer 22
    print(f"  Boundary (0.94): L{boundary}")
    print(f"  Front (Knowledge): L0-L{boundary-1}, Back (Skill): L{boundary}-L{n_layers-1}")

    # Baseline
    baseline_train = evaluate_accuracy(base_model, tok, TRAIN_FACTS)
    baseline_test = evaluate_accuracy(base_model, tok, TEST_FACTS)
    baseline_ppl = evaluate_perplexity(base_model, tok, FLUENCY_PROMPTS)
    print(f"  Baseline: train={baseline_train:.0%}, test={baseline_test:.0%}, ppl={baseline_ppl:.1f}")

    results = {
        'baseline': {
            'train_acc': baseline_train,
            'test_acc': baseline_test,
            'perplexity': baseline_ppl,
        }
    }

    # Experiment conditions
    conditions = [
        ('knowledge_only', True, False, 0.0),   # Only front LoRA
        ('skill_only', False, True, 0.0),        # Only back LoRA
        ('both_no_orth', True, True, 0.0),       # Both, no orthogonalization
        ('both_orth_01', True, True, 0.1),       # Both, weak orthogonalization
        ('both_orth_05', True, True, 0.5),       # Both, medium orthogonalization
        ('both_orth_10', True, True, 1.0),       # Both, strong orthogonalization
    ]

    for cond_name, use_knowledge, use_skill, orth_lambda in conditions:
        print(f"\n  === Condition: {cond_name} (orth_lambda={orth_lambda}) ===")

        # Reload fresh model
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.float16
        ).to(DEVICE)

        # Determine target modules
        knowledge_layers = [f"model.layers.{i}" for i in range(boundary)]
        skill_layers = [f"model.layers.{i}" for i in range(boundary, n_layers)]

        target_modules = []
        if use_knowledge:
            target_modules.extend([f"model.layers.{i}.mlp.gate_proj" for i in range(boundary)])
            target_modules.extend([f"model.layers.{i}.mlp.up_proj" for i in range(boundary)])
        if use_skill:
            target_modules.extend([f"model.layers.{i}.mlp.gate_proj" for i in range(boundary, n_layers)])
            target_modules.extend([f"model.layers.{i}.mlp.up_proj" for i in range(boundary, n_layers)])

        if not target_modules:
            continue

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.train()

        # Separate knowledge vs skill parameters
        knowledge_params = []
        skill_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_num = None
                for i in range(n_layers):
                    if f"layers.{i}." in name:
                        layer_num = i
                        break
                if layer_num is not None:
                    if layer_num < boundary:
                        knowledge_params.append(param)
                    else:
                        skill_params.append(param)

        # Training
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        n_steps = 50
        history = {'loss': [], 'orth_loss': [], 'cosine': []}

        for step in range(n_steps):
            total_loss = torch.tensor(0.0, device=DEVICE)

            # Standard causal LM loss on facts
            for prompt, answer in TRAIN_FACTS:
                text = prompt + answer
                inp = tok(text, return_tensors='pt', padding=True).to(DEVICE)
                out = model(**inp, labels=inp['input_ids'])
                total_loss = total_loss + out.loss

            total_loss = total_loss / len(TRAIN_FACTS)

            # Orthogonalization penalty
            orth_loss = torch.tensor(0.0, device=DEVICE)
            cosine_sim = 0.0
            if orth_lambda > 0 and knowledge_params and skill_params:
                k_flat = torch.cat([p.flatten().float() for p in knowledge_params])
                s_flat = torch.cat([p.flatten().float() for p in skill_params])
                min_len = min(len(k_flat), len(s_flat))
                if min_len > 0:
                    k_sub = k_flat[:min_len]
                    s_sub = s_flat[:min_len]
                    # Penalize cosine similarity (want it to be 0 = orthogonal)
                    cos = torch.nn.functional.cosine_similarity(
                        k_sub.unsqueeze(0), s_sub.unsqueeze(0))
                    orth_loss = orth_lambda * cos.abs()
                    cosine_sim = cos.item()

            final_loss = total_loss + orth_loss

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            history['loss'].append(total_loss.item())
            history['orth_loss'].append(orth_loss.item() if isinstance(orth_loss, torch.Tensor) else orth_loss)
            history['cosine'].append(cosine_sim)

            if step % 10 == 0 or step == n_steps - 1:
                print(f"    Step {step}: loss={total_loss.item():.3f}, orth={orth_loss.item() if isinstance(orth_loss, torch.Tensor) else 0:.4f}, cos={cosine_sim:.4f}")

        # Evaluate
        model.eval()
        train_acc = evaluate_accuracy(model, tok, TRAIN_FACTS)
        test_acc = evaluate_accuracy(model, tok, TEST_FACTS)
        ppl = evaluate_perplexity(model, tok, FLUENCY_PROMPTS)

        final_cosine = compute_orthogonality(knowledge_params, skill_params) if knowledge_params and skill_params else 0.0

        print(f"    Result: train={train_acc:.0%}, test={test_acc:.0%}, ppl={ppl:.1f}, cosine={final_cosine:.4f}")

        results[cond_name] = {
            'use_knowledge': use_knowledge,
            'use_skill': use_skill,
            'orth_lambda': orth_lambda,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'perplexity': ppl,
            'final_cosine': final_cosine,
            'history': history,
        }

        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")

    # Save
    out = {'phase': 115, 'name': 'KAT-LoRA Orthogonalization',
           'model': model_id, 'boundary_layer': boundary, 'results': results}
    with open(os.path.join(RESULTS_DIR, 'phase115_lora_orth.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Accuracy comparison
    ax = axes[0, 0]
    cond_names = [k for k in results if k != 'baseline']
    x = np.arange(len(cond_names) + 1)
    train_accs = [results['baseline']['train_acc']] + [results[c]['train_acc'] for c in cond_names]
    test_accs = [results['baseline']['test_acc']] + [results[c]['test_acc'] for c in cond_names]
    labels = ['Baseline'] + cond_names
    w = 0.35
    ax.bar(x - w/2, train_accs, w, label='Train', color='#2ecc71', alpha=0.8)
    ax.bar(x + w/2, test_accs, w, label='Test', color='#3498db', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Accuracy')
    ax.set_title('Factual Accuracy: Knowledge vs Skill LoRA')
    ax.legend()

    # Panel 2: Perplexity
    ax = axes[0, 1]
    ppls = [results['baseline']['perplexity']] + [results[c]['perplexity'] for c in cond_names]
    bars = ax.bar(x, ppls, color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Perplexity')
    ax.set_title('Fluency (Perplexity) - Lower is Better')

    # Panel 3: Cosine similarity trajectory (for orth conditions)
    ax = axes[1, 0]
    for cond in cond_names:
        if results[cond].get('orth_lambda', 0) > 0:
            ax.plot(results[cond]['history']['cosine'], label=f"{cond}", linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Knowledge-Skill Orthogonalization')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Orthogonal (0)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Loss trajectories
    ax = axes[1, 1]
    for cond in cond_names:
        ax.plot(results[cond]['history']['loss'], label=cond, linewidth=1.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Phase 115: KAT-LoRA Orthogonalization (Qwen2.5-0.5B)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase115_lora_orth.png'), dpi=150)
    plt.close()

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[Phase 115] Complete.")


if __name__ == '__main__':
    main()
