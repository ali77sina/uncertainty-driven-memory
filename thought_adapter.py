"""
Thought Adapter for Qwen 0.5B
===============================

Freeze Qwen, add a small trainable adapter between transformer layers
that injects corrective thought vectors when the model is uncertain.

Train on synthetic math problems. Compare base Qwen vs Qwen + adapter.

Run:  CUDA_VISIBLE_DEVICES=0 python3 thought_adapter.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import re

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_LAYER = 12  # inject after this layer (model has 24 layers)
THOUGHT_DIM = 64
LR = 1e-4
EPOCHS = 5
BATCH = 4
TRAIN_SIZE = 500
TEST_SIZE = 100
MAX_NEW_TOKENS = 150

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Synthetic math problems ──────────────────────────────────────────────────

def make_problems(n, rng, difficulty="mixed"):
    problems = []
    for _ in range(n):
        if difficulty == "mixed":
            d = rng.choice(["easy", "medium", "hard"])
        else:
            d = difficulty

        if d == "easy":
            a, b = rng.randint(10, 50), rng.randint(10, 50)
            op = rng.choice(["+", "-"])
            ans = a + b if op == "+" else a - b
            q = f"What is {a} {op} {b}?"

        elif d == "medium":
            a = rng.randint(10, 100)
            b = rng.randint(10, 100)
            c = rng.randint(5, 50)
            item = rng.choice(["apples", "books", "coins"])
            ans = a + b - c
            q = f"Tom had {a} {item}. He got {b} more. Then he gave away {c}. How many does he have?"

        else:  # hard
            a = rng.randint(10, 80)
            b = rng.randint(10, 80)
            c = rng.randint(5, 40)
            d_val = rng.randint(5, 30)
            item = rng.choice(["apples", "books", "coins"])
            ans = a + b - c + d_val
            q = (f"Tom had {a} {item}. He got {b} more. He gave away {c}. "
                 f"Then he found {d_val} more. How many does he have?")

        prompt = f"Q: {q}\nA: Let's solve step by step.\n"
        problems.append({"prompt": prompt, "answer": ans})
    return problems


# ── Thought Adapter ──────────────────────────────────────────────────────────

class ThoughtAdapter(nn.Module):
    """
    Small module that sits between transformer layers.
    Compresses hidden states to thought vectors, checks uncertainty,
    and injects corrective signal when uncertain.
    """
    def __init__(self, hidden_dim, thought_dim=THOUGHT_DIM):
        super().__init__()
        self.compress = nn.Linear(hidden_dim, thought_dim)
        self.expand = nn.Linear(thought_dim, hidden_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(thought_dim, thought_dim),
            nn.ReLU(),
            nn.Linear(thought_dim, 1),
            nn.Sigmoid(),
        )
        # running stats for uncertainty
        self.register_buffer("running_mean", torch.zeros(thought_dim))
        self.register_buffer("running_var", torch.ones(thought_dim))
        self.register_buffer("n_seen", torch.tensor(0.0))

    def forward(self, hidden_states):
        # compress to thought space
        thoughts = torch.tanh(self.compress(hidden_states))  # (B, seq_len, thought_dim)

        # expand back and gate
        gate = self.gate_net(thoughts)  # (B, seq_len, 1)
        correction = self.expand(thoughts)

        return hidden_states + gate * correction


# ── Model with adapter ───────────────────────────────────────────────────────

class QwenWithAdapter(nn.Module):
    def __init__(self, base_model, adapter, layer_idx):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.layer_idx = layer_idx
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        layers = self.base_model.model.layers
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                modified = self.adapter(hidden)
                return (modified,) + output[1:]
            else:
                return self.adapter(output)
        self._hook = layers[self.layer_idx].register_forward_hook(hook_fn)

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_answer(text):
    matches = re.findall(r"(?:answer is|total is|he has|she has|result is|=)\s*(-?\d+)", text.lower())
    if matches:
        return int(matches[-1])
    numbers = re.findall(r"(-?\d+)", text)
    if numbers:
        return int(numbers[-1])
    return None


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, problems, label=""):
    model.eval()
    correct = 0
    total_tokens = 0

    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["prompt"], return_tensors="pt").to(DEVICE)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
            )
        generated = outputs[0][input_len:]
        total_tokens += len(generated)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        pred = extract_answer(text)

        if pred == prob["answer"]:
            correct += 1

        if i < 3:
            print(f"  [{label}] Q: ...{prob['prompt'][-40:].strip()}")
            print(f"  [{label}] Pred={pred}, True={prob['answer']}, {'OK' if pred == prob['answer'] else 'WRONG'}")

    acc = correct / len(problems)
    avg_tokens = total_tokens / len(problems)
    return acc, avg_tokens


# ── Training ─────────────────────────────────────────────────────────────────

def train_adapter(wrapped_model, tokenizer, train_problems):
    """Train only the adapter parameters using teacher forcing."""
    adapter_params = list(wrapped_model.adapter.parameters())
    opt = torch.optim.Adam(adapter_params, lr=LR)

    for ep in range(EPOCHS):
        wrapped_model.train()
        wrapped_model.base_model.eval()  # keep base frozen
        total_loss = 0
        perm = np.random.permutation(len(train_problems))

        for i in range(0, len(train_problems), BATCH):
            idx = perm[i:i+BATCH]
            batch = [train_problems[j] for j in idx]

            # create full sequences: prompt + correct answer
            texts = []
            for prob in batch:
                texts.append(f"{prob['prompt']}The answer is {prob['answer']}.")

            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=256).to(DEVICE)
            labels = enc["input_ids"].clone()
            # only compute loss on answer tokens (mask prompt)
            for j, prob in enumerate(batch):
                prompt_len = len(tokenizer.encode(prob["prompt"]))
                labels[j, :prompt_len] = -100

            opt.zero_grad()
            outputs = wrapped_model(**enc, labels=labels)
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(adapter_params, 1.0)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(train_problems) / BATCH)
        print(f"  Epoch {ep+1}/{EPOCHS}  loss={avg_loss:.4f}")

    wrapped_model.eval()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Device: {DEVICE}")
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(DEVICE)

    # freeze base model
    for p in base_model.parameters():
        p.requires_grad = False

    hidden_dim = base_model.config.hidden_size
    n_layers = base_model.config.num_hidden_layers
    print(f"  Hidden dim: {hidden_dim}, Layers: {n_layers}")

    rng = np.random.RandomState(SEED)
    train_problems = make_problems(TRAIN_SIZE, rng)
    test_easy = make_problems(TEST_SIZE, np.random.RandomState(SEED + 100), "easy")
    test_medium = make_problems(TEST_SIZE, np.random.RandomState(SEED + 200), "medium")
    test_hard = make_problems(TEST_SIZE, np.random.RandomState(SEED + 300), "hard")

    # baseline: evaluate base Qwen
    print("\n=== Base Qwen (no adapter) ===")
    base_acc_easy, base_tok_easy = evaluate(base_model, tokenizer, test_easy, "base-easy")
    base_acc_med, base_tok_med = evaluate(base_model, tokenizer, test_medium, "base-med")
    base_acc_hard, base_tok_hard = evaluate(base_model, tokenizer, test_hard, "base-hard")
    print(f"  Easy:   acc={base_acc_easy:.3f}  tokens={base_tok_easy:.0f}")
    print(f"  Medium: acc={base_acc_med:.3f}  tokens={base_tok_med:.0f}")
    print(f"  Hard:   acc={base_acc_hard:.3f}  tokens={base_tok_hard:.0f}")

    # create adapter
    adapter_layer = n_layers // 2
    print(f"\n=== Training adapter at layer {adapter_layer} ===")
    adapter = ThoughtAdapter(hidden_dim).to(DEVICE).half()
    n_adapter_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Adapter params: {n_adapter_params:,} ({n_adapter_params/1e6:.2f}M)")

    wrapped = QwenWithAdapter(base_model, adapter, adapter_layer)

    t0 = time.perf_counter()
    train_adapter(wrapped, tokenizer, train_problems)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.1f}s")

    # evaluate with adapter
    print("\n=== Qwen + Adapter ===")
    adp_acc_easy, adp_tok_easy = evaluate(wrapped, tokenizer, test_easy, "adp-easy")
    adp_acc_med, adp_tok_med = evaluate(wrapped, tokenizer, test_medium, "adp-med")
    adp_acc_hard, adp_tok_hard = evaluate(wrapped, tokenizer, test_hard, "adp-hard")
    print(f"  Easy:   acc={adp_acc_easy:.3f}  tokens={adp_tok_easy:.0f}")
    print(f"  Medium: acc={adp_acc_med:.3f}  tokens={adp_tok_med:.0f}")
    print(f"  Hard:   acc={adp_acc_hard:.3f}  tokens={adp_tok_hard:.0f}")

    # summary
    print(f"\n{'='*60}")
    print(f"{'':20s} {'Easy':>8s} {'Medium':>8s} {'Hard':>8s}")
    print(f"{'-'*60}")
    print(f"{'Base Qwen':20s} {base_acc_easy:8.3f} {base_acc_med:8.3f} {base_acc_hard:8.3f}")
    print(f"{'Qwen + Adapter':20s} {adp_acc_easy:8.3f} {adp_acc_med:8.3f} {adp_acc_hard:8.3f}")
    print(f"{'='*60}")

    results = {
        "base": {"easy": base_acc_easy, "medium": base_acc_med, "hard": base_acc_hard,
                 "tokens": {"easy": base_tok_easy, "medium": base_tok_med, "hard": base_tok_hard}},
        "adapter": {"easy": adp_acc_easy, "medium": adp_acc_med, "hard": adp_acc_hard,
                    "tokens": {"easy": adp_tok_easy, "medium": adp_tok_med, "hard": adp_tok_hard},
                    "params": n_adapter_params, "train_time": train_time},
    }
    with open("results_adapter.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results_adapter.json")


if __name__ == "__main__":
    main()
