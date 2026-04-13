"""
Thought Adapter v2 — Fixed training + harder problems
=======================================================

Fixes:
  - Float32 adapter with proper casting (no NaN)
  - Much harder multi-step problems that Qwen actually fails on
  - Proper mixed-precision training with GradScaler

Run:  CUDA_VISIBLE_DEVICES=0 python3 thought_adapter_v2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import re
from torch.cuda.amp import GradScaler, autocast

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
THOUGHT_DIM = 128
LR = 5e-5
EPOCHS = 8
BATCH = 4
GRAD_ACCUM = 4
TRAIN_SIZE = 800
TEST_SIZE = 80
MAX_NEW_TOKENS = 250

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Hard multi-step problems ─────────────────────────────────────────────────

def make_hard_problems(n, rng):
    """Problems that require 4-7 steps with larger numbers and mixed ops."""
    problems = []
    for _ in range(n):
        difficulty = rng.choice(["multi_step", "percentage", "comparison", "sequential"])

        if difficulty == "multi_step":
            steps = rng.randint(4, 7)
            val = rng.randint(50, 500)
            desc_parts = [f"Start with {val}."]
            for _ in range(steps):
                op = rng.choice(["add", "sub", "mul", "div"])
                if op == "add":
                    n2 = rng.randint(10, 200)
                    val = val + n2
                    desc_parts.append(f"Add {n2}.")
                elif op == "sub":
                    n2 = rng.randint(1, max(2, min(200, val)))
                    val = val - n2
                    desc_parts.append(f"Subtract {n2}.")
                elif op == "mul":
                    n2 = rng.choice([2, 3, 4, 5])
                    val = val * n2
                    desc_parts.append(f"Multiply by {n2}.")
                elif op == "div":
                    n2 = rng.choice([2, 3, 4, 5])
                    val = val // n2
                    desc_parts.append(f"Divide by {n2} (round down).")
            q = " ".join(desc_parts) + " What is the final value?"
            ans = val

        elif difficulty == "percentage":
            price = rng.randint(100, 1000)
            disc1 = rng.choice([10, 15, 20, 25, 30])
            tax = rng.choice([5, 8, 10, 12])
            quantity = rng.randint(2, 6)
            discounted = price * (100 - disc1) // 100
            total = discounted * quantity
            with_tax = total * (100 + tax) // 100
            q = (f"An item costs ${price}. It has a {disc1}% discount. "
                 f"You buy {quantity} of them. Then {tax}% tax is added. "
                 f"What is the total cost in dollars? Round down at each step.")
            ans = with_tax

        elif difficulty == "comparison":
            a_start = rng.randint(100, 500)
            b_start = rng.randint(100, 500)
            a_val, b_val = a_start, b_start
            steps_desc = []
            for _ in range(rng.randint(3, 5)):
                who = rng.choice(["A", "B"])
                op = rng.choice(["gains", "loses"])
                amt = rng.randint(20, 150)
                if who == "A":
                    a_val = a_val + amt if op == "gains" else a_val - amt
                else:
                    b_val = b_val + amt if op == "gains" else b_val - amt
                steps_desc.append(f"{who} {op} {amt}.")
            q = (f"A starts with {a_start}. B starts with {b_start}. "
                 + " ".join(steps_desc)
                 + " What is A's value minus B's value?")
            ans = a_val - b_val

        else:  # sequential
            people = ["Alice", "Bob", "Carol"]
            vals = {p: rng.randint(10, 100) for p in people}
            desc = " ".join(f"{p} has {v}." for p, v in vals.items())
            transfers = []
            for _ in range(rng.randint(3, 6)):
                giver, receiver = rng.choice(people, 2, replace=False)
                amt = rng.randint(1, max(2, min(30, vals[giver])))
                vals[giver] -= amt
                vals[receiver] += amt
                transfers.append(f"{giver} gives {amt} to {receiver}.")
            target = rng.choice(people)
            q = desc + " " + " ".join(transfers) + f" How much does {target} have now?"
            ans = vals[target]

        prompt = f"Q: {q}\nA: Let me work through this step by step.\n"
        problems.append({"prompt": prompt, "answer": ans, "type": difficulty})
    return problems


# ── Adapter ──────────────────────────────────────────────────────────────────

class ThoughtAdapter(nn.Module):
    def __init__(self, hidden_dim, thought_dim=THOUGHT_DIM):
        super().__init__()
        self.compress = nn.Linear(hidden_dim, thought_dim)
        self.expand = nn.Linear(thought_dim, hidden_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(thought_dim, thought_dim),
            nn.GELU(),
            nn.Linear(thought_dim, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        h = hidden_states.float()
        h = self.norm(h)
        thoughts = torch.tanh(self.compress(h))
        gate = self.gate_net(thoughts)
        correction = self.expand(thoughts)
        out = hidden_states + (gate * correction).to(orig_dtype)
        return out


class QwenWithAdapter(nn.Module):
    def __init__(self, base_model, adapter, layer_idx):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.layer_idx = layer_idx
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


# ── Utils ────────────────────────────────────────────────────────────────────

def extract_answer(text):
    patterns = [
        r"(?:answer is|total is|final value is|result is|value is|has now|have now)[:\s]*\$?(-?\d+)",
        r"(?:=\s*)(-?\d+)(?:\s*(?:dollars|\.|$))",
        r"\$(-?\d+)",
    ]
    for p in patterns:
        matches = re.findall(p, text.lower())
        if matches:
            return int(matches[-1])
    numbers = re.findall(r"(-?\d+)", text.split("\n")[-1] if "\n" in text else text)
    if numbers:
        return int(numbers[-1])
    return None


def evaluate(model, tokenizer, problems, label=""):
    model.eval()
    correct = 0
    by_type = {}

    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["prompt"], return_tensors="pt").to(DEVICE)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            )
        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        pred = extract_answer(text)
        is_correct = pred == prob["answer"]
        correct += int(is_correct)

        t = prob.get("type", "unknown")
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0}
        by_type[t]["total"] += 1
        by_type[t]["correct"] += int(is_correct)

        if i < 3:
            print(f"  [{label}] pred={pred}, true={prob['answer']}, {'OK' if is_correct else 'WRONG'}")

    acc = correct / len(problems)
    print(f"  [{label}] Overall: {acc:.3f} ({correct}/{len(problems)})")
    for t, v in by_type.items():
        print(f"    {t}: {v['correct']}/{v['total']} = {v['correct']/v['total']:.3f}")
    return acc, by_type


def train_adapter(wrapped, tokenizer, train_problems):
    adapter_params = [p for p in wrapped.adapter.parameters()]
    opt = torch.optim.AdamW(adapter_params, lr=LR, weight_decay=0.01)
    scaler = GradScaler()

    for ep in range(EPOCHS):
        wrapped.adapter.train()
        wrapped.base_model.eval()
        perm = np.random.permutation(len(train_problems))
        total_loss = 0
        n_batches = 0
        opt.zero_grad()

        for i in range(0, len(train_problems), BATCH):
            idx = perm[i:i+BATCH]
            batch = [train_problems[j] for j in idx]

            texts = [f"{p['prompt']}The answer is {p['answer']}." for p in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=300).to(DEVICE)
            labels = enc["input_ids"].clone()
            for j, p in enumerate(batch):
                prompt_len = len(tokenizer.encode(p["prompt"]))
                labels[j, :prompt_len] = -100

            with autocast():
                outputs = wrapped(**enc, labels=labels)
                loss = outputs.loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (i // BATCH + 1) % GRAD_ACCUM == 0 or i + BATCH >= len(train_problems):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(adapter_params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

        avg = total_loss / n_batches
        print(f"  Epoch {ep+1}/{EPOCHS}  loss={avg:.4f}")

    wrapped.adapter.eval()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Device: {DEVICE}")
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16,
    ).to(DEVICE)
    for p in base_model.parameters():
        p.requires_grad = False

    hidden_dim = base_model.config.hidden_size
    n_layers = base_model.config.num_hidden_layers
    print(f"  Hidden: {hidden_dim}, Layers: {n_layers}")

    rng = np.random.RandomState(SEED)
    train_probs = make_hard_problems(TRAIN_SIZE, rng)
    test_probs = make_hard_problems(TEST_SIZE, np.random.RandomState(SEED + 999))

    print(f"\n=== Base Qwen ===")
    base_acc, base_by_type = evaluate(base_model, tokenizer, test_probs, "base")

    adapter_layer = n_layers // 2
    print(f"\n=== Training adapter (layer {adapter_layer}) ===")
    adapter = ThoughtAdapter(hidden_dim).to(DEVICE)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Adapter: {n_params:,} params")

    wrapped = QwenWithAdapter(base_model, adapter, adapter_layer)
    t0 = time.perf_counter()
    train_adapter(wrapped, tokenizer, train_probs)
    dt = time.perf_counter() - t0
    print(f"  Time: {dt:.1f}s")

    print(f"\n=== Qwen + Adapter ===")
    adp_acc, adp_by_type = evaluate(wrapped, tokenizer, test_probs, "adapter")

    print(f"\n{'='*50}")
    print(f"  Base Qwen:      {base_acc:.3f}")
    print(f"  Qwen + Adapter: {adp_acc:.3f}")
    print(f"{'='*50}")

    results = {"base": base_acc, "adapter": adp_acc, "adapter_params": n_params,
               "train_time": dt, "base_by_type": {k: v["correct"]/v["total"] for k,v in base_by_type.items()},
               "adapter_by_type": {k: v["correct"]/v["total"] for k,v in adp_by_type.items()}}
    with open("results_adapter_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results_adapter_v2.json")


if __name__ == "__main__":
    main()
