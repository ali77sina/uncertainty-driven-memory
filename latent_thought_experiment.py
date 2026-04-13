"""
Latent Thought Vectors for Long-Horizon Reasoning
===================================================

Compares three approaches to multi-step state tracking:

  1. Standard: transformer encodes full token sequence
  2. Token-CoT: generates intermediate answer tokens between steps
  3. Latent-Thought: generates continuous thought vectors between steps
     (with uncertainty-gated memory)

Task: track a number through a long chain of operations including
conditionals. Predict the final value.

Run on GPU:  python3 latent_thought_experiment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import json

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL = 128
NHEAD = 4
N_LAYERS = 2
THOUGHT_DIM = 64
N_THOUGHTS = 2  # thought vectors per step

TRAIN_SIZE = 20000
VAL_SIZE = 500
BATCH = 128
EPOCHS = 40
LR = 3e-4

N_VALUES = 200  # output range [0, 199]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Task: multi-step arithmetic with conditionals ────────────────────────────

OPS = [
    ("add_small", lambda x: (x + 3) % N_VALUES),
    ("add_big", lambda x: (x + 17) % N_VALUES),
    ("sub_small", lambda x: (x - 5) % N_VALUES),
    ("sub_big", lambda x: (x - 13) % N_VALUES),
    ("double", lambda x: (x * 2) % N_VALUES),
    ("half", lambda x: x // 2),
    ("if_gt50_add10_else_sub10", lambda x: (x + 10) % N_VALUES if x > 50 else (x - 10) % N_VALUES),
    ("if_even_double_else_add1", lambda x: (x * 2) % N_VALUES if x % 2 == 0 else (x + 1) % N_VALUES),
]

OP_TOKENS = {name: i + 2 for i, (name, _) in enumerate(OPS)}  # 0=pad, 1=sep
N_OP_TOKENS = len(OPS) + 2


def generate_chain(length, rng):
    start = rng.randint(0, N_VALUES)
    ops = []
    val = start
    intermediates = [val]
    for _ in range(length):
        op_idx = rng.randint(0, len(OPS))
        name, fn = OPS[op_idx]
        val = fn(val)
        ops.append(op_idx)
        intermediates.append(val)
    return start, ops, intermediates, val


def generate_dataset(n, lengths, rng):
    data = []
    for _ in range(n):
        length = rng.choice(lengths)
        start, ops, intermediates, answer = generate_chain(length, rng)
        data.append({
            "start": start,
            "ops": ops,
            "intermediates": intermediates,
            "answer": answer,
            "length": length,
        })
    return data


def encode_standard(example):
    """Encode as: [start_token, op1, op2, ..., sep] -> predict answer."""
    tokens = [example["start"] + N_OP_TOKENS]  # offset start value past op tokens
    for op in example["ops"]:
        tokens.append(op + 2)  # offset past pad and sep
    tokens.append(1)  # sep token
    return tokens


def encode_cot(example):
    """Encode as: [start, op1, =, intermediate1, op2, =, intermediate2, ..., sep]."""
    tokens = [example["start"] + N_OP_TOKENS]
    for i, op in enumerate(example["ops"]):
        tokens.append(op + 2)
        tokens.append(example["intermediates"][i + 1] + N_OP_TOKENS)  # intermediate value
    tokens.append(1)  # sep
    return tokens


# ── Models ───────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class StandardTransformer(nn.Module):
    """Baseline: encode full sequence, predict answer from final hidden state."""

    def __init__(self, vocab_size, d_model=D_MODEL, nhead=NHEAD, n_layers=N_LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, N_VALUES)

    def forward(self, x, pad_mask=None):
        h = self.pos(self.embed(x))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        pooled = h.mean(dim=1)
        return self.head(pooled)


class CoTTransformer(nn.Module):
    """Token chain-of-thought: same architecture but trained on sequences
    with intermediate answer tokens."""

    def __init__(self, vocab_size, d_model=D_MODEL, nhead=NHEAD, n_layers=N_LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=1024)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, N_VALUES)

    def forward(self, x, pad_mask=None):
        h = self.pos(self.embed(x))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        pooled = h.mean(dim=1)
        return self.head(pooled)


class LatentThoughtTransformer(nn.Module):
    """
    Processes operations one at a time. After each operation,
    produces a continuous thought vector that feeds into the next step.
    Thought vectors are the model's internal "non-verbal" reasoning.
    """

    def __init__(self, vocab_size, d_model=D_MODEL, nhead=NHEAD, n_layers=N_LAYERS,
                 thought_dim=THOUGHT_DIM, n_thoughts=N_THOUGHTS):
        super().__init__()
        self.d_model = d_model
        self.n_thoughts = n_thoughts
        self.thought_dim = thought_dim

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

        # thought production: hidden state -> thought vectors
        self.thought_proj = nn.Linear(d_model, n_thoughts * thought_dim)
        # thought injection: thought vectors -> embeddings that can be prepended
        self.thought_inject = nn.Linear(thought_dim, d_model)

        # uncertainty gate: determines how much to update thought memory
        self.register_buffer("running_mean", torch.zeros(n_thoughts * thought_dim))
        self.register_buffer("running_var", torch.ones(n_thoughts * thought_dim))
        self.register_buffer("n_updates", torch.tensor(0.0))
        self.gate = nn.Sequential(
            nn.Linear(n_thoughts * thought_dim + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.head = nn.Linear(d_model, N_VALUES)

    def compute_uncertainty(self, thought_flat):
        diff = thought_flat - self.running_mean
        return (diff.pow(2) / (self.running_var + 1e-6)).mean(dim=-1, keepdim=True)

    def update_stats(self, thought_flat):
        if self.training:
            with torch.no_grad():
                batch_mean = thought_flat.mean(dim=0)
                batch_var = thought_flat.var(dim=0)
                mom = min(0.1, 1.0 / (self.n_updates.item() + 1))
                self.running_mean.lerp_(batch_mean, mom)
                self.running_var.lerp_(batch_var, mom)
                self.n_updates += 1

    def forward(self, start_tokens, op_sequences, lengths):
        """
        start_tokens: (B,) - starting value tokens
        op_sequences: (B, max_len) - operation token sequences (padded)
        lengths: (B,) - actual number of operations per example
        """
        B = start_tokens.size(0)
        max_len = op_sequences.size(1)

        # initialize thought memory from start value
        start_emb = self.embed(start_tokens).unsqueeze(1)  # (B, 1, d_model)
        h = self.encoder(self.pos(start_emb))
        thought_flat = self.thought_proj(h.squeeze(1))  # (B, n_thoughts * thought_dim)
        thoughts = thought_flat.view(B, self.n_thoughts, self.thought_dim)

        # process operations one at a time
        for step in range(max_len):
            op_tokens = op_sequences[:, step]  # (B,)
            active = (step < lengths).float().unsqueeze(-1)  # (B, 1)

            # inject thought vectors as prefix tokens
            thought_embs = self.thought_inject(thoughts)  # (B, n_thoughts, d_model)
            op_emb = self.embed(op_tokens).unsqueeze(1)  # (B, 1, d_model)
            combined = torch.cat([thought_embs, op_emb], dim=1)  # (B, n_thoughts+1, d_model)
            combined = self.pos(combined)

            out = self.encoder(combined)

            # produce new thoughts from the output
            pooled = out.mean(dim=1)  # (B, d_model)
            new_thought_flat = self.thought_proj(pooled)
            new_thoughts = new_thought_flat.view(B, self.n_thoughts, self.thought_dim)

            # uncertainty-gated update
            uncertainty = self.compute_uncertainty(new_thought_flat)
            self.update_stats(new_thought_flat)
            gate_input = torch.cat([new_thought_flat, uncertainty], dim=-1)
            write_gate = self.gate(gate_input).unsqueeze(-1)  # (B, 1, 1)

            updated = write_gate * new_thoughts + (1 - write_gate) * thoughts
            thoughts = thoughts * (1 - active.unsqueeze(-1)) + updated * active.unsqueeze(-1)

        # final prediction from thought vectors
        final_embs = self.thought_inject(thoughts)
        final_out = self.encoder(self.pos(final_embs))
        pooled = final_out.mean(dim=1)
        return self.head(pooled)


# ── Training ─────────────────────────────────────────────────────────────────

def pad_sequences(seqs, pad_val=0):
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
    mask = torch.ones(len(seqs), max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = False
    return padded, mask


def train_standard_model(model, train_data, val_data, encode_fn, name="Model"):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best_acc = 0

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(train_data))
        total_loss = 0
        for i in range(0, len(train_data), BATCH):
            idx = perm[i:i+BATCH]
            batch = [train_data[j] for j in idx]
            seqs = [encode_fn(ex) for ex in batch]
            targets = torch.tensor([ex["answer"] for ex in batch]).to(DEVICE)
            x, mask = pad_sequences(seqs)
            x, mask = x.to(DEVICE), mask.to(DEVICE)

            opt.zero_grad()
            logits = model(x, pad_mask=mask)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()

        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            acc = eval_standard_model(model, val_data, encode_fn)
            best_acc = max(best_acc, acc)
            print(f"  Epoch {ep+1:3d}  loss={total_loss/len(train_data):.4f}  val_acc={acc:.3f}")

    return model, best_acc


def eval_standard_model(model, data, encode_fn):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(data), BATCH):
            batch = data[i:i+BATCH]
            seqs = [encode_fn(ex) for ex in batch]
            targets = torch.tensor([ex["answer"] for ex in batch]).to(DEVICE)
            x, mask = pad_sequences(seqs)
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            preds = model(x, pad_mask=mask).argmax(dim=-1)
            correct += (preds == targets).sum().item()
    return correct / len(data)


def train_latent_model(model, train_data, val_data):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best_acc = 0

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(train_data))
        total_loss = 0
        for i in range(0, len(train_data), BATCH):
            idx = perm[i:i+BATCH]
            batch = [train_data[j] for j in idx]

            starts = torch.tensor([ex["start"] + N_OP_TOKENS for ex in batch]).to(DEVICE)
            max_ops = max(len(ex["ops"]) for ex in batch)
            op_seqs = torch.zeros(len(batch), max_ops, dtype=torch.long).to(DEVICE)
            lengths = torch.tensor([len(ex["ops"]) for ex in batch]).to(DEVICE)
            for j, ex in enumerate(batch):
                for k, op in enumerate(ex["ops"]):
                    op_seqs[j, k] = op + 2
            targets = torch.tensor([ex["answer"] for ex in batch]).to(DEVICE)

            opt.zero_grad()
            logits = model(starts, op_seqs, lengths)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()

        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            acc = eval_latent_model(model, val_data)
            best_acc = max(best_acc, acc)
            print(f"  Epoch {ep+1:3d}  loss={total_loss/len(train_data):.4f}  val_acc={acc:.3f}")

    return model, best_acc


def eval_latent_model(model, data):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(data), BATCH):
            batch = data[i:i+BATCH]
            starts = torch.tensor([ex["start"] + N_OP_TOKENS for ex in batch]).to(DEVICE)
            max_ops = max(len(ex["ops"]) for ex in batch)
            op_seqs = torch.zeros(len(batch), max_ops, dtype=torch.long).to(DEVICE)
            lengths = torch.tensor([len(ex["ops"]) for ex in batch]).to(DEVICE)
            for j, ex in enumerate(batch):
                for k, op in enumerate(ex["ops"]):
                    op_seqs[j, k] = op + 2
            targets = torch.tensor([ex["answer"] for ex in batch]).to(DEVICE)
            preds = model(starts, op_seqs, lengths).argmax(dim=-1)
            correct += (preds == targets).sum().item()
    return correct / len(data)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    rng = np.random.RandomState(SEED)
    vocab_size = N_VALUES + N_OP_TOKENS + 10  # values + ops + special tokens

    # test at different chain lengths
    test_lengths = [5, 10, 15, 20, 30]
    # train on mixed lengths
    train_lengths = [3, 5, 7, 10, 12, 15]

    train_data = generate_dataset(TRAIN_SIZE, train_lengths, rng)

    results = {}

    for name, model_fn, train_fn, eval_fn in [
        ("Standard", lambda: StandardTransformer(vocab_size), "standard", "standard"),
        ("Token-CoT", lambda: CoTTransformer(vocab_size), "cot", "cot"),
        ("Latent-Thought", lambda: LatentThoughtTransformer(vocab_size), "latent", "latent"),
    ]:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        model = model_fn()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.perf_counter()
        if train_fn == "latent":
            model, _ = train_latent_model(model, train_data, train_data[:VAL_SIZE])
        else:
            enc = encode_standard if train_fn == "standard" else encode_cot
            model, _ = train_standard_model(model, train_data, train_data[:VAL_SIZE], enc, name)
        train_time = time.perf_counter() - t0
        print(f"  Training time: {train_time:.1f}s")

        results[name] = {"params": n_params, "train_time": train_time, "lengths": {}}

        for tl in test_lengths:
            test_data = generate_dataset(VAL_SIZE, [tl], np.random.RandomState(SEED + tl))
            if train_fn == "latent":
                acc = eval_latent_model(model, test_data)
            else:
                enc = encode_standard if train_fn == "standard" else encode_cot
                acc = eval_standard_model(model, test_data, enc)
            results[name]["lengths"][tl] = acc
            print(f"  Length {tl:3d}: acc={acc:.3f}")

    # print summary
    print(f"\n{'='*60}")
    print(f"{'Method':20s}", end="")
    for tl in test_lengths:
        print(f"  L={tl:2d}", end="")
    print()
    print("-" * 60)
    for name in results:
        print(f"{name:20s}", end="")
        for tl in test_lengths:
            print(f"  {results[name]['lengths'][tl]:.3f}", end="")
        print()
    print("=" * 60)

    with open("results_latent_thought.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_latent_thought.json")


if __name__ == "__main__":
    main()
