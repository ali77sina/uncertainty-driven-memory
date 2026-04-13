"""
Latent Thought Vectors v2 — Scaled Up
=======================================

Improvements over v1:
  1. Auxiliary loss: thought vectors must predict intermediate values
  2. Multi-pass thinking: thoughts go through encoder multiple times per step
  3. Bigger model (256 dim, 4 layers), more training (150 epochs)
  4. Train on longer sequences (up to 25 steps)

Run on GPU:  python3 latent_thought_v2.py
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

D_MODEL = 256
NHEAD = 8
N_LAYERS = 4
THOUGHT_DIM = 128
N_THOUGHTS = 4
THINK_PASSES = 1  # start with 1, can increase later

TRAIN_SIZE = 10000
VAL_SIZE = 500
BATCH = 128
EPOCHS = 80
LR = 3e-4
AUX_WEIGHT = 0.5  # weight for auxiliary intermediate prediction loss

N_VALUES = 200

torch.manual_seed(SEED)
np.random.seed(SEED)

OPS = [
    ("add_small", lambda x: (x + 3) % N_VALUES),
    ("add_big", lambda x: (x + 17) % N_VALUES),
    ("sub_small", lambda x: (x - 5) % N_VALUES),
    ("sub_big", lambda x: (x - 13) % N_VALUES),
    ("double", lambda x: (x * 2) % N_VALUES),
    ("half", lambda x: x // 2),
    ("cond_gt50", lambda x: (x + 10) % N_VALUES if x > 50 else (x - 10) % N_VALUES),
    ("cond_even", lambda x: (x * 2) % N_VALUES if x % 2 == 0 else (x + 1) % N_VALUES),
]
N_OPS = len(OPS)
N_OP_TOKENS = N_OPS + 2  # +pad +sep


def generate_chain(length, rng):
    start = rng.randint(0, N_VALUES)
    ops = []
    val = start
    intermediates = [val]
    for _ in range(length):
        op_idx = rng.randint(0, len(OPS))
        _, fn = OPS[op_idx]
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
            "start": start, "ops": ops,
            "intermediates": intermediates, "answer": answer, "length": length,
        })
    return data


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


def make_encoder(d_model, nhead, n_layers):
    layer = nn.TransformerEncoderLayer(
        d_model, nhead, d_model * 2, batch_first=True, dropout=0.1)
    return nn.TransformerEncoder(layer, n_layers)


class StandardTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
        self.pos = PositionalEncoding(D_MODEL)
        self.encoder = make_encoder(D_MODEL, NHEAD, N_LAYERS)
        self.head = nn.Linear(D_MODEL, N_VALUES)

    def forward(self, x, pad_mask=None):
        h = self.pos(self.embed(x))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.head(h.mean(dim=1))


class CoTTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
        self.pos = PositionalEncoding(D_MODEL, max_len=1024)
        self.encoder = make_encoder(D_MODEL, NHEAD, N_LAYERS)
        self.head = nn.Linear(D_MODEL, N_VALUES)

    def forward(self, x, pad_mask=None):
        h = self.pos(self.embed(x))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.head(h.mean(dim=1))


class LatentThoughtV2(nn.Module):
    """
    v2: auxiliary loss on thoughts + multi-pass thinking + bigger capacity.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
        self.pos = PositionalEncoding(D_MODEL)
        self.encoder = make_encoder(D_MODEL, NHEAD, N_LAYERS)

        self.thought_proj = nn.Linear(D_MODEL, N_THOUGHTS * THOUGHT_DIM)
        self.thought_inject = nn.Linear(THOUGHT_DIM, D_MODEL)

        # auxiliary head: predict intermediate value from thought vectors
        self.aux_head = nn.Sequential(
            nn.Linear(N_THOUGHTS * THOUGHT_DIM, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, N_VALUES),
        )

        # uncertainty gate
        self.register_buffer("running_mean", torch.zeros(N_THOUGHTS * THOUGHT_DIM))
        self.register_buffer("running_var", torch.ones(N_THOUGHTS * THOUGHT_DIM))
        self.register_buffer("n_updates", torch.tensor(0.0))
        self.gate = nn.Sequential(
            nn.Linear(N_THOUGHTS * THOUGHT_DIM + 1, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
            nn.Sigmoid(),
        )

        self.head = nn.Linear(D_MODEL, N_VALUES)

    def compute_uncertainty(self, thought_flat):
        diff = thought_flat - self.running_mean
        return (diff.pow(2) / (self.running_var + 1e-6)).mean(dim=-1, keepdim=True)

    def update_stats(self, thought_flat):
        if self.training:
            with torch.no_grad():
                bm = thought_flat.mean(dim=0)
                bv = thought_flat.var(dim=0)
                mom = min(0.1, 1.0 / (self.n_updates.item() + 1))
                self.running_mean.lerp_(bm, mom)
                self.running_var.lerp_(bv, mom)
                self.n_updates += 1

    def forward(self, starts, op_seqs, lengths):
        B = starts.size(0)
        max_len = op_seqs.size(1)

        # init thoughts from start
        s_emb = self.embed(starts).unsqueeze(1)
        h = self.encoder(self.pos(s_emb))
        thought_flat = self.thought_proj(h.squeeze(1))
        thoughts = thought_flat.view(B, N_THOUGHTS, THOUGHT_DIM)

        aux_logits_list = []

        for step in range(max_len):
            op_tokens = op_seqs[:, step]
            active = (step < lengths).float().unsqueeze(-1)

            # multi-pass thinking
            for pass_i in range(THINK_PASSES):
                thought_embs = self.thought_inject(thoughts)
                if pass_i == 0:
                    op_emb = self.embed(op_tokens).unsqueeze(1)
                    combined = torch.cat([thought_embs, op_emb], dim=1)
                else:
                    combined = thought_embs
                combined = self.pos(combined)
                out = self.encoder(combined)
                pooled = out[:, :N_THOUGHTS].mean(dim=1)
                new_flat = self.thought_proj(pooled)
                thoughts = new_flat.view(B, N_THOUGHTS, THOUGHT_DIM)

            # auxiliary prediction from current thought state
            aux_logits_list.append(self.aux_head(new_flat))

            # uncertainty-gated update
            uncertainty = self.compute_uncertainty(new_flat)
            self.update_stats(new_flat)
            gate_in = torch.cat([new_flat, uncertainty], dim=-1)
            write_gate = self.gate(gate_in).unsqueeze(-1)

            new_thoughts = thoughts
            updated = write_gate * new_thoughts + (1 - write_gate) * thoughts
            thoughts = thoughts * (1 - active.unsqueeze(-1)) + updated * active.unsqueeze(-1)

        # final prediction
        final_embs = self.thought_inject(thoughts)
        final_out = self.encoder(self.pos(final_embs))
        logits = self.head(final_out.mean(dim=1))

        aux_logits = torch.stack(aux_logits_list, dim=1)  # (B, max_len, N_VALUES)
        return logits, aux_logits


# ── Training ─────────────────────────────────────────────────────────────────

def pad_seqs(seqs, pad=0):
    ml = max(len(s) for s in seqs)
    p = torch.full((len(seqs), ml), pad, dtype=torch.long)
    m = torch.ones(len(seqs), ml, dtype=torch.bool)
    for i, s in enumerate(seqs):
        p[i, :len(s)] = torch.tensor(s)
        m[i, :len(s)] = False
    return p, m


def encode_standard(ex):
    t = [ex["start"] + N_OP_TOKENS]
    for op in ex["ops"]:
        t.append(op + 2)
    t.append(1)
    return t


def encode_cot(ex):
    t = [ex["start"] + N_OP_TOKENS]
    for i, op in enumerate(ex["ops"]):
        t.append(op + 2)
        t.append(ex["intermediates"][i + 1] + N_OP_TOKENS)
    t.append(1)
    return t


def train_seq_model(model, data, val_data, enc_fn, name):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(data))
        tloss = 0
        for i in range(0, len(data), BATCH):
            idx = perm[i:i+BATCH]
            batch = [data[j] for j in idx]
            seqs = [enc_fn(ex) for ex in batch]
            tgt = torch.tensor([ex["answer"] for ex in batch]).to(DEVICE)
            x, mask = pad_seqs(seqs)
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x, pad_mask=mask), tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item() * len(batch)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            acc = eval_seq_model(model, val_data, enc_fn)
            print(f"  {name} Ep {ep+1:4d}  loss={tloss/len(data):.4f}  val={acc:.3f}")
    return model


def eval_seq_model(model, data, enc_fn):
    model.eval()
    c = 0
    with torch.no_grad():
        for i in range(0, len(data), BATCH):
            b = data[i:i+BATCH]
            seqs = [enc_fn(ex) for ex in b]
            tgt = torch.tensor([ex["answer"] for ex in b]).to(DEVICE)
            x, mask = pad_seqs(seqs)
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            c += (model(x, pad_mask=mask).argmax(-1) == tgt).sum().item()
    return c / len(data)


def train_latent(model, data, val_data):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(data))
        tloss = 0
        for i in range(0, len(data), BATCH):
            idx = perm[i:i+BATCH]
            batch = [data[j] for j in idx]

            starts = torch.tensor([ex["start"] + N_OP_TOKENS for ex in batch]).to(DEVICE)
            max_ops = max(len(ex["ops"]) for ex in batch)
            op_seqs = torch.zeros(len(batch), max_ops, dtype=torch.long).to(DEVICE)
            lengths = torch.tensor([len(ex["ops"]) for ex in batch]).to(DEVICE)

            # intermediate targets for auxiliary loss
            inter_targets = torch.zeros(len(batch), max_ops, dtype=torch.long).to(DEVICE)
            for j, ex in enumerate(batch):
                for k, op in enumerate(ex["ops"]):
                    op_seqs[j, k] = op + 2
                for k in range(len(ex["ops"])):
                    inter_targets[j, k] = ex["intermediates"][k + 1]

            targets = torch.tensor([ex["answer"] for ex in batch]).to(DEVICE)

            opt.zero_grad()
            logits, aux_logits = model(starts, op_seqs, lengths)

            # main loss: predict final answer
            main_loss = F.cross_entropy(logits, targets)

            # auxiliary loss: predict intermediate values from thought vectors
            aux_loss = 0
            n_aux = 0
            for j in range(len(batch)):
                L = lengths[j].item()
                if L > 0:
                    aux_loss += F.cross_entropy(aux_logits[j, :L], inter_targets[j, :L])
                    n_aux += 1
            if n_aux > 0:
                aux_loss = aux_loss / n_aux

            loss = main_loss + AUX_WEIGHT * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item() * len(batch)
        sched.step()

        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            acc = eval_latent(model, val_data)
            print(f"  Latent Ep {ep+1:4d}  loss={tloss/len(data):.4f}  val={acc:.3f}")
    return model


def eval_latent(model, data):
    model.eval()
    c = 0
    with torch.no_grad():
        for i in range(0, len(data), BATCH):
            b = data[i:i+BATCH]
            starts = torch.tensor([ex["start"] + N_OP_TOKENS for ex in b]).to(DEVICE)
            max_ops = max(len(ex["ops"]) for ex in b)
            op_seqs = torch.zeros(len(b), max_ops, dtype=torch.long).to(DEVICE)
            lengths = torch.tensor([len(ex["ops"]) for ex in b]).to(DEVICE)
            for j, ex in enumerate(b):
                for k, op in enumerate(ex["ops"]):
                    op_seqs[j, k] = op + 2
            tgt = torch.tensor([ex["answer"] for ex in b]).to(DEVICE)
            logits, _ = model(starts, op_seqs, lengths)
            c += (logits.argmax(-1) == tgt).sum().item()
    return c / len(data)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    rng = np.random.RandomState(SEED)
    vocab_size = N_VALUES + N_OP_TOKENS + 10

    train_lengths = [3, 5, 7, 10, 15, 20, 25]
    test_lengths = [5, 10, 15, 20, 30, 40, 50]

    train_data = generate_dataset(TRAIN_SIZE, train_lengths, rng)
    results = {}

    for name, model_fn, is_latent, enc_fn in [
        ("Standard", lambda: StandardTransformer(vocab_size), False, encode_standard),
        ("Token-CoT", lambda: CoTTransformer(vocab_size), False, encode_cot),
        ("Latent-Thought-v2", lambda: LatentThoughtV2(vocab_size), True, None),
    ]:
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        model = model_fn()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        t0 = time.perf_counter()
        if is_latent:
            model = train_latent(model, train_data, train_data[:VAL_SIZE])
        else:
            model = train_seq_model(model, train_data, train_data[:VAL_SIZE], enc_fn, name)
        dt = time.perf_counter() - t0
        print(f"  Train time: {dt:.1f}s")

        results[name] = {"params": n_params, "train_time": dt, "lengths": {}}
        for tl in test_lengths:
            td = generate_dataset(VAL_SIZE, [tl], np.random.RandomState(SEED + tl))
            if is_latent:
                acc = eval_latent(model, td)
            else:
                acc = eval_seq_model(model, td, enc_fn)
            results[name]["lengths"][tl] = acc
            print(f"  L={tl:3d}: {acc:.3f}")

    print(f"\n{'='*70}")
    print(f"{'Method':22s}", end="")
    for tl in test_lengths:
        print(f" L={tl:2d}", end="")
    print()
    print("-" * 70)
    for name in results:
        print(f"{name:22s}", end="")
        for tl in test_lengths:
            print(f" {results[name]['lengths'][tl]:.3f}", end="")
        print()
    print("=" * 70)

    with open("results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results_v2.json")


if __name__ == "__main__":
    main()
