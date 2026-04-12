"""
Uncertainty-Guided Search for Multi-Step Reasoning
====================================================

A model predicts continuous transformations step by step.
Because the task is continuous, the model is naturally imperfect
and produces calibrated uncertainty.

Three inference strategies:
  1. Greedy: always take the top prediction
  2. Beam search: always keep top-k sampled candidates  
  3. Uncertainty-guided: sample more candidates when uncertain

Run:  python experiment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from plotting import plot_accuracy_vs_chain_length, plot_accuracy_vs_compute, plot_entropy_heatmap

SEED = 42
HIDDEN = 24
N_OPS = 6
TRAIN_SIZE = 3000
EPOCHS = 15
BATCH = 256
LR = 1e-3
TOLERANCE = 0.3

torch.manual_seed(SEED)
np.random.seed(SEED)

OP_NAMES = ["sin*2", "x^2-1", "tanh*1.5", "saw", "clip+0.3", "inv"]


def apply_op(x, op_id):
    if op_id == 0:
        return np.sin(x) * 2
    elif op_id == 1:
        return np.clip(x ** 2 - 1, -3, 3)
    elif op_id == 2:
        return np.tanh(x) * 1.5
    elif op_id == 3:
        return ((x + np.pi) % (2 * np.pi)) - np.pi  # sawtooth-like wrapping
    elif op_id == 4:
        return np.clip(x, -1.5, 1.5) + 0.3
    elif op_id == 5:
        return 1.0 / (1.0 + np.abs(x)) * np.sign(x) * 2  # soft inverse
    return x


class StepPredictor(nn.Module):
    """Predicts next value and uncertainty from (current_value, operation)."""

    def __init__(self):
        super().__init__()
        self.op_embed = nn.Embedding(N_OPS, HIDDEN)
        self.net = nn.Sequential(
            nn.Linear(1 + HIDDEN, HIDDEN * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(HIDDEN, 1)
        self.logvar_head = nn.Linear(HIDDEN, 1)

    def forward(self, val, op):
        """Returns predicted mean and log-variance."""
        op_emb = self.op_embed(op)
        h = torch.cat([val.unsqueeze(-1) if val.dim() == 1 else val, op_emb], dim=-1)
        h = self.net(h)
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        return mean, logvar

    def predict(self, val, op, n_samples=1):
        """Sample n predictions. Returns (samples, uncertainty)."""
        with torch.no_grad():
            mean, logvar = self.forward(val, op)
            std = torch.exp(0.5 * logvar)
            if n_samples == 1:
                return mean, std
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(n_samples, *mean.shape)
            return samples, std


def generate_training_data():
    rng = np.random.RandomState(SEED)
    vals = rng.uniform(-2, 2, TRAIN_SIZE).astype(np.float32)
    ops = rng.randint(0, N_OPS, TRAIN_SIZE)
    targets = np.array([apply_op(v, o) for v, o in zip(vals, ops)], dtype=np.float32)
    return (
        torch.tensor(vals),
        torch.tensor(ops, dtype=torch.long),
        torch.tensor(targets),
    )


def train_model():
    print("Training step predictor...")
    model = StepPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    vals, ops, targets = generate_training_data()

    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(TRAIN_SIZE)
        for i in range(0, TRAIN_SIZE, BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            mean, logvar = model(vals[idx], ops[idx])
            # negative log-likelihood of Gaussian
            nll = 0.5 * (logvar + (targets[idx] - mean).pow(2) / logvar.exp())
            nll.mean().backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        test_vals = torch.linspace(-2, 2, 200)
        total_err = 0
        count = 0
        for o in range(N_OPS):
            op_t = torch.full((200,), o, dtype=torch.long)
            mean, _ = model(test_vals, op_t)
            true = torch.tensor([apply_op(v.item(), o) for v in test_vals])
            total_err += (mean - true).abs().mean().item()
            count += 1
        avg_err = total_err / count
    print(f"  Average absolute error: {avg_err:.4f}")
    return model, avg_err


def generate_chains(n_chains, chain_length, rng=None):
    if rng is None:
        rng = np.random.RandomState(SEED + 999)
    chains = []
    for _ in range(n_chains):
        start = rng.uniform(-1.5, 1.5)
        ops = rng.randint(0, N_OPS, chain_length).tolist()
        val = start
        for o in ops:
            val = apply_op(val, o)
        chains.append((float(start), ops, float(val)))
    return chains


# ── Inference strategies ─────────────────────────────────────────────────────

def greedy_decode(model, start, ops):
    val = torch.tensor([start])
    n_fwd = 0
    for o in ops:
        op_t = torch.tensor([o])
        mean, _ = model.predict(val, op_t)
        val = mean
        n_fwd += 1
    return val.item(), n_fwd


def beam_decode(model, start, ops, beam_width=4, n_samples_per_beam=None):
    """Sample n candidates per beam entry, keep top beam_width by score."""
    if n_samples_per_beam is None:
        n_samples_per_beam = beam_width
    beams = [(start, 0.0)]  # (value, neg_total_variance)
    n_fwd = 0

    for o in ops:
        op_t = torch.tensor([o])
        candidates = []
        for val, score in beams:
            val_t = torch.tensor([val])
            mean, std = model.predict(val_t, op_t)
            n_fwd += 1
            # sample candidates from the predicted distribution
            samples = mean + std * torch.randn(n_samples_per_beam)
            for s in samples:
                # score: penalize high-variance paths
                candidates.append((s.item(), score - std.item()))
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    best = max(beams, key=lambda x: x[1])
    return best[0], n_fwd


def uncertainty_guided_decode(model, start, ops, high_k=4, std_threshold=0.3):
    """Adaptive: sample more when model is uncertain (high std)."""
    beams = [(start, 0.0)]
    n_fwd = 0

    for o in ops:
        op_t = torch.tensor([o])
        candidates = []

        for val, score in beams:
            val_t = torch.tensor([val])
            mean, std = model.predict(val_t, op_t)
            n_fwd += 1

            if std.item() > std_threshold:
                n_samp = high_k
            else:
                n_samp = 1

            if n_samp == 1:
                candidates.append((mean.item(), score - std.item()))
            else:
                samples = mean + std * torch.randn(n_samp)
                for s in samples:
                    candidates.append((s.item(), score - std.item()))

        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:high_k]

    best = max(beams, key=lambda x: x[1])
    return best[0], n_fwd


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_strategy(model, chains, strategy_fn, **kwargs):
    correct = 0
    total_fwd = 0
    for start, ops, target in chains:
        pred, n_fwd = strategy_fn(model, start, ops, **kwargs)
        if abs(pred - target) < TOLERANCE:
            correct += 1
        total_fwd += n_fwd
    acc = correct / len(chains)
    avg_fwd = total_fwd / len(chains)
    return acc, avg_fwd


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    model, avg_err = train_model()

    chain_lengths = [3, 5, 7, 10, 13, 15]
    n_test = 500

    # calibrate std threshold from model's actual predictions
    print("\nCalibrating std threshold...")
    all_stds = []
    test_vals = torch.linspace(-2, 2, 200)
    for o in range(N_OPS):
        op_t = torch.full((200,), o, dtype=torch.long)
        _, std = model.predict(test_vals, op_t)
        all_stds.extend(std.numpy().tolist())
    all_stds = np.array(all_stds)
    threshold = np.percentile(all_stds, 60)
    print(f"  Std stats: mean={all_stds.mean():.3f}, p60={threshold:.3f}, max={all_stds.max():.3f}")

    strategies = {
        "Greedy": lambda m, s, o: greedy_decode(m, s, o),
        "Beam-2": lambda m, s, o: beam_decode(m, s, o, beam_width=2),
        "Beam-4": lambda m, s, o: beam_decode(m, s, o, beam_width=4),
        "Beam-8": lambda m, s, o: beam_decode(m, s, o, beam_width=8),
        "Uncertainty (k=4)": lambda m, s, o: uncertainty_guided_decode(
            m, s, o, high_k=4, std_threshold=threshold,
        ),
        "Uncertainty (k=8)": lambda m, s, o: uncertainty_guided_decode(
            m, s, o, high_k=8, std_threshold=threshold,
        ),
    }

    results = {name: {} for name in strategies}

    for cl in chain_lengths:
        print(f"\nChain length: {cl}")
        chains = generate_chains(n_test, cl)

        for name, fn in strategies.items():
            acc, avg_fwd = evaluate_strategy(model, chains, fn)
            results[name][cl] = (acc, avg_fwd)
            print(f"  {name:22s}  acc={acc:.3f}  avg_fwd={avg_fwd:.1f}")

    print("\nGenerating plots...")
    plot_accuracy_vs_chain_length(results, chain_lengths)
    plot_accuracy_vs_compute(results, chain_lengths)

    # entropy heatmap for one chain
    sample_chain = generate_chains(1, 10)[0]
    stds = []
    start, ops, _ = sample_chain
    val = torch.tensor([start])
    for o in ops:
        op_t = torch.tensor([o])
        mean, std = model.predict(val, op_t)
        stds.append(std.item())
        val = mean
    plot_entropy_heatmap(stds, [OP_NAMES[o] for o in ops], threshold)

    print("Done.")


if __name__ == "__main__":
    main()
