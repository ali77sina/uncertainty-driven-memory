import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COLORS = {
    "Greedy": "#a8a29e",
    "Beam-2": "#93c5fd",
    "Beam-4": "#2563eb",
    "Beam-8": "#1e3a5f",
    "Uncertainty (k=4)": "#dc2626",
    "Uncertainty (k=8)": "#7f1d1d",
}
MARKERS = {
    "Greedy": "o",
    "Beam-2": "s",
    "Beam-4": "s",
    "Beam-8": "s",
    "Uncertainty (k=4)": "D",
    "Uncertainty (k=8)": "D",
}
LINESTYLES = {
    "Greedy": "-",
    "Beam-2": "--",
    "Beam-4": "-",
    "Beam-8": "--",
    "Uncertainty (k=4)": "-",
    "Uncertainty (k=8)": "--",
}


def plot_accuracy_vs_chain_length(results, chain_lengths):
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, data in results.items():
        accs = [data[cl][0] for cl in chain_lengths]
        ax.plot(chain_lengths, accs, f"{MARKERS[name]}{LINESTYLES[name]}",
                color=COLORS[name], linewidth=1.5, markersize=6, label=name)

    ax.set_xlabel("Chain length (steps)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Chain Length")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", ncol=2)

    fig.savefig(FIGURES_DIR / "accuracy_vs_length.png")
    plt.close(fig)
    print("  Saved accuracy_vs_length.png")


def plot_accuracy_vs_compute(results, chain_lengths):
    """The money plot: accuracy vs forward passes at each chain length."""
    # pick a few representative chain lengths
    show_lengths = [5, 10, 15]
    available = [cl for cl in show_lengths if cl in chain_lengths]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5),
                              gridspec_kw={"wspace": 0.3})
    if len(available) == 1:
        axes = [axes]

    for ax, cl in zip(axes, available):
        for name, data in results.items():
            acc, fwd = data[cl]
            ax.scatter(fwd, acc, s=80, color=COLORS[name], marker=MARKERS[name],
                       label=name, zorder=5, edgecolors="white", linewidth=0.5)

        ax.set_xlabel("Avg forward passes")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Chain length = {cl}")
        ax.set_ylim(0, 1.05)
        if cl == available[0]:
            ax.legend(fontsize=8, loc="lower right")

    fig.savefig(FIGURES_DIR / "accuracy_vs_compute.png")
    plt.close(fig)
    print("  Saved accuracy_vs_compute.png")


def plot_entropy_heatmap(entropies, op_names, threshold):
    """Show entropy at each step of a single chain."""
    fig, ax = plt.subplots(figsize=(10, 2.5))

    n = len(entropies)
    colors_bar = ["#dc2626" if e > threshold else "#2563eb" for e in entropies]

    bars = ax.bar(range(n), entropies, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax.axhline(threshold, color="#a8a29e", linewidth=1.2, linestyle="--", label=f"Threshold ({threshold:.2f})")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"Step {i+1}\n{op}" for i, op in enumerate(op_names)], fontsize=8)
    ax.set_ylabel("Entropy")
    ax.set_title("Per-Step Uncertainty (red = branch, blue = commit)")
    ax.legend(fontsize=9)

    fig.savefig(FIGURES_DIR / "entropy_heatmap.png")
    plt.close(fig)
    print("  Saved entropy_heatmap.png")
