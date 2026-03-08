import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_combination_matrix(matrix, sequence, quartets=None, title="Combination Matrix", figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(matrix, cmap="Blues", aspect="equal", origin="upper")

    if quartets:
        for q in quartets:
            rect1 = patches.Rectangle((q.l - 0.5, q.k - 0.5), 1, 1,
                                       linewidth=2, edgecolor="red", facecolor="red", alpha=0.3)
            rect2 = patches.Rectangle((q.l - 1 - 0.5, q.k + 1 - 0.5), 1, 1,
                                       linewidth=2, edgecolor="red", facecolor="red", alpha=0.3)
            ax.add_patch(rect1)
            ax.add_patch(rect2)

    n = len(sequence)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(list(sequence), fontsize=8)
    ax.set_yticklabels(list(sequence), fontsize=8)
    ax.set_xlabel("3' position (j)")
    ax.set_ylabel("5' position (i)")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_qaoa_convergence(convergence_history, exact_energy=None, title="QAOA Convergence", figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(convergence_history, "b-", linewidth=1.5, label="QAOA objective")

    if exact_energy is not None:
        ax.axhline(y=exact_energy, color="r", linestyle="--", linewidth=1,
                    label=f"Exact ({exact_energy:.3f})")

    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("Objective (kcal/mol)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_energy_landscape(samples, best_energy, classical_energy=None,
                          title="Sampled Energies", figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    energies = list(samples.values())
    ax.hist(energies, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(x=best_energy, color="red", linewidth=2, label=f"Best ({best_energy:.3f})")

    if classical_energy is not None:
        ax.axvline(x=classical_energy, color="green", linestyle="--", linewidth=2,
                   label=f"Classical ({classical_energy:.3f})")

    ax.set_xlabel("Energy (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_structure_comparison(sequence, quantum_structure, classical_structure,
                              quantum_energy, classical_energy,
                              title="Quantum vs Classical", figsize=(12, 6)):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    def draw_arcs(ax, seq, structure, energy, label, color):
        n = len(seq)
        for i, base in enumerate(seq):
            ax.text(i, 0, base, ha="center", va="center", fontsize=7, fontweight="bold")

        stack = []
        for i, ch in enumerate(structure):
            if ch == "(":
                stack.append(i)
            elif ch == ")" and stack:
                j = stack.pop()
                mid = (i + j) / 2
                arc = patches.Arc((mid, 0), j - i, j - i, angle=0,
                                  theta1=0, theta2=180, color=color, linewidth=1.5)
                ax.add_patch(arc)

        ax.set_xlim(-1, n)
        ax.set_ylim(-0.5, n / 2 + 1)
        ax.set_aspect("equal")
        ax.set_title(f"{label}: {structure}  (E = {energy:.2f})", fontfamily="monospace", fontsize=10)
        ax.axis("off")

    draw_arcs(ax1, sequence, quantum_structure, quantum_energy, "QAOA", "tab:blue")
    draw_arcs(ax2, sequence, classical_structure, classical_energy, "Classical", "tab:green")
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_scaling_analysis(sequence_lengths, num_qubits, quantum_times, classical_times,
                          title="Scaling", figsize=(12, 5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(sequence_lengths, num_qubits, "bo-", linewidth=2, markersize=6)
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Qubits (quartets)")
    ax1.set_title("Problem Size", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(sequence_lengths, quantum_times, "bo-", linewidth=2, markersize=6, label="QAOA (sim)")
    ax2.semilogy(sequence_lengths, classical_times, "gs-", linewidth=2, markersize=6, label="Nussinov")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("Time (s)")
    ax2.set_title("Computation Time", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
