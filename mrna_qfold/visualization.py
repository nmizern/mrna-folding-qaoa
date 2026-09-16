import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

    ax.set_xlabel("Objective evaluation")
    ax.set_ylabel("Expected QUBO objective")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_energy_landscape(candidates, best_energy=None, classical_energy=None,
                          title="Sampled Energy Distribution", figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)

    if candidates and hasattr(candidates[0], "qubo_energy"):
        probabilities = {}
        for candidate in candidates:
            energy = round(candidate.qubo_energy, 10)
            if energy not in probabilities:
                probabilities[energy] = [0.0, 0.0]
            index = 0 if candidate.is_valid else 1
            probabilities[energy][index] += candidate.probability

        energies = sorted(probabilities)
        valid = [probabilities[energy][0] for energy in energies]
        invalid = [probabilities[energy][1] for energy in energies]
        if len(energies) > 1:
            width = min(b - a for a, b in zip(energies, energies[1:])) * 0.7
        else:
            width = 0.5

        ax.bar(energies, valid, width=width, color="steelblue",
               edgecolor="black", label="Valid structures")
        ax.bar(energies, invalid, width=width, bottom=valid, color="salmon",
               edgecolor="black", label="Invalid structures")
        ax.set_ylabel("Probability")
    elif isinstance(candidates, dict):
        energies = list(candidates.values())
        ax.hist(energies, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_ylabel("Count")
    else:
        ax.hist(candidates, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_ylabel("Count")

    if best_energy is not None:
        ax.axvline(x=best_energy, color="red", linewidth=2, label=f"Best ({best_energy:.2f})")

    if classical_energy is not None:
        ax.axvline(x=classical_energy, color="green", linestyle="--", linewidth=2,
                   label=f"Classical ({classical_energy:.2f})")

    ax.set_xlabel("QUBO objective")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_structure_comparison(sequence, quantum_structure, classical_structure,
                              quantum_energy=None, classical_energy=None,
                              quantum_label="QAOA", classical_label="Nussinov",
                              title="Quantum vs Classical", figsize=(12, 6)):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    def draw_arcs(ax, seq, structure, energy, label, color):
        n = len(seq)
        for i, base in enumerate(seq):
            ax.text(i, 0, base, ha="center", va="center", fontsize=8, fontweight="bold")

        stack = []
        for i, ch in enumerate(structure):
            if ch == "(":
                stack.append(i)
            elif ch == ")" and stack:
                j = stack.pop()
                left, right = j, i
                mid = (left + right) / 2
                span = right - left
                arc = patches.Arc((mid, 0), span, span, angle=0,
                                  theta1=0, theta2=180, color=color, linewidth=1.5)
                ax.add_patch(arc)

        ax.set_xlim(-1, n)
        ax.set_ylim(-0.5, n / 2 + 1)
        ax.set_aspect("equal")
        e_str = f"  (E = {energy:.2f})" if energy is not None else ""
        ax.set_title(f"{label}: {structure}{e_str}", fontfamily="monospace", fontsize=10)
        ax.axis("off")

    draw_arcs(ax1, sequence, quantum_structure, quantum_energy, quantum_label, "tab:blue")
    draw_arcs(ax2, sequence, classical_structure, classical_energy, classical_label, "tab:green")
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_benchmark_summary(records, title="Local benchmark", figsize=(12, 8)):
    records = sorted(records, key=lambda record: record["qubits"])
    qubits = [record["qubits"] for record in records]
    lengths = [record["length"] for record in records]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].scatter(lengths, qubits, color="tab:blue", s=60)
    axes[0, 0].set_xlabel("Sequence length (nt)")
    axes[0, 0].set_ylabel("Qubits (quartets)")
    axes[0, 0].set_title("Problem size", fontweight="bold")

    axes[0, 1].semilogy(
        qubits,
        [record["time_qaoa_median_s"] for record in records],
        "bo-",
        label="QAOA simulator",
    )
    axes[0, 1].semilogy(
        qubits,
        [record["time_nussinov_median_s"] for record in records],
        "gs-",
        label="Nussinov",
    )
    axes[0, 1].set_xlabel("Qubits")
    axes[0, 1].set_ylabel("Median time (s)")
    axes[0, 1].set_title("Runtime", fontweight="bold")
    axes[0, 1].legend()

    axes[1, 0].errorbar(
        qubits,
        [100 * record["qaoa_ground_probability_mean"] for record in records],
        yerr=[100 * record["qaoa_ground_probability_std"] for record in records],
        fmt="bo-",
        capsize=3,
        label="QAOA",
    )
    axes[1, 0].plot(
        qubits,
        [100 * record["random_ground_probability"] for record in records],
        "k--",
        label="Uniform sampling",
    )
    axes[1, 0].set_xlabel("Qubits")
    axes[1, 0].set_ylabel("Ground-state probability (%)")
    axes[1, 0].set_title("Sampling the optimum", fontweight="bold")
    axes[1, 0].legend()

    axes[1, 1].errorbar(
        qubits,
        [100 * record["qaoa_valid_probability_mean"] for record in records],
        yerr=[100 * record["qaoa_valid_probability_std"] for record in records],
        fmt="ro-",
        capsize=3,
        label="QAOA",
    )
    axes[1, 1].plot(
        qubits,
        [100 * record["random_valid_probability"] for record in records],
        "k--",
        label="Uniform sampling",
    )
    axes[1, 1].set_xlabel("Qubits")
    axes[1, 1].set_ylabel("Valid probability (%)")
    axes[1, 1].set_title("Valid structures", fontweight="bold")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig
