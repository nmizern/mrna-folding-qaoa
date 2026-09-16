import argparse
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from itertools import product

from mrna_qfold.classical_baseline import nussinov_predict, vienna_mfe_predict
from mrna_qfold.postprocessing import (
    decode_bitstring,
    evaluate_qubo_energy,
    postprocess,
    validate_structure,
)
from mrna_qfold.preprocessing import preprocess
from mrna_qfold.quantum_solver import BackendType, SolverConfig, solve_exact, solve_qaoa
from mrna_qfold.qubo import build_qubo
from mrna_qfold.visualization import (
    plot_benchmark_summary,
    plot_combination_matrix,
    plot_energy_landscape,
    plot_qaoa_convergence,
    plot_structure_comparison,
)


BENCHMARK_SUITE = [
    {"seq": "GCAAAGC", "desc": "7nt hairpin"},
    {"seq": "GUCAAAGAU", "desc": "9nt stem-loop"},
    {"seq": "GGCCAAAUGGCC", "desc": "12nt GC stem"},
    {"seq": "CUUUAUGAG", "desc": "9nt competing quartets"},
    {"seq": "GCGCAAAGCGC", "desc": "11nt GC-rich stem"},
    {"seq": "AGAAUCUUU", "desc": "9nt competing quartets"},
    {"seq": "UUUGCGAAGG", "desc": "10nt wobble pairs"},
]


def _mean(values):
    return float(statistics.mean(values))


def _std(values):
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _base_pairs(structure):
    stack = []
    pairs = set()
    for i, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(i)
        elif symbol == ")" and stack:
            pairs.add((stack.pop(), i))
    return pairs


def _pair_f1(structure_a, structure_b):
    pairs_a = _base_pairs(structure_a)
    pairs_b = _base_pairs(structure_b)
    if not pairs_a and not pairs_b:
        return 1.0
    if not pairs_a or not pairs_b:
        return 0.0
    overlap = len(pairs_a & pairs_b)
    precision = overlap / len(pairs_a)
    recall = overlap / len(pairs_b)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _enumerate_states(prep, stacking_reward, crossing_penalty):
    states = []
    for values in product([0, 1], repeat=len(prep.quartets)):
        bitstring = "".join(str(value) for value in values)
        active, structure = decode_bitstring(bitstring, prep.quartets, len(prep.sequence))
        states.append({
            "bitstring": bitstring,
            "energy": evaluate_qubo_energy(
                bitstring,
                prep,
                stacking_reward,
                crossing_penalty,
            ),
            "is_valid": validate_structure(active, len(prep.sequence)),
            "structure": structure,
        })
    return states


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def benchmark_config(qaoa_reps, max_iter, shots, runs, seed,
                     stacking_reward, crossing_penalty):
    return {
        "qaoa_reps": qaoa_reps,
        "max_iter": max_iter,
        "shots": shots,
        "runs": runs,
        "seeds": list(range(seed, seed + runs)),
        "stacking_reward": stacking_reward,
        "crossing_penalty": crossing_penalty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "qiskit": _version("qiskit"),
            "qiskit-algorithms": _version("qiskit-algorithms"),
            "qiskit-optimization": _version("qiskit-optimization"),
        },
    }


def run_benchmark(sequences=None, qaoa_reps=2, max_iter=100, shots=4096,
                  runs=5, seed=42, stacking_reward=-2.0,
                  crossing_penalty=10.0, save_plots=False,
                  figures_dir="figures"):
    if sequences is None:
        sequences = BENCHMARK_SUITE

    print("mRNA folding benchmark")
    print(f"QAOA p={qaoa_reps}, {shots} shots, {runs} seeds")

    records = []
    convergence_examples = {}
    candidate_examples = {}

    for item in sequences:
        sequence = item["seq"] if isinstance(item, dict) else item
        description = item.get("desc", "") if isinstance(item, dict) else ""
        prep = preprocess(sequence)
        if not prep.quartets:
            print(f"{sequence}: no quartets, skipped")
            continue

        qp = build_qubo(prep, stacking_reward, crossing_penalty)
        num_qubits = qp.get_num_binary_vars()

        exact_times = []
        exact = None
        for _ in range(3):
            started = time.perf_counter()
            exact = solve_exact(qp)
            exact_times.append(time.perf_counter() - started)

        _, exact_structure = decode_bitstring(
            exact.best_bitstring,
            prep.quartets,
            len(sequence),
        )

        states = _enumerate_states(prep, stacking_reward, crossing_penalty)
        ground_states = [
            state for state in states
            if abs(state["energy"] - exact.best_objective) < 1e-9
        ]
        random_ground_probability = len(ground_states) / len(states)
        random_valid_probability = sum(state["is_valid"] for state in states) / len(states)
        random_expected_objective = _mean([state["energy"] for state in states])

        nussinov_times = []
        nussinov = None
        for _ in range(20):
            nussinov = nussinov_predict(sequence)
            nussinov_times.append(nussinov["elapsed_seconds"])
        vienna = vienna_mfe_predict(sequence)

        run_records = []
        for run_index in range(runs):
            run_seed = seed + run_index
            config = SolverConfig(
                backend_type=BackendType.LOCAL_SIMULATOR,
                qaoa_reps=qaoa_reps,
                max_iter=max_iter,
                shots=shots,
                seed=run_seed,
            )

            started = time.perf_counter()
            qaoa = solve_qaoa(qp, config)
            qaoa_time = time.perf_counter() - started
            post = postprocess(
                qaoa,
                prep,
                top_k=None,
                stacking_reward=stacking_reward,
                crossing_penalty=crossing_penalty,
            )

            energies = {
                bitstring: evaluate_qubo_energy(
                    bitstring,
                    prep,
                    stacking_reward,
                    crossing_penalty,
                )
                for bitstring in qaoa.samples
            }
            expected_objective = sum(
                qaoa.samples[bitstring] * energy
                for bitstring, energy in energies.items()
            )
            ground_probability = sum(
                probability
                for bitstring, probability in qaoa.samples.items()
                if abs(energies[bitstring] - exact.best_objective) < 1e-9
            )

            run_records.append({
                "seed": run_seed,
                "best_sample_objective": float(qaoa.best_objective),
                "best_structure": post.best_candidate.dot_bracket,
                "expected_objective": float(expected_objective),
                "ground_state_probability": float(ground_probability),
                "valid_probability": float(post.valid_fraction),
                "found_ground_state": ground_probability > 0,
                "nussinov_pair_f1": _pair_f1(
                    post.best_candidate.dot_bracket,
                    nussinov["structure"],
                ),
                "time_s": qaoa_time,
            })

            if run_index == 0:
                convergence_examples[sequence] = qaoa.convergence_history
                candidate_examples[sequence] = post.candidates

        expected_values = [run["expected_objective"] for run in run_records]
        ground_probabilities = [run["ground_state_probability"] for run in run_records]
        valid_probabilities = [run["valid_probability"] for run in run_records]
        qaoa_times = [run["time_s"] for run in run_records]

        record = {
            "sequence": sequence,
            "description": description,
            "length": len(sequence),
            "qubits": num_qubits,
            "conflicts": len(prep.crossing_pairs),
            "stacking": sum(len(items) for items in prep.stacking_sets.values()) // 2,
            "exact_objective": float(exact.best_objective),
            "exact_structure": exact_structure,
            "exact_ground_state_count": len(ground_states),
            "qaoa_expected_objective_mean": _mean(expected_values),
            "qaoa_expected_objective_std": _std(expected_values),
            "qaoa_expected_gap_mean": _mean(expected_values) - float(exact.best_objective),
            "qaoa_ground_probability_mean": _mean(ground_probabilities),
            "qaoa_ground_probability_std": _std(ground_probabilities),
            "qaoa_valid_probability_mean": _mean(valid_probabilities),
            "qaoa_valid_probability_std": _std(valid_probabilities),
            "qaoa_ground_state_success_rate": _mean([
                float(run["found_ground_state"]) for run in run_records
            ]),
            "random_expected_objective": random_expected_objective,
            "random_ground_probability": random_ground_probability,
            "random_valid_probability": random_valid_probability,
            "nussinov_structure": nussinov["structure"],
            "exact_nussinov_pair_f1": _pair_f1(exact_structure, nussinov["structure"]),
            "vienna_structure": vienna["structure"],
            "vienna_energy": vienna["energy"],
            "time_exact_median_s": float(statistics.median(exact_times)),
            "time_qaoa_median_s": float(statistics.median(qaoa_times)),
            "time_nussinov_median_s": float(statistics.median(nussinov_times)),
            "runs": run_records,
        }
        records.append(record)

        print(
            f"{sequence:12} {num_qubits:2}q  "
            f"E[QAOA]={record['qaoa_expected_objective_mean']:6.2f}  "
            f"exact={record['exact_objective']:6.2f}  "
            f"P(opt)={record['qaoa_ground_probability_mean']:6.1%}  "
            f"uniform={record['random_ground_probability']:6.1%}  "
            f"valid={record['qaoa_valid_probability_mean']:6.1%}"
        )

    print("\n| Sequence | Qubits | Exact | Mean E[QAOA] | P(opt) | Uniform P(opt) | P(valid) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for record in records:
        print(
            f"| `{record['sequence']}` | {record['qubits']} | "
            f"{record['exact_objective']:.2f} | "
            f"{record['qaoa_expected_objective_mean']:.2f} | "
            f"{record['qaoa_ground_probability_mean']:.1%} | "
            f"{record['random_ground_probability']:.1%} | "
            f"{record['qaoa_valid_probability_mean']:.1%} |"
        )

    if save_plots and records:
        import os
        import matplotlib.pyplot as plt

        os.makedirs(figures_dir, exist_ok=True)
        focus_sequence = "GCGCAAAGCGC"
        focus_record = next(
            (record for record in records if record["sequence"] == focus_sequence),
            records[0],
        )
        focus_sequence = focus_record["sequence"]
        focus_prep = preprocess(focus_sequence)

        figures = {
            "combination_matrix.png": plot_combination_matrix(
                focus_prep.combination_matrix,
                focus_sequence,
                focus_prep.quartets,
                title=f"Combination matrix and quartets ({focus_sequence})",
            ),
            "qaoa_convergence.png": plot_qaoa_convergence(
                convergence_examples[focus_sequence],
                exact_energy=focus_record["exact_objective"],
                title=f"QAOA objective (p={qaoa_reps}) - {focus_sequence}",
            ),
            "energy_landscape.png": plot_energy_landscape(
                candidate_examples[focus_sequence],
                best_energy=focus_record["exact_objective"],
                title=f"Sampled QUBO energies ({focus_sequence})",
            ),
            "structure_comparison.png": plot_structure_comparison(
                focus_sequence,
                focus_record["exact_structure"],
                focus_record["nussinov_structure"],
                focus_record["exact_objective"],
                None,
                quantum_label="QUBO ground state",
                classical_label="Nussinov",
                title=f"Secondary structures ({focus_sequence})",
            ),
            "benchmark_summary.png": plot_benchmark_summary(records),
        }

        for filename, figure in figures.items():
            figure.savefig(f"{figures_dir}/{filename}", dpi=200)
            plt.close(figure)
        print(f"\nsaved {len(figures)} figures to {figures_dir}/")

    return records


def main():
    parser = argparse.ArgumentParser(description="local mRNA QAOA benchmark")
    parser.add_argument("--reps", "-p", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", "-o", default="benchmark_results.json")
    args = parser.parse_args()

    records = run_benchmark(
        qaoa_reps=args.reps,
        max_iter=args.max_iter,
        shots=args.shots,
        runs=args.runs,
        seed=args.seed,
        save_plots=args.plot,
    )
    payload = {
        "config": benchmark_config(
            args.reps,
            args.max_iter,
            args.shots,
            args.runs,
            args.seed,
            -2.0,
            10.0,
        ),
        "results": records,
    }
    with open(args.output, "w") as output_file:
        json.dump(payload, output_file, indent=2)
    print(f"saved results to {args.output}")


if __name__ == "__main__":
    main()
