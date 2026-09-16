import argparse
import time

from mrna_qfold.preprocessing import preprocess
from mrna_qfold.qubo import build_qubo
from mrna_qfold.quantum_solver import (
    BackendType,
    SolverConfig,
    solve_qaoa,
    solve_exact,
)
from mrna_qfold.postprocessing import evaluate_qubo_energy, postprocess
from mrna_qfold.classical_baseline import classical_benchmark

DEFAULT_SEQUENCES = [
    "GCAAAGC",
    "GCGCAAAGCGC",
    "GGCCAAAUGGCC",
]


def run_local(sequence, qaoa_reps=2, max_iter=200,
              stacking_reward=-2.0, crossing_penalty=10.0, plot=False):
    print(f"\n--- {sequence} ({len(sequence)} nt) ---")

    # preprocessing
    t0 = time.perf_counter()
    prep = preprocess(sequence)
    print(f"quartets: {len(prep.quartets)}, crossings: {len(prep.crossing_pairs)}, "
          f"stacking: {sum(len(v) for v in prep.stacking_sets.values()) // 2}  "
          f"({time.perf_counter()-t0:.3f}s)")

    # QUBO
    qp = build_qubo(prep, stacking_reward, crossing_penalty)
    n_qubits = qp.get_num_binary_vars()
    print(f"QUBO: {n_qubits} qubits")

    # exact (small instances)
    exact_result = None
    if n_qubits <= 20:
        t0 = time.perf_counter()
        exact_result = solve_exact(qp)
        print(f"exact: {exact_result.best_objective:.4f} [{exact_result.best_bitstring}]  "
              f"({time.perf_counter()-t0:.3f}s)")

    # QAOA
    config = SolverConfig(
        backend_type=BackendType.LOCAL_SIMULATOR,
        qaoa_reps=qaoa_reps,
        max_iter=max_iter,
    )
    t0 = time.perf_counter()
    quantum_result = solve_qaoa(qp, config)
    qaoa_time = time.perf_counter() - t0
    expected_objective = sum(
        probability * evaluate_qubo_energy(
            bitstring,
            prep,
            stacking_reward,
            crossing_penalty,
        )
        for bitstring, probability in quantum_result.samples.items()
    )
    print(f"QAOA (p={qaoa_reps}): expected={expected_objective:.4f}, "
          f"best sample={quantum_result.best_objective:.4f} "
          f"[{quantum_result.best_bitstring}]  ({qaoa_time:.3f}s)")

    # postprocess
    post = postprocess(
        quantum_result,
        prep,
        stacking_reward=stacking_reward,
        crossing_penalty=crossing_penalty,
    )
    best = post.best_candidate
    print(f"structure: {best.dot_bracket}  valid={best.is_valid}  "
          f"valid_frac={post.valid_fraction:.0%}")
    if best.vienna_energy is not None:
        print(f"vienna energy: {best.vienna_energy:.2f} kcal/mol")

    # classical comparison
    baselines = classical_benchmark(sequence)
    for name, res in baselines.items():
        if res["structure"] is None:
            print(f"  {name}: unavailable")
            continue
        e = f"{res['energy']:.2f}" if res.get("energy") is not None else "n/a"
        print(f"  {name}: {res['structure']}  pairs={res['num_pairs']}  E={e}")

    if exact_result:
        ground_probability = sum(
            probability
            for bitstring, probability in quantum_result.samples.items()
            if abs(evaluate_qubo_energy(
                bitstring,
                prep,
                stacking_reward,
                crossing_penalty,
            ) - exact_result.best_objective) < 1e-9
        )
        gap = expected_objective - exact_result.best_objective
        print(f"expected gap: {gap:.4f}, P(opt)={ground_probability:.1%}")

    if plot:
        import os
        from mrna_qfold.visualization import plot_structure_comparison
        os.makedirs("figures", exist_ok=True)
        class_struct = baselines.get("nussinov", {}).get("structure", "." * len(sequence))
        fig = plot_structure_comparison(sequence, best.dot_bracket, class_struct, best.qubo_energy, None)
        out_path = f"figures/{sequence}_structure.png"
        fig.savefig(out_path, dpi=150)
        print(f"saved structure plot to {out_path}")

    return {
        "sequence": sequence,
        "quantum_structure": best.dot_bracket,
        "quantum_energy": best.qubo_energy,
        "quantum_expected_energy": expected_objective,
        "exact_energy": exact_result.best_objective if exact_result else None,
        "classical": {k: v["structure"] for k, v in baselines.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="mRNA folding via QAOA (local sim)")
    parser.add_argument("--sequence", "-s", type=str, default=None)
    parser.add_argument("--reps", "-p", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--plot", action="store_true", help="save structure comparison plot to figures/")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.sequence:
        run_local(args.sequence, qaoa_reps=args.reps, max_iter=args.max_iter, plot=args.plot)
    elif args.all:
        for seq in DEFAULT_SEQUENCES:
            run_local(seq, qaoa_reps=args.reps, max_iter=args.max_iter, plot=args.plot)
    else:
        run_local(DEFAULT_SEQUENCES[0], qaoa_reps=args.reps, max_iter=args.max_iter, plot=args.plot)


if __name__ == "__main__":
    main()
