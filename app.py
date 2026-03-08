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
from mrna_qfold.postprocessing import postprocess
from mrna_qfold.classical_baseline import classical_benchmark

DEFAULT_SEQUENCES = [
    "GCAAAGC",
    "GCGCAAAGCGC",
    "GGCCAAAUGGCC",
]


def run_local(sequence, qaoa_reps=2, max_iter=200,
              stacking_reward=-2.0, crossing_penalty=10.0):
    print(f"\n--- {sequence} ({len(sequence)} nt) ---")

    # preprocessing
    t0 = time.perf_counter()
    prep = preprocess(sequence)
    print(f"quartets: {len(prep.quartets)}, crossings: {len(prep.crossing_pairs)}, "
          f"stacking: {sum(len(v) for v in prep.stacking_sets.values()) // 2}  "
          f"({time.perf_counter()-t0:.3f}s)")

    if not prep.quartets:
        print("no quartets found, nothing to fold")
        return {"sequence": sequence, "result": "no_structure"}

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
    print(f"QAOA (p={qaoa_reps}): {quantum_result.best_objective:.4f} "
          f"[{quantum_result.best_bitstring}]  ({time.perf_counter()-t0:.3f}s)")

    # postprocess
    post = postprocess(quantum_result, prep)
    best = post.best_candidate
    print(f"structure: {best.dot_bracket}  valid={best.is_valid}  "
          f"valid_frac={post.valid_fraction:.0%}")
    if best.vienna_energy is not None:
        print(f"vienna energy: {best.vienna_energy:.2f} kcal/mol")

    # classical comparison
    baselines = classical_benchmark(sequence)
    for name, res in baselines.items():
        e = f"{res['energy']:.2f}" if res.get("energy") is not None else "n/a"
        print(f"  {name}: {res['structure']}  pairs={res['num_pairs']}  E={e}")

    if exact_result and exact_result.best_objective != 0:
        ratio = quantum_result.best_objective / exact_result.best_objective
        print(f"QAOA/exact ratio: {ratio:.3f}")

    return {
        "sequence": sequence,
        "quantum_structure": best.dot_bracket,
        "quantum_energy": best.qubo_energy,
        "exact_energy": exact_result.best_objective if exact_result else None,
        "classical": {k: v["structure"] for k, v in baselines.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="mRNA folding via QAOA (local sim)")
    parser.add_argument("--sequence", "-s", type=str, default=None)
    parser.add_argument("--reps", "-p", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.sequence:
        run_local(args.sequence, qaoa_reps=args.reps, max_iter=args.max_iter)
    elif args.all:
        for seq in DEFAULT_SEQUENCES:
            run_local(seq, qaoa_reps=args.reps, max_iter=args.max_iter)
    else:
        run_local(DEFAULT_SEQUENCES[0], qaoa_reps=args.reps, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
