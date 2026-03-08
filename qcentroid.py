import logging
import time

from mrna_qfold.preprocessing import preprocess
from mrna_qfold.qubo import build_qubo
from mrna_qfold.quantum_solver import SolverConfig, solve_qaoa_qcentroid, solve_exact
from mrna_qfold.postprocessing import postprocess
from mrna_qfold.classical_baseline import classical_benchmark

logger = logging.getLogger("qcentroid-user-log")


def run(input_data, solver_params, extra_arguments):
    start_time = time.time()

    sequence = input_data["sequence"]
    qaoa_reps = input_data.get("qaoa_reps", 2)
    max_iter = input_data.get("max_iter", 200)
    stacking_reward = input_data.get("stacking_reward", -2.0)
    crossing_penalty = input_data.get("crossing_penalty", 10.0)

    run_classical = extra_arguments.get("run_classical", True)
    run_exact_flag = extra_arguments.get("run_exact", True)
    top_k = extra_arguments.get("top_k", 10)

    logger.info(f"sequence: {sequence} ({len(sequence)} nt), p={qaoa_reps}")

    # preprocessing
    prep = preprocess(sequence)
    logger.info(f"{len(prep.quartets)} quartets, {len(prep.crossing_pairs)} crossings")

    # QUBO
    qp = build_qubo(prep, stacking_reward=stacking_reward, crossing_penalty=crossing_penalty)
    n_qubits = qp.get_num_binary_vars()
    logger.info(f"QUBO: {n_qubits} qubits")

    # QAOA via QCentroid runtime
    config = SolverConfig(qaoa_reps=qaoa_reps, max_iter=max_iter)
    quantum_result = solve_qaoa_qcentroid(qp, solver_params, config)
    logger.info(f"best objective: {quantum_result.best_objective:.4f}")

    # postprocess
    post = postprocess(quantum_result, prep, top_k=top_k)
    logger.info(f"structure: {post.best_candidate.dot_bracket}  "
                f"valid_frac={post.valid_fraction:.0%}")

    # classical baselines (optional)
    classical_results = {}
    if run_classical:
        classical_results = classical_benchmark(sequence)

    # exact solver for validation (optional, small instances only)
    exact_result = None
    if run_exact_flag and n_qubits <= 20:
        exact_result = solve_exact(qp)
        logger.info(f"exact: {exact_result.best_objective:.4f}")

    elapsed = time.time() - start_time
    logger.info(f"done in {elapsed:.1f}s")

    return {
        "sequence": sequence,
        "sequence_length": len(sequence),
        "num_quartets": len(prep.quartets),
        "num_qubits": n_qubits,
        "best_structure": post.best_candidate.dot_bracket,
        "best_qubo_energy": post.best_candidate.qubo_energy,
        "best_vienna_energy": post.best_candidate.vienna_energy,
        "valid_fraction": post.valid_fraction,
        "qaoa_convergence": quantum_result.convergence_history,
        "candidates": [
            {
                "structure": c.dot_bracket,
                "qubo_energy": c.qubo_energy,
                "vienna_energy": c.vienna_energy,
                "is_valid": c.is_valid,
                "count": c.count,
            }
            for c in post.candidates
        ],
        "classical": classical_results,
        "exact_optimum": exact_result.best_objective if exact_result else None,
        "execution_time_seconds": elapsed,
    }
