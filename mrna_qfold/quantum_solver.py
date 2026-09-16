import logging
import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from qiskit_optimization import QuadraticProgram
from scipy.sparse import SparseEfficiencyWarning

logger = logging.getLogger(__name__)


def _solve(optimizer, quadratic_program):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        return optimizer.solve(quadratic_program)


class BackendType(Enum):
    LOCAL_SIMULATOR = "local_simulator"
    IONQ_SIMULATOR = "ionq_simulator"
    IONQ_QPU = "ionq_qpu"


@dataclass
class SolverConfig:
    backend_type: BackendType = BackendType.LOCAL_SIMULATOR
    qaoa_reps: int = 2
    optimizer_name: str = "COBYLA"
    max_iter: int = 200
    shots: int = 4096
    seed: int = 42
    ionq_api_token: Optional[str] = None


@dataclass
class QuantumResult:
    best_bitstring: str
    best_objective: float
    samples: dict[str, float] = field(default_factory=dict)
    optimal_parameters: np.ndarray = field(default_factory=lambda: np.array([]))
    num_qubits: int = 0
    convergence_history: list[float] = field(default_factory=list)


def _empty_problem_result(quadratic_program):
    if quadratic_program.get_num_binary_vars() != 0:
        return None
    return QuantumResult(
        best_bitstring="",
        best_objective=0.0,
        samples={"": 1.0},
        num_qubits=0,
        convergence_history=[0.0],
    )


def _sample_probabilities(result, num_qubits):
    eigenstate = result.min_eigen_solver_result.eigenstate

    try:
        from qiskit.quantum_info import Statevector
        from qiskit.result import QuasiDistribution
    except ImportError:
        Statevector = ()
        QuasiDistribution = ()

    if isinstance(eigenstate, QuasiDistribution):
        raw_samples = eigenstate.binary_probabilities(num_bits=num_qubits)
    elif isinstance(eigenstate, Statevector):
        raw_samples = {
            format(i, f"0{num_qubits}b"): probability
            for i, probability in enumerate(eigenstate.probabilities())
            if probability > 0
        }
    elif isinstance(eigenstate, dict):
        values = np.asarray(list(eigenstate.values()))
        if np.all(np.isreal(values)) and np.all(values.real >= 0):
            raw_samples = {state: float(value.real) for state, value in eigenstate.items()}
        else:
            raw_samples = {state: float(abs(value) ** 2) for state, value in eigenstate.items()}
    else:
        values = np.asarray(eigenstate)
        raw_samples = {
            format(i, f"0{num_qubits}b"): float(abs(value) ** 2)
            for i, value in enumerate(values)
            if abs(value) > 0
        }

    samples = {}
    for state, probability in raw_samples.items():
        raw_bits = format(state, f"0{num_qubits}b") if isinstance(state, int) else str(state)
        bits = raw_bits.zfill(num_qubits)[::-1]
        samples[bits] = samples.get(bits, 0.0) + probability

    total = sum(samples.values())
    if total > 0:
        samples = {bits: probability / total for bits, probability in samples.items()}
    return samples


def _get_optimizer(config):
    if config.optimizer_name == "COBYLA":
        from qiskit_algorithms.optimizers import COBYLA
        return COBYLA(maxiter=config.max_iter)
    elif config.optimizer_name == "SPSA":
        from qiskit_algorithms.optimizers import SPSA
        return SPSA(maxiter=config.max_iter)
    elif config.optimizer_name == "NELDER_MEAD":
        from qiskit_algorithms.optimizers import NELDER_MEAD
        return NELDER_MEAD(maxiter=config.max_iter)
    raise ValueError(f"Unknown optimizer: {config.optimizer_name}")


def _get_ionq_backend(config):
    from qiskit_ionq import IonQProvider

    token = config.ionq_api_token or os.environ.get("IONQ_API_TOKEN")
    if not token:
        raise ValueError("Need IONQ_API_TOKEN env var or ionq_api_token in config")

    provider = IonQProvider(token)
    if config.backend_type == BackendType.IONQ_SIMULATOR:
        return provider.get_backend("ionq_simulator")
    else:
        return provider.get_backend("ionq_qpu")


def solve_qaoa(quadratic_program, config=None):
    if config is None:
        config = SolverConfig()

    empty_result = _empty_problem_result(quadratic_program)
    if empty_result is not None:
        return empty_result

    from qiskit_algorithms import QAOA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    algorithm_globals.random_seed = config.seed

    if config.backend_type == BackendType.LOCAL_SIMULATOR:
        from qiskit.primitives import StatevectorSampler
        sampler = StatevectorSampler(default_shots=config.shots, seed=config.seed)
    else:
        backend = _get_ionq_backend(config)
        from qiskit.primitives import BackendSamplerV2
        sampler = BackendSamplerV2(
            backend=backend,
            options={"default_shots": config.shots},
        )

    optimizer = _get_optimizer(config)
    convergence = []
    _, offset = quadratic_program.to_ising()

    def callback(eval_count, parameters, mean, metadata):
        convergence.append(float(np.real(mean)) + offset)

    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=config.qaoa_reps,
        callback=callback,
    )

    result = _solve(MinimumEigenOptimizer(qaoa), quadratic_program)

    best_bits = "".join(str(int(v)) for v in result.x)
    samples = _sample_probabilities(
        result,
        quadratic_program.get_num_binary_vars(),
    )

    raw_result = result.min_eigen_solver_result
    optimal_parameters = getattr(raw_result, "optimal_point", np.array([]))
    if optimal_parameters is None:
        optimal_parameters = np.array([])

    return QuantumResult(
        best_bitstring=best_bits,
        best_objective=result.fval,
        samples=samples,
        optimal_parameters=np.asarray(optimal_parameters),
        num_qubits=quadratic_program.get_num_binary_vars(),
        convergence_history=convergence,
    )


def solve_qaoa_qcentroid(quadratic_program, solver_params, config=None):
    if config is None:
        config = SolverConfig()

    empty_result = _empty_problem_result(quadratic_program)
    if empty_result is not None:
        return empty_result

    from qcentroid_runtime import QCentroidRuntimeQiskit
    from qiskit_algorithms import QAOA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    runtime = QCentroidRuntimeQiskit(solver_params)
    optimizer = _get_optimizer(config)
    convergence = []
    _, offset = quadratic_program.to_ising()

    def callback(eval_count, parameters, mean, metadata):
        convergence.append(float(np.real(mean)) + offset)

    qaoa = QAOA(
        sampler=runtime,
        optimizer=optimizer,
        reps=config.qaoa_reps,
        callback=callback,
    )

    result = _solve(MinimumEigenOptimizer(qaoa), quadratic_program)
    best_bits = "".join(str(int(v)) for v in result.x)

    samples = _sample_probabilities(
        result,
        quadratic_program.get_num_binary_vars(),
    )

    raw_result = result.min_eigen_solver_result
    optimal_parameters = getattr(raw_result, "optimal_point", np.array([]))
    if optimal_parameters is None:
        optimal_parameters = np.array([])

    return QuantumResult(
        best_bitstring=best_bits,
        best_objective=result.fval,
        samples=samples,
        optimal_parameters=np.asarray(optimal_parameters),
        num_qubits=quadratic_program.get_num_binary_vars(),
        convergence_history=convergence,
    )


def solve_exact(quadratic_program):
    empty_result = _empty_problem_result(quadratic_program)
    if empty_result is not None:
        return empty_result

    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    result = _solve(
        MinimumEigenOptimizer(NumPyMinimumEigensolver()),
        quadratic_program,
    )
    best_bits = "".join(str(int(v)) for v in result.x)

    return QuantumResult(
        best_bitstring=best_bits,
        best_objective=result.fval,
        samples={best_bits: 1.0},
        num_qubits=quadratic_program.get_num_binary_vars(),
        convergence_history=[result.fval],
    )
