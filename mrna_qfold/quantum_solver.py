import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from qiskit_optimization import QuadraticProgram

logger = logging.getLogger(__name__)


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
    all_samples: dict[str, int] = field(default_factory=dict)
    optimal_parameters: np.ndarray = field(default_factory=lambda: np.array([]))
    num_qubits: int = 0
    convergence_history: list[float] = field(default_factory=list)


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

    from qiskit_algorithms import QAOA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    if config.backend_type == BackendType.LOCAL_SIMULATOR:
        from qiskit.primitives import StatevectorSampler
        sampler = StatevectorSampler(seed=config.seed)
    else:
        backend = _get_ionq_backend(config)
        from qiskit.primitives import BackendSamplerV2
        sampler = BackendSamplerV2(backend=backend)

    optimizer = _get_optimizer(config)
    convergence = []

    def callback(eval_count, parameters, mean, metadata):
        convergence.append(mean)

    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=config.qaoa_reps,
        callback=callback,
    )

    result = MinimumEigenOptimizer(qaoa).solve(quadratic_program)

    best_bits = "".join(str(int(v.value)) for v in result.variables)
    samples = {}
    if hasattr(result, "samples") and result.samples:
        for s in result.samples:
            bits = "".join(str(int(v)) for v in s.x)
            samples[bits] = samples.get(bits, 0) + 1

    return QuantumResult(
        best_bitstring=best_bits,
        best_objective=result.fval,
        all_samples=samples,
        num_qubits=quadratic_program.get_num_binary_vars(),
        convergence_history=convergence,
    )


def solve_qaoa_qcentroid(quadratic_program, solver_params, config=None):
    if config is None:
        config = SolverConfig()

    from qcentroid_runtime import QCentroidRuntimeQiskit
    from qiskit_algorithms import QAOA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    runtime = QCentroidRuntimeQiskit(solver_params)
    optimizer = _get_optimizer(config)
    convergence = []

    def callback(eval_count, parameters, mean, metadata):
        convergence.append(mean)

    qaoa = QAOA(
        sampler=runtime,
        optimizer=optimizer,
        reps=config.qaoa_reps,
        callback=callback,
    )

    result = MinimumEigenOptimizer(qaoa).solve(quadratic_program)
    best_bits = "".join(str(int(v.value)) for v in result.variables)

    samples = {}
    if hasattr(result, "samples") and result.samples:
        for s in result.samples:
            bits = "".join(str(int(v)) for v in s.x)
            samples[bits] = samples.get(bits, 0) + 1

    return QuantumResult(
        best_bitstring=best_bits,
        best_objective=result.fval,
        all_samples=samples,
        num_qubits=quadratic_program.get_num_binary_vars(),
        convergence_history=convergence,
    )


def solve_exact(quadratic_program):
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(quadratic_program)
    best_bits = "".join(str(int(v.value)) for v in result.variables)

    return QuantumResult(
        best_bitstring=best_bits,
        best_objective=result.fval,
        all_samples={best_bits: 1},
        num_qubits=quadratic_program.get_num_binary_vars(),
        convergence_history=[result.fval],
    )
