# H(q) = sum_i e_i*q_i + r*sum_{QS} q_i*q_j + lambda*sum_{QC} q_i*q_j

import numpy as np
from qiskit_optimization import QuadraticProgram


def build_qubo_matrix(preprocess_result, stacking_reward=-2.0, crossing_penalty=10.0):
    n = len(preprocess_result.quartets)
    Q = np.zeros((n, n))

    for i, energy in preprocess_result.quartet_energies.items():
        Q[i][i] = energy

    for i, neighbors in preprocess_result.stacking_sets.items():
        for j in neighbors:
            ii, jj = min(i, j), max(i, j)
            Q[ii][jj] += stacking_reward / 2

    for (i, j) in preprocess_result.crossing_pairs:
        ii, jj = min(i, j), max(i, j)
        Q[ii][jj] += crossing_penalty

    return Q


def build_quadratic_program(preprocess_result, stacking_reward=-2.0, crossing_penalty=10.0):
    qp = QuadraticProgram("mRNA_folding")

    var_names = []
    for i, q in enumerate(preprocess_result.quartets):
        name = f"q{i}_k{q.k}_l{q.l}"
        qp.binary_var(name)
        var_names.append(name)

    linear = {}
    quadratic = {}

    for i, energy in preprocess_result.quartet_energies.items():
        linear[var_names[i]] = energy

    for i, neighbors in preprocess_result.stacking_sets.items():
        for j in neighbors:
            if i < j:
                key = (var_names[i], var_names[j])
                quadratic[key] = quadratic.get(key, 0.0) + stacking_reward

    for (i, j) in preprocess_result.crossing_pairs:
        ii, jj = min(i, j), max(i, j)
        key = (var_names[ii], var_names[jj])
        quadratic[key] = quadratic.get(key, 0.0) + crossing_penalty

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def build_qubo(preprocess_result, stacking_reward=-2.0, crossing_penalty=10.0):
    return build_quadratic_program(preprocess_result, stacking_reward, crossing_penalty)
