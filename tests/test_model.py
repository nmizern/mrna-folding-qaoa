from itertools import product

import numpy as np

from mrna_qfold.postprocessing import (
    decode_bitstring,
    evaluate_qubo_energy,
    postprocess,
    validate_structure,
)
from mrna_qfold.preprocessing import Quartet, preprocess
from mrna_qfold.quantum_solver import (
    QuantumResult,
    _sample_probabilities,
    solve_exact,
    solve_qaoa,
)
from mrna_qfold.qubo import build_qubo, build_qubo_matrix


def test_adjacent_quartets_form_one_stem():
    result = preprocess("GGCCAAAUGGCC")

    assert result.stacking_sets == {0: {1}, 1: {0, 2}, 2: {1}}
    assert result.crossing_pairs == set()

    active, structure = decode_bitstring("111", result.quartets, len(result.sequence))
    assert validate_structure(active, len(result.sequence))
    assert structure == "((((....))))"


def test_same_base_with_different_partners_is_invalid():
    quartets = [Quartet(0, 8), Quartet(0, 10)]
    assert not validate_structure(quartets, 11)


def test_qubo_representations_have_the_same_energy():
    result = preprocess("GCGCAAAGCGC")
    quadratic_program = build_qubo(result)
    matrix = build_qubo_matrix(result)

    for values in product([0, 1], repeat=len(result.quartets)):
        bitstring = "".join(str(value) for value in values)
        expected = evaluate_qubo_energy(bitstring, result)
        vector = np.array(values)

        assert np.isclose(quadratic_program.objective.evaluate(list(values)), expected)
        assert np.isclose(vector @ matrix @ vector, expected)


def test_postprocessing_uses_probabilities_and_qubo_parameters():
    result = preprocess("GCGCAAAGCGC")
    quantum_result = QuantumResult(
        best_bitstring="01110",
        best_objective=-16.2,
        samples={"01110": 0.75, "11000": 0.25},
    )

    processed = postprocess(
        quantum_result,
        result,
        stacking_reward=-3.5,
        crossing_penalty=12.0,
    )

    assert processed.valid_fraction == 0.75
    assert processed.best_candidate.dot_bracket == "((((...))))"
    assert processed.best_candidate.qubo_energy == -16.2


def test_exact_solver_returns_variable_values():
    result = preprocess("GCAAAGC")
    exact = solve_exact(build_qubo(result))

    assert exact.best_bitstring == "1"
    assert exact.best_objective == -3.4
    assert exact.samples == {"1": 1.0}


def test_solvers_handle_sequence_without_quartets():
    result = preprocess("AAAAAAA")
    quadratic_program = build_qubo(result)

    for solved in (solve_exact(quadratic_program), solve_qaoa(quadratic_program)):
        assert solved.best_bitstring == ""
        assert solved.best_objective == 0.0
        assert solved.samples == {"": 1.0}


def test_sampler_probabilities_are_not_squared_again():
    class RawResult:
        eigenstate = {"00": 0.25, "01": 0.75}

    class Result:
        min_eigen_solver_result = RawResult()

    assert _sample_probabilities(Result(), 2) == {"00": 0.25, "10": 0.75}


def test_sampler_counts_are_normalized():
    class RawResult:
        eigenstate = {"0": 3, "1": 1}

    class Result:
        min_eigen_solver_result = RawResult()

    assert _sample_probabilities(Result(), 1) == {"0": 0.75, "1": 0.25}
