import numpy as np
import pytest

from mrna_qfold.preprocessing import (
    Quartet,
    build_combination_matrix,
    enumerate_quartets,
    find_crossing_pairs,
    find_stacking_sets,
    parse_sequence,
    preprocess,
)


class TestParseSequence:
    def test_valid_uppercase(self):
        assert parse_sequence("AUGC") == "AUGC"

    def test_valid_lowercase(self):
        assert parse_sequence("augc") == "AUGC"

    def test_invalid_base(self):
        with pytest.raises(ValueError):
            parse_sequence("AUTC")

    def test_too_short(self):
        with pytest.raises(ValueError):
            parse_sequence("AUG")


class TestCombinationMatrix:
    def test_au_pair(self):
        seq = "AGCGU"
        matrix = build_combination_matrix(seq)
        assert matrix[0][4] == 1

    def test_min_loop_enforced(self):
        seq = "AUAU"
        matrix = build_combination_matrix(seq)
        assert matrix[0][3] == 0  # |0-3| = 3, need > 3


class TestQuartets:
    def test_simple_stem(self):
        seq = "GCAAAGC"
        matrix = build_combination_matrix(seq)
        quartets = enumerate_quartets(seq, matrix)
        assert Quartet(k=0, l=6) in quartets

    def test_no_quartets(self):
        seq = "AAAAAAA"
        matrix = build_combination_matrix(seq)
        assert len(enumerate_quartets(seq, matrix)) == 0


class TestPreprocess:
    def test_full_pipeline(self):
        result = preprocess("GCAAAGC")
        assert result.sequence == "GCAAAGC"
        assert len(result.quartets) > 0
        assert len(result.quartet_energies) == len(result.quartets)

    def test_longer_sequence(self):
        result = preprocess("GCGCAAAGCGC")
        assert len(result.quartets) > 0


# TODO: tests for qubo, postprocessing, classical baseline
