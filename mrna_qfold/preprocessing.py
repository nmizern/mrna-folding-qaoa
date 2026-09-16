# Based on Zaborniak et al. (2022) and Alevras et al. (2024)

from dataclasses import dataclass, field

import numpy as np

from .energy_params import is_valid_pair, get_quartet_energy

MIN_LOOP_LENGTH = 3  # minimum unpaired bases in a hairpin


@dataclass(frozen=True)
class Quartet:
    # two consecutive stacked base pairs: (k, k+1) -- (l, l-1)
    k: int
    l: int

    @property
    def positions(self):
        return {self.k, self.k + 1, self.l - 1, self.l}

    @property
    def pairs(self):
        return {(self.k, self.l), (self.k + 1, self.l - 1)}

    def __repr__(self):
        return f"Q(k={self.k}, l={self.l})"


@dataclass
class PreprocessingResult:
    sequence: str
    combination_matrix: np.ndarray
    quartets: list[Quartet]
    quartet_index: dict[Quartet, int] = field(default_factory=dict)
    quartet_energies: dict[int, float] = field(default_factory=dict)
    stacking_sets: dict[int, set[int]] = field(default_factory=dict)
    crossing_pairs: set[tuple[int, int]] = field(default_factory=set)


def parse_sequence(sequence):
    seq = sequence.strip().upper()
    valid_bases = set("AUGC")
    for i, base in enumerate(seq):
        if base not in valid_bases:
            raise ValueError(f"Invalid base '{base}' at position {i}")
    if len(seq) < 4:
        raise ValueError(f"Sequence too short ({len(seq)} bases), need at least 4")
    return seq


def build_combination_matrix(sequence):
    n = len(sequence)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + MIN_LOOP_LENGTH + 1, n):
            if is_valid_pair(sequence[i], sequence[j]):
                matrix[i][j] = 1
    return matrix


def enumerate_quartets(sequence, combination_matrix):
    n = len(sequence)
    quartets = []
    for k in range(n - 3):
        for l in range(k + MIN_LOOP_LENGTH + 2, n):
            if l - 1 <= k + 1:
                continue
            if combination_matrix[k][l] == 1 and combination_matrix[k + 1][l - 1] == 1:
                quartets.append(Quartet(k=k, l=l))
    quartets.sort(key=lambda q: (q.k, q.l))
    return quartets


def compute_quartet_energies(sequence, quartets):
    energies = {}
    for i, q in enumerate(quartets):
        energies[i] = get_quartet_energy(
            sequence[q.k], sequence[q.l],
            sequence[q.k + 1], sequence[q.l - 1],
        )
    return energies


def find_stacking_sets(quartets):
    lookup = {(q.k, q.l): i for i, q in enumerate(quartets)}
    stacking = {i: set() for i in range(len(quartets))}

    for i, q in enumerate(quartets):
        outer = (q.k - 1, q.l + 1)
        if outer in lookup:
            j = lookup[outer]
            stacking[i].add(j)
            stacking[j].add(i)
        inner = (q.k + 1, q.l - 1)
        if inner in lookup:
            j = lookup[inner]
            stacking[i].add(j)
            stacking[j].add(i)

    return stacking


def find_crossing_pairs(quartets):
    crossing = set()
    n = len(quartets)
    for i in range(n):
        for j in range(i + 1, n):
            pairs = quartets[i].pairs | quartets[j].pairs
            partners = {}
            has_overlap = False
            for left, right in pairs:
                if ((left in partners and partners[left] != right)
                        or (right in partners and partners[right] != left)):
                    has_overlap = True
                    break
                partners[left] = right
                partners[right] = left

            if has_overlap:
                crossing.add((i, j))
                continue

            sorted_pairs = sorted(pairs)
            for a, (left_a, right_a) in enumerate(sorted_pairs):
                for left_b, right_b in sorted_pairs[a + 1:]:
                    if left_a < left_b < right_a < right_b:
                        crossing.add((i, j))
                        break
                if (i, j) in crossing:
                    break

    return crossing


def preprocess(sequence):
    seq = parse_sequence(sequence)
    combo = build_combination_matrix(seq)
    quartets = enumerate_quartets(seq, combo)
    q_index = {q: i for i, q in enumerate(quartets)}
    energies = compute_quartet_energies(seq, quartets)
    stacking = find_stacking_sets(quartets)
    crossing = find_crossing_pairs(quartets)

    return PreprocessingResult(
        sequence=seq,
        combination_matrix=combo,
        quartets=quartets,
        quartet_index=q_index,
        quartet_energies=energies,
        stacking_sets=stacking,
        crossing_pairs=crossing,
    )
