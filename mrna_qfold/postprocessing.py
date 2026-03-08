
from dataclasses import dataclass, field
from typing import Optional

from .preprocessing import PreprocessingResult, Quartet


@dataclass
class FoldingCandidate:
    bitstring: str
    active_quartets: list[Quartet]
    dot_bracket: str
    is_valid: bool
    qubo_energy: float
    vienna_energy: Optional[float] = None
    count: int = 1


@dataclass
class PostprocessingResult:
    sequence: str
    candidates: list[FoldingCandidate] = field(default_factory=list)
    best_candidate: Optional[FoldingCandidate] = None
    total_samples: int = 0
    valid_fraction: float = 0.0


def decode_bitstring(bitstring, quartets, sequence_length):
    active = [quartets[i] for i, bit in enumerate(bitstring) if bit == "1"]

    structure = list("." * sequence_length)
    for q in active:
        structure[q.k] = "("
        structure[q.k + 1] = "("
        structure[q.l - 1] = ")"
        structure[q.l] = ")"

    return active, "".join(structure)


def validate_structure(active_quartets, sequence_length):
    if not active_quartets:
        return True

    # position overlap check
    all_pos = set()
    for q in active_quartets:
        if q.positions & all_pos:
            return False
        all_pos |= q.positions

    # pseudoknot check
    pairs = []
    for q in active_quartets:
        pairs.append((q.k, q.l))
        pairs.append((q.k + 1, q.l - 1))
    pairs.sort()
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            i1, j1 = pairs[a]
            i2, j2 = pairs[b]
            if i1 < i2 < j1 < j2:
                return False

    return True


def compute_vienna_energy(sequence, dot_bracket):
    try:
        import RNA
        fc = RNA.fold_compound(sequence)
        return fc.eval_structure(dot_bracket)
    except ImportError:
        return None


def compute_vienna_mfe(sequence):
    try:
        import RNA
        fc = RNA.fold_compound(sequence)
        structure, energy = fc.mfe()
        return structure, energy
    except ImportError:
        return None, None


def evaluate_qubo_energy(bitstring, preprocess_result, stacking_reward=-2.0, crossing_penalty=10.0):
    bits = [int(b) for b in bitstring]
    energy = 0.0

    for i, e in preprocess_result.quartet_energies.items():
        energy += e * bits[i]
    for i, neighbors in preprocess_result.stacking_sets.items():
        for j in neighbors:
            if i < j:
                energy += stacking_reward * bits[i] * bits[j]
    for (i, j) in preprocess_result.crossing_pairs:
        ii, jj = min(i, j), max(i, j)
        energy += crossing_penalty * bits[ii] * bits[jj]

    return energy


def postprocess(quantum_result, preprocess_result, top_k=10):
    seq = preprocess_result.sequence
    quartets = preprocess_result.quartets
    n = len(seq)

    candidates = []
    total_count = 0

    all_bitstrings = quantum_result.all_samples
    if not all_bitstrings:
        all_bitstrings = {quantum_result.best_bitstring: 1}

    for bitstring, count in all_bitstrings.items():
        total_count += count
        active_q, dot_bracket = decode_bitstring(bitstring, quartets, n)
        is_valid = validate_structure(active_q, n)
        qubo_e = evaluate_qubo_energy(bitstring, preprocess_result)
        vienna_e = compute_vienna_energy(seq, dot_bracket) if is_valid else None

        candidates.append(FoldingCandidate(
            bitstring=bitstring,
            active_quartets=active_q,
            dot_bracket=dot_bracket,
            is_valid=is_valid,
            qubo_energy=qubo_e,
            vienna_energy=vienna_e,
            count=count,
        ))

    candidates.sort(key=lambda c: (not c.is_valid, c.qubo_energy))
    top = candidates[:top_k]
    valid_count = sum(c.count for c in candidates if c.is_valid)
    best = next((c for c in candidates if c.is_valid), candidates[0] if candidates else None)

    return PostprocessingResult(
        sequence=seq,
        candidates=top,
        best_candidate=best,
        total_samples=total_count,
        valid_fraction=valid_count / total_count if total_count > 0 else 0.0,
    )
