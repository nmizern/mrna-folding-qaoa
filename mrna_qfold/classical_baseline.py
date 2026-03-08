import time

import numpy as np

from .energy_params import is_valid_pair

MIN_LOOP = 3


def nussinov_fill(sequence):
    n = len(sequence)
    dp = np.zeros((n, n), dtype=int)

    for span in range(MIN_LOOP + 1, n):
        for i in range(n - span):
            j = i + span
            best = dp[i + 1][j]
            best = max(best, dp[i][j - 1])

            if is_valid_pair(sequence[i], sequence[j]):
                inner = dp[i + 1][j - 1] if (i + 1 <= j - 1) else 0
                best = max(best, inner + 1)

            for k in range(i + 1, j):
                best = max(best, dp[i][k] + dp[k + 1][j])

            dp[i][j] = best
    return dp


def nussinov_traceback(sequence, dp):
    n = len(sequence)
    structure = list("." * n)

    def traceback(i, j):
        if i >= j:
            return
        if dp[i][j] == dp[i + 1][j]:
            traceback(i + 1, j)
        elif dp[i][j] == dp[i][j - 1]:
            traceback(i, j - 1)
        elif (is_valid_pair(sequence[i], sequence[j])
              and dp[i][j] == (dp[i + 1][j - 1] if i + 1 <= j - 1 else 0) + 1):
            structure[i] = "("
            structure[j] = ")"
            traceback(i + 1, j - 1)
        else:
            for k in range(i + 1, j):
                if dp[i][j] == dp[i][k] + dp[k + 1][j]:
                    traceback(i, k)
                    traceback(k + 1, j)
                    break

    traceback(0, n - 1)
    return "".join(structure)


def nussinov_predict(sequence):
    seq = sequence.strip().upper()
    t0 = time.perf_counter()
    dp = nussinov_fill(seq)
    structure = nussinov_traceback(seq, dp)
    elapsed = time.perf_counter() - t0

    return {
        "structure": structure,
        "num_pairs": structure.count("("),
        "energy": None,
        "elapsed_seconds": elapsed,
    }


def vienna_mfe_predict(sequence):
    seq = sequence.strip().upper()
    t0 = time.perf_counter()
    try:
        import RNA
        fc = RNA.fold_compound(seq)
        structure, energy = fc.mfe()
        return {
            "structure": structure,
            "num_pairs": structure.count("("),
            "energy": energy,
            "elapsed_seconds": time.perf_counter() - t0,
        }
    except ImportError:
        return {
            "structure": "." * len(seq),
            "num_pairs": 0,
            "energy": None,
            "elapsed_seconds": time.perf_counter() - t0,
        }


def classical_benchmark(sequence):
    return {
        "nussinov": nussinov_predict(sequence),
        "vienna_mfe": vienna_mfe_predict(sequence),
    }
