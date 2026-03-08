# Turner 2004 stacking parameters (https://rna.urmc.rochester.edu/NNDB/)

VALID_PAIRS = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
    ("G", "U"), ("U", "G"),
}

# stacking free energies in kcal/mol at 37C
# key: (5' closing pair, 3' closing pair)
STACKING_ENERGIES = {
    ("AU", "AU"): -0.9, ("AU", "CG"): -2.2, ("AU", "GC"): -2.1,
    ("AU", "UA"): -1.1, ("AU", "GU"): -1.4, ("AU", "UG"): -0.6,
    ("CG", "AU"): -2.1, ("CG", "CG"): -3.3, ("CG", "GC"): -2.4,
    ("CG", "UA"): -2.1, ("CG", "GU"): -2.1, ("CG", "UG"): -1.4,
    ("GC", "AU"): -2.4, ("GC", "CG"): -3.4, ("GC", "GC"): -3.3,
    ("GC", "UA"): -2.2, ("GC", "GU"): -2.5, ("GC", "UG"): -1.5,
    ("UA", "AU"): -1.3, ("UA", "CG"): -2.4, ("UA", "GC"): -2.1,
    ("UA", "UA"): -0.9, ("UA", "GU"): -1.0, ("UA", "UG"): -0.6,
    # wobble
    ("GU", "AU"): -1.3, ("GU", "CG"): -2.5, ("GU", "GC"): -2.1,
    ("GU", "UA"): -1.4, ("GU", "GU"): -0.5, ("GU", "UG"):  1.3,
    ("UG", "AU"): -1.0, ("UG", "CG"): -1.5, ("UG", "GC"): -1.4,
    ("UG", "UA"): -0.6, ("UG", "GU"): -0.3, ("UG", "UG"):  0.3,
}

TERMINAL_AU_PENALTY = 0.5


def is_valid_pair(base_i, base_j):
    return (base_i.upper(), base_j.upper()) in VALID_PAIRS


def get_quartet_energy(seq_k, seq_l, seq_k1, seq_l1):
    # stacking energy for quartet (k, k+1, l-1, l)
    outer = f"{seq_k.upper()}{seq_l.upper()}"
    inner = f"{seq_k1.upper()}{seq_l1.upper()}"
    return STACKING_ENERGIES.get((outer, inner), 0.0)
