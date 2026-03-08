# mRNA Folding with Quantum Computing

GenQ Hackathon project -- predicting mRNA secondary structure using QAOA on IonQ Aria via [QCentroid](https://qcentroid.xyz).

## What this does

Takes an mRNA sequence (A/U/G/C), formulates the folding problem as a QUBO using quartet-based encoding (Fox et al. 2022), and solves it with QAOA. We ran it on IonQ Aria trapped-ion hardware during the hackathon and compared against classical baselines (Nussinov, ViennaRNA).

The pipeline:
```
sequence → preprocessing → QUBO → QAOA → postprocessing → dot-bracket structure
              (quartets,      (energy +      (IonQ /        (validate,
               stacking,       stacking +     local sim)      rank by E)
               crossings)      penalties)
```

## Quick start

```bash
pip install -r requirements.txt
python app.py --sequence GCGCAAAGCGC --reps 2
python app.py --all  # run all test sequences
```

The QCentroid entry point is `qcentroid.py` (standard solver template format).

## Project structure

```
├── qcentroid.py              # QCentroid solver entry point
├── app.py                    # local testing CLI
├── mrna_qfold/
│   ├── energy_params.py      # Turner 2004 stacking parameters
│   ├── preprocessing.py      # sequence → quartets, stacking sets, crossings
│   ├── qubo.py               # QUBO matrix construction
│   ├── quantum_solver.py     # QAOA (local / IonQ / QCentroid)
│   ├── postprocessing.py     # bitstring → dot-bracket + validation
│   ├── classical_baseline.py # Nussinov DP baseline
│   └── visualization.py      # plots
└── tests/
```

## Results

On the local simulator (statevector, QAOA p=2), we match the exact ground state for small sequences (7-12 nt). On IonQ Aria with p=1, we got correct structures in the top-3 sampled bitstrings for sequences up to 11 nt. Noise was the main limitation for going longer.

Classical methods (Nussinov, ViennaRNA) are obviously faster at these sizes -- quantum advantage isn't the point here. The point is demonstrating the formulation works end-to-end on real hardware.

## How it works

1. **Preprocessing**: Parse sequence, build combination matrix, enumerate quartets (consecutive stacked base pairs), compute Turner 2004 energies, find stacking/crossing relationships
2. **QUBO**: H(q) = Σ e_i·q_i + r·Σ q_i·q_j (stacking) + λ·Σ q_i·q_j (crossing penalty)
3. **QAOA**: Map QUBO to Ising Hamiltonian, run QAOA with COBYLA optimizer
4. **Postprocessing**: Decode bitstrings to dot-bracket, validate (no overlaps/pseudoknots), rank by energy

## References

- Fox et al. (2022) [arXiv:2208.04367](https://arxiv.org/abs/2208.04367) -- quartet-based QUBO for RNA folding
- Robert et al. (2024) [arXiv:2405.20328](https://arxiv.org/abs/2405.20328) -- extended approach, IonQ results
- Turner 2004 nearest-neighbor parameters (Mathews et al., PNAS 2004)

## Hackathon

Built at the **GenQ Hackathon** (Microsoft, IonQ, Moderna) addressing [UN SDG 3](https://sdgs.un.org/goals/goal3).
Platform: [QCentroid](https://qcentroid.xyz) | Hardware: [IonQ Aria](https://ionq.com)
