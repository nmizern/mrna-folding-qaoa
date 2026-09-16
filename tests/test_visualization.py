import matplotlib.pyplot as plt

from mrna_qfold.preprocessing import preprocess
from mrna_qfold.postprocessing import FoldingCandidate
from mrna_qfold.visualization import (
    plot_combination_matrix,
    plot_qaoa_convergence,
    plot_energy_landscape,
    plot_structure_comparison,
    plot_benchmark_summary,
)


def test_plot_combination_matrix():
    prep = preprocess("GCGCAAAGCGC")
    fig = plot_combination_matrix(prep.combination_matrix, prep.sequence, prep.quartets)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_qaoa_convergence():
    history = [12.0, 5.0, 1.2, -0.5, -2.1]
    fig = plot_qaoa_convergence(history, exact_energy=-3.4)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_energy_landscape_with_candidates():
    candidates = [
        FoldingCandidate("1", [], "((...))", True, -3.4, probability=0.8),
        FoldingCandidate("0", [], ".......", False, 5.0, probability=0.2),
    ]
    fig = plot_energy_landscape(candidates, best_energy=-3.4)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_structure_comparison():
    fig = plot_structure_comparison("GCGCAAAGCGC", "((((...))))", "((((...))))", -13.2, None)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_benchmark_summary():
    records = [{
        "length": 7,
        "qubits": 1,
        "time_qaoa_median_s": 0.4,
        "time_nussinov_median_s": 0.001,
        "qaoa_ground_probability_mean": 0.8,
        "qaoa_ground_probability_std": 0.1,
        "random_ground_probability": 0.5,
        "qaoa_valid_probability_mean": 1.0,
        "qaoa_valid_probability_std": 0.0,
        "random_valid_probability": 1.0,
    }]
    fig = plot_benchmark_summary(records)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
