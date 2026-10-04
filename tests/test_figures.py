import unittest

from scripts.make_figures import spearman


class FigureMetricTests(unittest.TestCase):
    def test_spearman_handles_perfect_monotonic_relationships(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [30, 20, 10]), -1.0)

    def test_paper_figure_values_have_reported_correlation(self):
        baseline_accuracy = [6.1, 16.6, 18.5, 33.8, 41.1, 45.6]
        remove_minus_tag = [0.5, -1.2, -3.4, -8.7, -9.5, -11.7]

        self.assertAlmostEqual(spearman(baseline_accuracy, remove_minus_tag), -1.0)


if __name__ == "__main__":
    unittest.main()
