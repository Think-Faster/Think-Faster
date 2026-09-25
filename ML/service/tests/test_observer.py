import unittest

import numpy as np

from observer import psi


class ObserverPsiTest(unittest.TestCase):
    def test_psi_on_same_distribution_is_small(self):
        ref = np.random.default_rng(0).random(1000)
        win = ref[:500]
        val, grade = psi(ref, win)
        self.assertTrue(np.isfinite(val))
        self.assertIn(grade, (0, 1))

    def test_psi_on_shifted_distribution_grows(self):
        ref = np.random.default_rng(1).random(1000)
        win = np.random.default_rng(2).random(500) * 0.2 + 0.8
        a, _ = psi(ref, win)
        same = ref[:500]
        b, _ = psi(ref, same)
        self.assertGreater(a, b)

    def test_psi_tolerates_quantile_ties(self):
        """B9: массовые ти-и в рангах парка не роняют histogram (границы кванттелей уникальны)."""
        ref = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.9, 0.9, 0.9, 0.9], np.float32)
        win = np.array([0.3, 0.3, 0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], np.float32)
        val, grade = psi(ref, win, bins=4)
        self.assertTrue(np.isfinite(val))
        self.assertIn(grade, (0, 1, 2))


if __name__ == '__main__':
    unittest.main()