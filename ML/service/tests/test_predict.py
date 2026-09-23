import unittest

from predict import rank_mean

import numpy as np


class PredictTest(unittest.TestCase):
    def test_rank_mean_maps_to_0_1_and_is_robust(self):
        rng = np.random.default_rng(0)
        p = np.stack([rng.random(50), rng.random(50)])     # два зерна
        s = rank_mean(p)
        self.assertEqual(s.shape, (50,))
        self.assertTrue((s >= 0).all() and (s <= 1).all())

    def test_rank_ignores_calibration(self):
        # монотонное сжатие вероятности не меняет ранг — место в распределении устойчиво
        p = np.array([[0.01, 0.5, 0.99], [0.4, 0.6, 0.8]])
        flat = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        self.assertTrue(np.allclose(rank_mean(p), rank_mean(np.tanh(p) / 2 + 0.5)))
        self.assertEqual(float(rank_mean(flat).mean()), 0.5)


if __name__ == '__main__':
    unittest.main()