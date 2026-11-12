"""Training invariants: complete batches and validation-based stopping."""

import unittest
from fragc3.batching import BalancedBatchSampler
from fragc3.stopping import EarlyStopping


class Rules(unittest.TestCase):
    def test_complete_balanced_batches(self):
        for n in [2, 3, 255, 256, 257, 512, 513, 1001]:
            batches = list(BalancedBatchSampler(range(n), 256))
            self.assertEqual(sum(batches, []), list(range(n)))
            self.assertGreaterEqual(min(map(len, batches)), 2)
            self.assertLessEqual(max(map(len, batches)), 256)
            self.assertLessEqual(max(map(len, batches)) - min(map(len, batches)), 1)

    def test_exact_best_and_patience_anchor(self):
        s = EarlyStopping("max", 12, 1e-4, 5)
        s.observe(0.7, 1, [0.001], [0.001])
        s.observe(0.70001, 2, [0.001], [0.001])
        self.assertEqual(s.best_epoch, 2)
        self.assertEqual(s.anchor_epoch, 1)
        self.assertEqual(s.bad_epochs, 1)

    def test_real_lr_drop_grace(self):
        s = EarlyStopping("min", 2, 0.01, 5)
        for epoch in range(1, 8):
            s.observe(1.0, epoch, [0.001], [0.0005] if epoch == 3 else [0.001])
            self.assertFalse(s.should_stop())
        s.observe(1.0, 8, [0.0005], [0.0005])
        self.assertTrue(s.should_stop())

    def test_no_drop_does_not_extend_grace(self):
        s = EarlyStopping("max", 2, 0.01, 5)
        for epoch in range(1, 4):
            s.observe(0.5, epoch, [1e-5], [1e-5])
        self.assertTrue(s.should_stop())

    def test_restore_policy(self):
        s = EarlyStopping("min", 12, 0.001, 5)
        s.observe(1.0, 1, [0.001], [0.001])
        t = EarlyStopping("min", 12, 0.001, 5)
        t.load_state_dict(s.state_dict())
        self.assertEqual(t.state_dict(), s.state_dict())
        with self.assertRaises(ValueError):
            EarlyStopping("min", 5, 0.001, 5).load_state_dict(s.state_dict())


if __name__ == "__main__":
    unittest.main()
