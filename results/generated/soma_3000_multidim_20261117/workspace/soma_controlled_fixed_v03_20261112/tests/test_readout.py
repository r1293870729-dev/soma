from pathlib import Path
import sys
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.manual_v03.readout import FrozenReadout


class ReadoutTests(unittest.TestCase):
    def test_nonlinear_readout_applied_separately(self):
        class SquareReadout(FrozenReadout):
            def __call__(self, value):
                return value.detach().square()
        readout = SquareReadout('controlled', {'kind': 'identity'}, torch.device('cpu'))
        x = torch.tensor([[3., 4.]], requires_grad=True)
        predicted = torch.tensor([[1., 2.]], requires_grad=True)
        value = readout.residual(x, predicted)
        self.assertEqual(value.tolist(), [[8., 12.]])
        self.assertIsNone(x.grad)
        self.assertIsNone(predicted.grad)

    def test_no_random_projection_substitution(self):
        with self.assertRaises(ValueError):
            FrozenReadout('cifar', {'kind': 'identity'}, torch.device('cpu'))
        with self.assertRaises(ValueError):
            FrozenReadout('controlled', {'kind': 'random_projection'}, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
