from pathlib import Path
import copy
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.b_weight_sensitivity.formal_common import load_config, planned_units, arm_specs


class FixedPlanTests(unittest.TestCase):
    def test_exactly_three_fixed_controlled_runs(self):
        units = planned_units(load_config())
        self.assertEqual(len(units), 3)
        self.assertEqual([u['seed'] for u in units], [0, 1, 2])
        self.assertTrue(all(u['dataset'] == 'controlled' and u['method'] == 'fixed'
                            and u['beta'] == .001 and u['arm'] == 'historical_fixed' for u in units))

    def test_changed_human_weights_require_new_decision(self):
        config = copy.deepcopy(load_config())
        config['fixed_coefficients']['kl'] = 1.
        with self.assertRaises(ValueError):
            arm_specs(config)
