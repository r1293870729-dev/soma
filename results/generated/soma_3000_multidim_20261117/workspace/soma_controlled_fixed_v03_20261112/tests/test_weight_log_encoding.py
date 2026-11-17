from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_manual import validate_fixed_weights

class WeightLogEncodingTests(unittest.TestCase):
    def test_actual_float32_csv_encoding_is_the_frozen_coefficient(self):
        validate_fixed_weights([{'reconstruction':'1.0','kl':'0.0010000000474974513'}])
    def test_real_coefficient_change_is_still_rejected(self):
        with self.assertRaises(ValueError):
            validate_fixed_weights([{'reconstruction':'1.0','kl':'0.0011000000489875674'}])
