from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.manual_v03.contract import require_author_decisions


class ContractTests(unittest.TestCase):
    def test_absent_author_decisions_prevent_full_rerun(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, 'author-input-needed'):
                require_author_decisions(Path(name))
