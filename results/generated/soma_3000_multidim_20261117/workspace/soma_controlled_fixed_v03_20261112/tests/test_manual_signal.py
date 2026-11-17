import copy
from dataclasses import replace
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests/oracle'))
from experiments.manual_v03.signal import SignalConfig, FixedPsi, SignalError, shape_spectrum, mass, validate_roles, midrank_position
from experiments.manual_v03.roles import partition_source, draws
from soma_b import BMonitor, BConfig


class SignalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(702)
        self.white = rng.normal(size=(128, 7))
        self.edge = rng.normal(size=(16 * 20, 7))
        self.online = rng.normal(size=(96, 7)) * [4, 1, 1, .3, .3, .3, .3]
        self.config = SignalConfig(16, .1, .05, 1e-12, 1e-10)
        self.roles = dict(white=['w:' + str(i) for i in range(128)],
                          edge=['e:' + str(i) for i in range(320)])

    def monitor(self):
        return FixedPsi.fit(self.white, self.edge, self.config, self.roles)

    def test_independent_prior_reference_agrees_every_window(self):
        old = BMonitor.fit(self.white, self.edge, BConfig(16, .1, .05, 1e-12, 1e-10), self.roles)
        new = self.monitor()
        for chunk in np.array_split(self.online, 12):
            a, b = old.update(chunk), new.update(chunk)
            self.assertEqual(a['status'], b['status'])
            self.assertIsNone(b['psi']) if a['psi_b'] is None else self.assertAlmostEqual(a['psi_b'], b['psi'], delta=1e-10)
            np.testing.assert_allclose(a['spectrum'], b['spectrum'], atol=1e-12)

    def test_mass_multiple_modes_and_peak_distinction(self):
        a = mass([2, 1.8, .1, .1], 1.5)
        b = mass([2, 1, .5, .5], 1.5)
        self.assertAlmostEqual(a['psi'], 2 / 1.5 - 1 + 1.8 / 1.5 - 1)
        self.assertEqual(a['peak_excess'], b['peak_excess'])
        self.assertGreater(a['psi'], b['psi'])

    def test_single_mode_mass_equals_peak(self):
        a = mass([3, .5, .3, .2], 1.5)
        self.assertEqual(a['psi'], a['peak_excess'])

    def test_positive_iff_peak_strictly_exceeds_edge(self):
        for edge in [.1, 1, 2, 4, 8]:
            self.assertEqual(mass([4, 2, 1, 0], edge)['psi'] > 0, 4 > edge)

    def test_threshold_equality_zero(self):
        self.assertEqual(mass([2, 2, 0], 2)['psi'], 0)

    def test_signal_is_not_a_probability(self):
        self.assertGreater(mass([4, 0, 0, 0], 1)['psi'], 1)

    def test_low_energy_equality_retained(self):
        x = np.array([[-1., 0], [1., 0]])
        cfg = replace(self.config, window=2, energy_floor=2.)
        eig, trace, status, _ = shape_spectrum(x, cfg)
        self.assertEqual(trace, 2.)
        self.assertEqual(status, 'low-energy')
        self.assertEqual(float(eig.sum()), 0.)
        m = FixedPsi(np.zeros(2), np.eye(2), 1, cfg, {})
        result = m.update(x)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['psi'], 0.)

    def test_warming_not_observed_zero(self):
        row = self.monitor().update(self.online[:2])
        self.assertIsNone(row['psi'])
        self.assertFalse(row['eligible'])

    def test_uniform_scale_invariance(self):
        a = self.monitor().measure_window(self.online[-16:])
        for k in [-3., .1, 4.]:
            b = self.monitor().measure_window(k * self.online[-16:])
            self.assertAlmostEqual(a['psi'], b['psi'], places=10)

    def test_global_sign_flip_including_reference(self):
        a = self.monitor().update(self.online)
        b = FixedPsi.fit(-self.white, -self.edge, self.config, self.roles).update(-self.online)
        self.assertAlmostEqual(a['psi'], b['psi'], delta=1e-10)

    def test_no_stage_api_or_adaptive_reference(self):
        m = self.monitor(); before = m.reference_hash()
        m.update(self.online)
        self.assertEqual(before, m.reference_hash())
        with self.assertRaises(TypeError):
            m.update(self.online, stage_name='oracle')

    def test_all_rank_deficient_regimes_covariance_gram(self):
        rng = np.random.default_rng(19)
        for n, d in [(9, 40), (40, 9), (16, 16)]:
            x = rng.standard_t(3, size=(n, d)); x[:, -1] = x[:, 0]
            a = shape_spectrum(x, replace(self.config, backend='covariance'))[0]
            b = shape_spectrum(x, replace(self.config, backend='gram'))[0]
            self.assertEqual(len(a), min(d, n - 1))
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

    def test_order_statistic_not_interpolated_quantile(self):
        m = self.monitor()
        maxima = m.metadata['edge_maxima']; k = m.metadata['order_statistic_k']
        self.assertEqual(m.edge, sorted(maxima)[k - 1])
        self.assertEqual(k, 19)

    def test_zero_energy_edge_windows_count_in_denominator(self):
        white = np.array([[-1., -1], [-1, 1], [1, -1], [1, 1]])
        edge = np.vstack([np.zeros((4, 2)), white])
        ids = dict(white=['w' + str(i) for i in range(4)], edge=['e' + str(i) for i in range(8)])
        m = FixedPsi.fit(white, edge, replace(self.config, window=4, beta=.25), ids)
        self.assertEqual(m.metadata['edge_windows'], 2)
        self.assertEqual(m.metadata['low_energy_edge_windows'], 1)
        cover = m.coverage(np.zeros((8, 2)))
        self.assertEqual(cover['windows'], 2)
        self.assertEqual(cover['low_energy_windows'], 2)

    def test_bad_input_is_failure_not_window_exclusion(self):
        x = self.online.copy(); x[0, 0] = np.nan
        with self.assertRaises(SignalError): self.monitor().update(x)
        with self.assertRaises(SignalError): FixedPsi.fit(np.zeros((16, 7)), self.edge, self.config, self.roles)
        with self.assertRaises(SignalError): mass([1, 2], 0)
        with self.assertRaises(SignalError): mass([1, -.1], 1)
        with self.assertRaises(SignalError): FixedPsi.fit(self.white, self.edge[:-1], self.config, self.roles)

    def test_role_overlap_blocks_before_calibration(self):
        bad = copy.deepcopy(self.roles); bad['edge'][0] = bad['white'][0]
        with self.assertRaises(SignalError): FixedPsi.fit(self.white, self.edge, self.config, bad)

    def test_reserved_small_pool_and_replacement_honest(self):
        roles = partition_source(100, np.arange(100), 0, dict(white=.1, edge=.1, **{'edge-cover': .1}))
        self.assertEqual(sum(roles == 'white'), 10)
        self.assertEqual(sum(roles == 'train'), 49)
        self.assertEqual(sum(roles == 'validation'), 21)
        pool = np.flatnonzero(roles == 'edge')
        with self.assertRaises(ValueError): draws(pool, 128, 0, False)
        ids = draws(pool, 128, 0, True)
        self.assertEqual(len(ids), 128)
        self.assertLessEqual(len(set(ids)), 10)

    def test_display_midrank_ties(self):
        self.assertEqual(midrank_position([0, 1, 1, 2], 1), (1 + 1 + 1) / 5)

    def test_small_negative_eigenvalues_and_tolerance_boundary(self):
        x = np.array([[-1., 0.], [1., 0.]])
        tol = self.config.negative_tolerance
        with patch('numpy.linalg.eigvalsh', return_value=np.array([-tol, 2.])):
            eig = shape_spectrum(x, self.config)[0]
            self.assertTrue(np.all(eig >= 0))
        with patch('numpy.linalg.eigvalsh', return_value=np.array([-2 * tol, 2.])):
            with self.assertRaises(SignalError):
                shape_spectrum(x, self.config)

    def test_bad_configuration_rejected(self):
        for update in [dict(window=True), dict(window=1), dict(alpha=0), dict(beta=1),
                       dict(energy_floor=-1), dict(negative_tolerance=-1), dict(backend='mp')]:
            with self.subTest(update=update), self.assertRaises(SignalError):
                replace(self.config, **update)


if __name__ == '__main__':
    unittest.main()
