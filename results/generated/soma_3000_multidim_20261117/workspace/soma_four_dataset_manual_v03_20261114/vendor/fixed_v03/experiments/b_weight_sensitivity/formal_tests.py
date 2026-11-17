"""Integration contracts for the new full-budget wiring; run before any sweep."""
import copy
import unittest
import numpy as np
import torch
from torch.optim import AdamW
from .formal_common import ROOT, load_config, planned_units
from .formal_data import role_mask, select_ids
from .formal_runner import (args_for, checked_update, initial_references, losses_for,
                            make_controller, tensor_digest, validation)
from .gradient_diagnostics import collect_gradient_vectors
from .historical_loader import HistoricalSource
from .weighting import assert_disjoint_optimizers


class FormalContracts(unittest.TestCase):
    def setUp(self):
        self.source = HistoricalSource(ROOT / 'legacy/03_psi_xib_unified')
        self.entry = self.source.load('train_xib_unified')
        self.models = self.source.load('models')
        self.config = load_config()
        self.args = args_for(self.entry, self.config, 'controlled')
        torch.manual_seed(123)
        self.model, self.likelihood = self.entry.build_model(self.args, 4, torch.device('cpu'))
        self.x = torch.randn(8, 64, generator=torch.Generator().manual_seed(7))

    def test_global_roles_disjoint_across_nested_stages(self):
        mask = role_mask(1000, 4)
        train, val = set(), set()
        for index, pool in enumerate([np.arange(30), np.arange(300), np.arange(1000)]):
            train.update(select_ids(pool, mask, True, 4096, index).tolist())
            val.update(select_ids(pool, mask, False, 4096, index + 50).tolist())
        self.assertFalse(train & val)
        self.assertEqual(int(mask.sum()), 700)

    def test_same_base_image_roles_across_corruptions(self):
        one = role_mask(10000, 2)
        two = role_mask(10000, 2)
        t = select_ids(np.arange(10000), one, True, 4096, 5)
        v = select_ids(np.arange(10000), two, False, 4096, 6)
        self.assertFalse(set(t) & set(v))

    def test_fixed_step_matches_original_objective(self):
        direct = copy.deepcopy(self.model)
        opt_direct = AdamW(direct.parameters(), lr=.001, weight_decay=0)
        opt = AdamW(self.model.parameters(), lr=.001, weight_decay=0)
        state = torch.get_rng_state()
        l, _, _ = losses_for(self.entry, self.models, self.args, direct, self.x, self.likelihood)
        opt_direct.zero_grad(set_to_none=True)
        (l['reconstruction'] + .001 * l['kl']).backward(); opt_direct.step()
        torch.set_rng_state(state)
        l, _, _ = losses_for(self.entry, self.models, self.args, self.model, self.x, self.likelihood)
        rows, _ = collect_gradient_vectors(l, self.model.encoder.parameters())
        control, control_opt = make_controller({'method': 'fixed', 'beta': .001},
                                              {'reconstruction': 1., 'kl': 1.}, self.config, torch.device('cpu'))
        checked_update(self.model, opt, control, control_opt, l, torch.tensor([r['raw_norm'] for r in rows]))
        self.assertEqual(tensor_digest(direct), tensor_digest(self.model))

    def test_gradnorm_integrated_step_normalizes_weights(self):
        l, _, _ = losses_for(self.entry, self.models, self.args, self.model, self.x, self.likelihood)
        refs = {k: float(v.detach()) for k, v in l.items()}
        control, co = make_controller({'method': 'gradnorm'}, refs, self.config, torch.device('cpu'))
        opt = AdamW(self.model.parameters(), lr=.001, weight_decay=0)
        assert_disjoint_optimizers(opt, co)
        rows, _ = collect_gradient_vectors(l, self.model.encoder.parameters())
        checked_update(self.model, opt, control, co, l, torch.tensor([r['raw_norm'] for r in rows]))
        self.assertAlmostEqual(sum(control.weights_dict().values()), 2., places=6)
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))

    def test_uncertainty_has_separate_parameter_update(self):
        l, _, _ = losses_for(self.entry, self.models, self.args, self.model, self.x, self.likelihood)
        control, co = make_controller({'method': 'uncertainty'}, {}, self.config, torch.device('cpu'))
        opt = AdamW(self.model.parameters(), lr=.001, weight_decay=0)
        assert_disjoint_optimizers(opt, co)
        checked_update(self.model, opt, control, co, l, torch.ones(2))
        self.assertTrue(any(float(p.detach()) != 0 for p in control.parameters()))

    def test_validation_restores_model_mode_and_rng(self):
        bundle = self.entry.build_controlled_bundle(self.args, 0)
        stage = bundle.probe_stages[0]
        tiny = self.source.load('data').StageSpec(stage.name, torch.utils.data.Subset(stage.dataset, range(8)), stage.active_tasks, stage.num_classes)
        rng, before = torch.get_rng_state().clone(), tensor_digest(self.model)
        first = validation(self.entry, self.models, self.args, self.model, [tiny], self.likelihood, 0, torch.device('cpu'))
        second = validation(self.entry, self.models, self.args, self.model, [tiny], self.likelihood, 0, torch.device('cpu'))
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, tensor_digest(self.model)); self.assertEqual(first, second)

    def test_calibration_is_read_only_and_positive(self):
        bundle = self.entry.build_controlled_bundle(self.args, 0)
        before, rng = tensor_digest(self.model), torch.get_rng_state().clone()
        refs = initial_references(self.entry, self.models, self.args, self.model, bundle.train_stages[0], self.likelihood, 0, torch.device('cpu'), 8)
        self.assertTrue(all(v > 0 for v in refs.values()))
        self.assertEqual(before, tensor_digest(self.model))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_plan_is_exactly_approved_budget(self):
        units = planned_units(self.config)
        self.assertEqual(len(units), 240)
        self.assertEqual(len({(u['dataset'], u['latent_dim'], u['seed']) for u in units}), 30)
        self.assertEqual(sum(self.config['stage_steps']) * len(units), 720000)


if __name__ == '__main__':
    unittest.main()
