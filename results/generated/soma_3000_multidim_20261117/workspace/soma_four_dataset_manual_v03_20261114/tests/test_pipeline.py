import inspect,json,sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,plan,sources,FixedPsi
from assets import Assets
from engine import model_for,forward

class Contracts(unittest.TestCase):
    def test_signal_has_no_stage_argument(self):
        self.assertNotIn('stage_name',inspect.signature(FixedPsi.measure_window).parameters)
    def test_future_data_access_raises(self):
        data=Assets('dsprites')
        with self.assertRaises(ValueError):data.read(1,[0],0)
        with self.assertRaises(ValueError):data.obs_keys(1,[0],0)
    def test_explicit_noise_preserves_rng(self):
        _,entry,_=sources();model,_=model_for(entry,'controlled',10,4,'product_fourier')
        x=torch.zeros(64,64);eps=torch.zeros(64,8);rng=torch.get_rng_state().clone()
        forward(model,x,eps,True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_calibration_roles_have_distinct_sources(self):
        p=plan()
        for ds in ['dsprites','shapes3d','cifar']:
            for seed in p['seeds']:
                d=json.loads((ROOT/f'reports/prepared/{ds}_seed{seed}/roles.json').read_text());seen=set()
                for role,row in d['calibration'].items():
                    values=set(row['ids']);self.assertEqual(len(values),row['rows']);self.assertFalse(seen&values);seen.update(values)
    def test_identity_hash_cache_not_partial(self):
        for ds in ['dsprites','shapes3d','cifar']:
            values=np.load(ROOT/f'data_cache/{ds}_hashes.npy',mmap_mode='r',allow_pickle=False)
            self.assertFalse(np.any(values==np.void(b'\x00'*32)))

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
