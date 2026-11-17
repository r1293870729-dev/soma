"""One update on a cloned fork; verify stable continuation against an independent stored prefix."""
import sys,json,types
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from paths import PLAN,IMAGE,PAIR,read,write,verify
sys.path.insert(0,str(PAIR/'src'))
from observer import SchemeBObserver
from common import sources,tensor_digest,seed_for,hash_array
from engine import model_for,forward
from assets import Assets
def main(ds):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);verify();dim=PLAN['image_dimensions'][ds][0];seed=10;name=f'{ds}_z{dim}_seed{seed}';folder=ROOT/'results'/PLAN['run_id']/name;state=torch.load(folder/'checkpoint_fork.pt',map_location='cpu',weights_only=True)
    source,entry,models=sources();model,likelihood=model_for(entry,ds,seed,dim,ds);model.load_state_dict(state['model']);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=PLAN['learning_rate'],weight_decay=PLAN['weight_decay']);optimizer.load_state_dict(state['optimizer']);ages=[int(v['step']) for v in optimizer.state.values()];assert ages and set(ages)=={1000}
    original={int(r['step']):r for r in read(IMAGE/'results/generated/image_dims_refs_001'/name/'changing/trajectory.csv') if int(r['step']) in [1001,1002]};assert tensor_digest(model)==original[1001]['pre_model_sha256']
    ids=np.load(folder/'prepared/source_ids.npz',allow_pickle=False)['train_0'];idx=np.load(IMAGE/'results/generated/image_dims_refs_001'/name/'batch_indices.npy',allow_pickle=False)[1000];assert hash_array(ids[idx])==original[1001]['input_ID_sha256'];epsilon=torch.from_numpy(np.random.default_rng(seed_for('training-epsilon',ds,seed)).normal(size=(1001,64,dim)).astype(np.float32)[1000]);assets=Assets(ds)
    # S1 is still active at1001 in the independent old4000-step prefix, so it is
    # an external expected result for replaying a known input on the new fork.
    # The actual paired stable/changing branches use the same new natural-stage
    # index stream; that stream is not the old S1 continuation's permutation.
    x=assets.read(0,ids[idx],0);logits,mu,lv=forward(model,x,epsilon);rec,mse,hat=entry.compute_reconstruction_terms(types.SimpleNamespace(dataset=ds),logits,x,likelihood);kl=models.kl_standard_normal(mu,lv);objective=rec+PLAN['beta_kl']*kl
    for key,v in [('reconstruction',rec),('pixel_mse',mse),('kl',kl),('objective',objective)]:assert float(v.detach())==float(original[1001][key])
    optimizer.zero_grad(set_to_none=True);objective.backward();optimizer.step();assert tensor_digest(model)==original[1002]['pre_model_sha256'];assert {int(v['step']) for v in optimizer.state.values()}=={1001}
    write(ROOT/'reports'/f'fork_qualification_{ds}.json',dict(status='pass',dataset=ds,dimension=dim,seed=seed,cloned_actual_fork_age=1000,known_S1_batch_losses_equal_external_prefix=True,after_one_matched_input_update_model_equal_external_prefix=True,optimizer_age_after_clone_update=1001,diagnostic_clone_updates=1,live_training_untouched=True));print(ds,'fork qualification PASS',flush=True)
if __name__=='__main__':main(sys.argv[1])
