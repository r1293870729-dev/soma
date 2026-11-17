"""Prepare/inference-check official reader only; no image training or dataset access."""
from pathlib import Path
import json,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.manual_v03.readout import FrozenReadout
from experiments.b_weight_sensitivity.formal_runner import tensor_digest

torch.set_num_threads(1)
spec=json.loads((ROOT/'configs/image_readout_prepared.json').read_text())
spec['weights_path']=str(ROOT/spec['weights_path'])
torch.manual_seed(4803);rng=torch.get_rng_state().clone()
reader=FrozenReadout('cifar',spec,torch.device('cpu'))
assert torch.equal(torch.get_rng_state(),rng)
before=tensor_digest(reader.model)
g=torch.Generator().manual_seed(5803)
x=torch.rand(4,3,32,32,generator=g);prediction=torch.rand(4,3,32,32,generator=g)
reader.residual(x,prediction)
start=time.perf_counter();repetitions=3
for _ in range(repetitions):value=reader.residual(x,prediction)
elapsed=time.perf_counter()-start
assert value.shape==(4,512) and all(not p.requires_grad for p in reader.model.parameters())
assert before==tensor_digest(reader.model)
gray=reader.residual(x[:1,:1],prediction[:1,:1]);assert gray.shape==(1,512)
result=dict(status='passed_cpu_inference_fixture',images='synthetic random tensors only',image_training_started=False,
            output_shape=list(value.shape),frozen_parameters=sum(p.numel() for p in reader.model.parameters()),
            parameters_unchanged=True,constructor_preserves_rng=True,gray_replication_checked=True,
            cpu_threads=1,repetitions=repetitions,batch_size=4,
            wall_seconds=elapsed,seconds_per_residual_pair=elapsed/(repetitions*4),
            note='readout on both observation and reconstruction; includes transforms; not a full image training cost estimate',
            specification=spec)
(ROOT/'reports/readout_benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='specification'},ensure_ascii=False,indent=2))
