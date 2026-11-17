"""Export controlled source IDs and verify role disjointness; no image access."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.b_weight_sensitivity.formal_common import load_config
from experiments.b_weight_sensitivity.formal_runner import args_for
from experiments.b_weight_sensitivity.historical_loader import HistoricalSource
from experiments.b_weight_sensitivity.formal_data import build_disjoint_bundle
from experiments.manual_v03.signal import validate_roles
h=HistoricalSource(ROOT/'legacy/03_psi_xib_unified');c=load_config();records=[]
for seed in c['seeds']:
    args=args_for(h.load('train_xib_unified'),c,'controlled')
    bundle,digest=build_disjoint_bundle(args,seed,h,ROOT/'results/assets')
    validate_roles(bundle.role_manifest['roles'])
    records.append(dict(seed=seed,role_manifest_sha256=digest,
        source_ids={role:len(set(ids)) for role,ids in bundle.role_manifest['roles'].items()},
        intersections_empty=True,policy=bundle.role_manifest['policy']))
(ROOT/'reports/controlled_role_audit.json').write_text(json.dumps(dict(status='passed',records=records,image_roles='deferred'),indent=2)+'\n')
print(json.dumps(records,indent=2))
