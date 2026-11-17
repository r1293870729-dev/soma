"""Explicit frozen readout. Nonlinear readouts act on observation/reconstruction separately."""
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F


class FrozenReadout:
    def __init__(self, dataset, specification, device):
        self.device, self.specification = device, dict(specification)
        self.model = None
        kind = specification.get('kind')
        if dataset == 'controlled':
            if kind != 'identity':
                raise ValueError('Manual controlled main readout must be identity')
        elif kind == 'resnet18':
            # No implicit download, random-weight fallback, or unspecified transform.
            required = ['weights_path', 'sha256', 'pretraining', 'supervision', 'license',
                        'resize', 'mean', 'std', 'output_layer', 'dimension']
            if any(specification.get(k) is None for k in required):
                raise ValueError('Image readout metadata requires author input')
            if specification['output_layer'] != 'avgpool' or specification['dimension'] != 512:
                raise ValueError('This adapter implements ResNet18 avgpool/512 only')
            path = Path(specification['weights_path']).expanduser()
            if hashlib.sha256(path.read_bytes()).hexdigest() != specification['sha256']:
                raise ValueError('Readout weight SHA256 mismatch')
            from torchvision.models import resnet18, ResNet18_Weights
            # Model construction must not advance the online training RNG.
            with torch.random.fork_rng(devices=[]):
                self.model = resnet18(weights=None)
            state = torch.load(path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state, strict=True)
            self.model.fc = torch.nn.Identity()
            self.model.to(device).eval().requires_grad_(False)
            if specification.get('transform_id') != 'torchvision_0.21_IMAGENET1K_V1':
                raise ValueError('Official pretrained reader requires its frozen official transform')
            if specification.get('resize') != [256] or specification.get('crop_size') != [224]:
                raise ValueError('Expected official resize-shorter-side 256 / center-crop 224')
            if specification['mean'] != [.485, .456, .406] or specification['std'] != [.229, .224, .225]:
                raise ValueError('Official pretrained normalization differs')
            self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True)
        else:
            raise ValueError('No frozen visual readout approved; refusing to substitute pixel projection')

    @torch.no_grad()
    def __call__(self, images):
        if self.model is None:
            return images.detach().flatten(start_dim=1)
        x = images.detach()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[1] != 3:
            raise ValueError('Expected one or three image channels')
        return self.model(self.preprocess(x))

    @torch.no_grad()
    def residual(self, observations, reconstructed_means):
        # Equation 3.1, not h(observation - reconstruction).
        value = self(observations) - self(reconstructed_means)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite frozen readout residual')
        return value.detach().cpu().numpy()
