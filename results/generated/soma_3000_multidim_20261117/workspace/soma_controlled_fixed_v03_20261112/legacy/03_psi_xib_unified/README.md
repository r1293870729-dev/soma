# Unified Xi-B runner

This directory is a unified loss-only Xi-B/Psi runner for four datasets:

- `controlled`
- `dsprites`
- `shapes3d`
- `cifar`

The previous code had two separate entries:

- controlled manifold / I1-I4: `run_pw_signal_experiment.py`
- dSprites / 3D Shapes / CIFAR: `train_dataset_generalization.py`

This directory merges the training loop and keeps only Xi-B. Xi-A and Xi-C are
not computed here.

## Xi-B definition in this code

For each batch, the model produces a reconstruction residual:

```text
epsilon = x_hat - x
```

The residual is projected into a fixed feature space:

```text
features = h(epsilon)
```

For each stage, the code maintains a baseline mean and covariance:

```text
mu_0, Sigma_0
```

The current residual feature is whitened by the current readout baseline:

```text
eta = Sigma_0^{-1/2} (features - mu_0)
```

A rolling window of `W` whitened residual samples is converted into a trace-normalized covariance:

```text
C = cov(window(eta))
C_tilde = d_epsilon * C / tr(C)
```

The Marchenko-Pastur edge is:

```text
lambda_edge = (1 + sqrt(d_epsilon / (W - 1)))^2
```

Xi-B is the total super-edge spectral mass:

```text
Xi-B = sum_i [lambda_i(C_tilde) / lambda_edge - 1]_+
```

## Data split

Every dataset has two stage lists:

- `train_stages`: used by VAE training and Xi-B residual diagnostics.
- `probe_stages`: used only by online linear probes.

Default split size:

- `Dtrain = 4096` samples per stage.
- `Dprobe = 4096` samples per stage.
- probe classifier internal split: 70 percent train, 30 percent test.

Controlled data uses independent seeds for `Dtrain` and `Dprobe`.

dSprites and 3D Shapes use independent sampled factor streams for `Dtrain` and
`Dprobe`. Some low-complexity stages have fewer possible unique factor
combinations than 4096, so the split is an independent diagnostic stream rather
than a mathematical guarantee that every underlying image index is unique.

CIFAR/CIFAR-C uses fixed non-overlapping subsets for `Dtrain` and `Dprobe` in
each stage.

## Default latent dimensions

- `controlled`: 4
- `dsprites`: 8, 16, 32
- `shapes3d`: 16, 32, 64
- `cifar`: 16, 32, 64

## Main command

Run one dataset:

```powershell
$env:PYTHONNOUSERSITE="1"
python train_xib_unified.py `
  --dataset controlled `
  --seeds 0 1 2 `
  --output-dir runs\xib_unified_controlled `
  --device cpu `
  --quiet-console
```

Run dSprites:

```powershell
$env:PYTHONNOUSERSITE="1"
python train_xib_unified.py `
  --dataset dsprites `
  --seeds 0 `
  --output-dir runs\xib_unified_dsprites `
  --device cpu `
  --quiet-console
```

Run 3D Shapes:

```powershell
$env:PYTHONNOUSERSITE="1"
python train_xib_unified.py `
  --dataset shapes3d `
  --seeds 0 `
  --output-dir runs\xib_unified_shapes3d `
  --device cpu `
  --quiet-console
```

Run CIFAR/CIFAR-C:

```powershell
$env:PYTHONNOUSERSITE="1"
python train_xib_unified.py `
  --dataset cifar `
  --seeds 0 `
  --output-dir runs\xib_unified_cifar `
  --device cpu `
  --quiet-console
```

## Important options

- `--train-samples-per-stage 4096`: samples used by VAE/Xi-B per stage.
- `--probe-samples-per-stage 4096`: independent samples used by online probe per stage.
- `--probe-train-fraction 0.7`: probe internal train/test split.
- `--stage-steps 1000 1000 1000`: training steps for S1/S2/S3.
- `--x-window 8`: sets `W = x_window * batch_size`.
- `--batch-size 64`: default batch size for image datasets.
- `--residual-feature-dim 256`: residual projection dimension.
- `--baseline-warmup-batches 20`: batches before a stage baseline is ready.
- `--alpha 0.1`: covariance shrinkage.
- `--beta-kl 0.001`: VAE KL weight.

For controlled runs matching the recent setting, use:

```powershell
python train_xib_unified.py `
  --dataset controlled `
  --seeds 0 1 2 `
  --latent-dims 4 `
  --stage-steps 1000 1000 1000 `
  --batch-size 128 `
  --train-samples-per-stage 4096 `
  --probe-samples-per-stage 4096 `
  --controlled-hidden-dim 32 `
  --controlled-hidden-layers 1 `
  --intrinsic-dim 8 `
  --split-offset 0.8 `
  --local-probability 1.0 `
  --output-dir runs\xib_unified_controlled_z4_seed3 `
  --device cpu `
  --quiet-console
```

## Outputs

Each run directory contains:

- `train_history.csv`: reconstruction, KL, Xi-B, spike rank, window count.
- `probe_history.csv`: online probe accuracy/loss for each task.
- `checkpoint.pt`: trained model checkpoint.
- `summary.json`: final run summary.
- `DONE`: marker for `--resume`.

The output root also contains:

- `{dataset}_summary.csv`
- `{dataset}_all_results.json`
