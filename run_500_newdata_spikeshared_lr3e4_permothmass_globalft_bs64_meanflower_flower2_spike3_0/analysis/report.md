# 500-Epoch Spike-Shared Latent Alignment Report

## Run Summary

This report summarizes the new 500-epoch run on the `fatmoth_new` dataset:

- Run directory: `run_500_newdata_spikeshared_lr3e4_permothmass_globalft_bs64_meanflower_flower2_spike3_0`
- Dataset:
  - High mass: `/hpc/group/tarokhlab/hy190/data/fatmoth_new/fatmoth_HIGH.pickle`
  - Low mass: `/hpc/group/tarokhlab/hy190/data/fatmoth_new/fatmoth_LOW.pickle`
- Training:
  - Epochs: `500`
  - Batch size: `64`
  - Learning rate: `3e-4`
  - Eval interval: every `5` epochs
- Model settings:
  - Flower latent dimension: `2`
  - Spike latent dimension: `3`
  - Flower reconstruction mode: `mean`
  - Flower decoder latent source: `spike_shared`
  - FT target axes: `Fx`, `Ty`, `Tz`
  - FT normalization: global
  - FT predictor mode: per-moth and per-mass linear heads

Primary figures are embedded in the relevant sections below.

## Architecture

The model used here is not the original two-independent-autoencoder alignment architecture. In this run, the flower decoder is driven directly from the shared dimensions of the spike latent state.

Inputs:

- Spike train: shape `[batch, 400, 10]`, one time series for the 10 muscles.
- Flower signal: shape `[batch, 400, 2]`, containing flower position and velocity.
- Mass label: derived from the dataloader batch, with `0` for low mass and `1` for high mass.
- Moth ID: categorical moth identity, embedded into a 1D condition vector.
- FT target: shape `[batch, 3]`, using only `Fx`, `Ty`, and `Tz`.

Spike encoder:

- The spike encoder receives the spike train plus the moth-ID condition.
- It outputs a 3D spike latent:
  - `z_spike[0:2]`: the 2D shared latent.
  - `z_spike[2]`: the mass logit / mass indicator.
- The mass dimension is trained using binary cross entropy against the low/high mass batch label.

Flower reconstruction path:

- Because `flower_decoder_latent_source = spike_shared`, the flower encoder is not used for the flower reconstruction path in this run.
- The flower decoder receives only the shared spike latent `z_spike[0:2]`.
- Since `flower_recon_mode = mean`, the target is the mean flower position and mean flower velocity over the 400 time steps, not the full flower sequence.
- Output shape: `[batch, 2]`, interpreted as mean position and mean velocity.

Spike reconstruction path:

- The spike decoder receives the full 3D spike latent plus the moth-ID condition.
- It reconstructs the full spike train.
- Output shape: `[batch, 400, 10]`.

Force/torque prediction path:

- The FT predictor is linear.
- The model uses separate linear heads for each moth and mass condition.
- The selected head maps the 2D shared latent to the 3 FT axes:
  - Input: `z_spike[0:2]`
  - Output: `[Fx, Ty, Tz]`
- In this mode, the FT head does not use the moth embedding as a concatenated input; the moth ID selects which moth-specific head is used.

Cross reconstruction / alignment interpretation:

- In this `spike_shared` mode, cross-flower reconstruction is effectively the same as flower reconstruction because both decode from the spike shared latent.
- Cross-spike reconstruction is also effectively the same as spike reconstruction because the spike latent is already the source latent.
- The most meaningful alignment-style test for this run is therefore the shared-latent importance analysis and the latent sweep, rather than a separate flower-encoder-to-spike-decoder transfer test.

## Main Results

Overall test metrics:

| Metric | Value |
|---|---:|
| Test samples | 656 |
| Mass accuracy | 0.994 |
| Mean mass probability | 0.500 |
| Spike reconstruction R2 | 0.257 |
| Cross-spike reconstruction R2 | 0.257 |
| Flower mean reconstruction R2 | 0.765 |
| Cross-flower mean reconstruction R2 | 0.765 |
| Flower sequence-style R2 | 0.748 |
| FT prediction R2 | 0.635 |

FT R2 by channel:

| Channel | R2 |
|---|---:|
| Fx | 0.778 |
| Ty | 0.471 |
| Tz | 0.645 |

Flower mean R2 by channel:

| Channel | R2 |
|---|---:|
| Mean position | 0.239 |
| Mean velocity | -0.051 |

The overall flower mean R2 is high, but the per-channel view shows the important caveat: mean velocity is not being predicted well. The aggregate flower score is therefore mostly not evidence that both flower channels are learned equally well.

Mass prediction:

![Mass prediction histogram](mass_prediction_hist.png)

Spike reconstruction examples:

![Spike reconstruction examples](spike_reconstruction_examples.png)

Cross-spike reconstruction examples:

![Cross-spike reconstruction examples](cross_spike_reconstruction_examples.png)

Flower mean prediction:

![Flower mean prediction scatter](flower_mean_prediction_scatter.png)

Cross-flower mean prediction:

![Cross-flower mean prediction scatter](cross_flower_mean_prediction_scatter.png)

FT prediction:

![FT prediction scatter](ft_prediction_scatter.png)

## Low vs High Mass

| Split | Samples | Mass acc. | Mean mass prob. | Spike R2 | Flower mean R2 | FT R2 |
|---|---:|---:|---:|---:|---:|---:|
| Low mass | 328 | 0.994 | 0.010 | 0.242 | 0.784 | 0.691 |
| High mass | 328 | 0.994 | 0.990 | 0.268 | 0.745 | 0.559 |

FT R2 by mass and channel:

| Split | Fx | Ty | Tz |
|---|---:|---:|---:|
| Low mass | 0.752 | 0.593 | 0.741 |
| High mass | 0.795 | 0.273 | 0.473 |

Flower mean R2 by mass and channel:

| Split | Mean position | Mean velocity |
|---|---:|---:|
| Low mass | 0.328 | -0.063 |
| High mass | 0.157 | -0.042 |

Mass prediction is very strong and well separated: low-mass examples have mean predicted mass probability near `0.01`, and high-mass examples have mean predicted mass probability near `0.99`. FT prediction is stronger on the low-mass split than the high-mass split, especially for `Ty` and `Tz`.

Low-mass FT prediction:

![Low-mass FT prediction scatter](ft_prediction_scatter_low.png)

High-mass FT prediction:

![High-mass FT prediction scatter](ft_prediction_scatter_high.png)

## By-Moth Results

| Moth | Samples | Mass acc. | Spike R2 | Flower mean R2 | FT R2 |
|---|---:|---:|---:|---:|---:|
| 0 | 98 | 1.000 | 0.268 | 0.751 | 0.490 |
| 1 | 87 | 0.989 | 0.175 | 0.768 | 0.421 |
| 2 | 105 | 0.990 | 0.233 | 0.808 | 0.446 |
| 3 | 79 | 0.975 | 0.347 | 0.795 | 0.790 |
| 4 | 88 | 1.000 | 0.280 | 0.607 | 0.723 |
| 5 | 106 | 1.000 | 0.256 | 0.725 | 0.287 |
| 6 | 93 | 1.000 | 0.257 | 0.803 | 0.694 |

Moth 3 and moth 4 have the strongest FT prediction among the per-moth summaries. Moth 5 has the weakest FT R2 despite strong mass prediction and reasonable spike reconstruction, so its force/torque dynamics may not be captured as well by the current 2D shared latent and linear head.

By-moth metric summary:

![Metrics by moth](metrics_by_moth.png)

Spike reconstruction examples by moth:

![Moth 0 spike reconstruction examples](spike_recon_by_moth/moth_0_spike_examples.png)

![Moth 1 spike reconstruction examples](spike_recon_by_moth/moth_1_spike_examples.png)

![Moth 2 spike reconstruction examples](spike_recon_by_moth/moth_2_spike_examples.png)

![Moth 3 spike reconstruction examples](spike_recon_by_moth/moth_3_spike_examples.png)

![Moth 4 spike reconstruction examples](spike_recon_by_moth/moth_4_spike_examples.png)

![Moth 5 spike reconstruction examples](spike_recon_by_moth/moth_5_spike_examples.png)

![Moth 6 spike reconstruction examples](spike_recon_by_moth/moth_6_spike_examples.png)

## Shared Latent Importance

The shared latent has 2 dimensions inside a 3D spike latent. To test whether these dimensions matter for spike reconstruction, each shared dimension was zeroed and the spike reconstruction R2 drop was measured.

Overall:

| Perturbation | Spike R2 | Delta R2 |
|---|---:|---:|
| Baseline | 0.257 | 0.000 |
| Zero both shared dims | 0.128 | -0.129 |
| Zero shared dim 0 | 0.180 | -0.077 |
| Zero shared dim 1 | 0.223 | -0.034 |

By mass:

| Split | Baseline R2 | Zero both shared dims R2 | Delta R2 |
|---|---:|---:|---:|
| Low mass | 0.242 | 0.095 | -0.148 |
| High mass | 0.268 | 0.154 | -0.114 |

Dimension-specific importance:

| Split | Dim 0 delta R2 | Dim 1 delta R2 |
|---|---:|---:|
| Overall | -0.077 | -0.034 |
| Low mass | -0.088 | -0.008 |
| High mass | -0.067 | -0.056 |

The shared dimensions are being used by the spike decoder: zeroing both shared dimensions reduces spike R2 from `0.257` to `0.128`. Dimension 0 is more important overall, while high-mass examples use the two shared dimensions more evenly than low-mass examples.

Overall shared latent importance:

![Overall shared latent importance](overall_shared_spike_latent_importance.png)

Overall shared latent importance by muscle:

![Overall shared latent importance by muscle](overall_shared_spike_latent_importance_by_muscle.png)

Low-mass shared latent importance:

![Low-mass shared latent importance](low_shared_spike_latent_importance.png)

Low-mass shared latent importance by muscle:

![Low-mass shared latent importance by muscle](low_shared_spike_latent_importance_by_muscle.png)

High-mass shared latent importance:

![High-mass shared latent importance](high_shared_spike_latent_importance.png)

High-mass shared latent importance by muscle:

![High-mass shared latent importance by muscle](high_shared_spike_latent_importance_by_muscle.png)

## Latent Sweep

The latent sweep was regenerated with a 1st to 99th percentile range and 11 sweep steps. The sweep changes the shared latent dimensions and visualizes how spike reconstruction, mean flower prediction, and FT prediction change by moth.

Moth 0:

![Moth 0 latent sweep](shared_latent_sweep_by_moth/moth_0_shared_latent_sweep.gif)

Moth 1:

![Moth 1 latent sweep](shared_latent_sweep_by_moth/moth_1_shared_latent_sweep.gif)

Moth 2:

![Moth 2 latent sweep](shared_latent_sweep_by_moth/moth_2_shared_latent_sweep.gif)

Moth 3:

![Moth 3 latent sweep](shared_latent_sweep_by_moth/moth_3_shared_latent_sweep.gif)

Moth 4:

![Moth 4 latent sweep](shared_latent_sweep_by_moth/moth_4_shared_latent_sweep.gif)

Moth 5:

![Moth 5 latent sweep](shared_latent_sweep_by_moth/moth_5_shared_latent_sweep.gif)

Moth 6:

![Moth 6 latent sweep](shared_latent_sweep_by_moth/moth_6_shared_latent_sweep.gif)

The sweep confirms that the shared latent affects the reconstructed spike train and downstream predictions. Because the range is based on the empirical 1st to 99th percentile of encoded shared latents, it is wider than the previous sweep but still tied to the model's observed latent distribution.

## Training Behavior

Final training loss at epoch 500:

| Loss | Value |
|---|---:|
| Total | 1.069 |
| FT | 0.296 |
| Flower reconstruction | 0.612 |
| Spike reconstruction | 0.024 |
| Spike count | 1.348 |

Final validation loss at the last scheduled validation point:

| Loss | Value |
|---|---:|
| Total | 2.761 |
| FT | 0.392 |
| Flower reconstruction | 2.149 |
| Spike reconstruction | 0.024 |
| Spike count | 1.349 |

Best scheduled validation losses:

| Loss | Best value | Approx. epoch |
|---|---:|---:|
| Total | 1.976 | 140 |
| FT | 0.318 | 240 |
| Flower reconstruction | 1.419 | 25 |
| Spike reconstruction | 0.023 | 485 |
| Spike count | 1.339 | 485 |

The training curve suggests overfitting, especially in the flower reconstruction objective: training reconstruction loss kept improving, while validation reconstruction loss worsened by the end. The final automatic evaluation metrics are still useful, but the best validation behavior likely occurred substantially earlier than epoch 500.

Training and validation loss curves:

![Training and validation loss curves](../training_validation_loss_curve.png)

## Interpretation

The model succeeds at three things:

- It learns the mass indicator very reliably from the spike latent, reaching `99.4%` test accuracy.
- The shared spike latent is genuinely used for spike reconstruction; zeroing the shared dimensions causes a substantial R2 drop.
- The per-moth/per-mass linear FT heads produce useful FT prediction, with overall R2 `0.635` and especially strong `Fx` prediction.

The main weakness is the flower reconstruction target:

- The aggregate flower mean R2 is `0.765`, but the channel-wise metrics show only modest mean-position prediction and poor mean-velocity prediction.
- Mean velocity has negative R2 overall and in both low/high mass splits.
- Since this run decodes flower state directly from the spike shared latent, the result suggests that the current 2D shared latent captures some flower-related structure, but not enough velocity information.

Overall, this run is a useful checkpoint for the spike-shared architecture. It shows strong mass separation, meaningful spike-latent usage, and reasonable FT prediction, but it does not yet solve the flower velocity reconstruction problem.
