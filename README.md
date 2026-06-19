# Image Inverse Problems: Denoising and Deblurring via Regularization and Learning-Based Methods

## Project Goal

This repository is a reproducible computational mini-project on image inverse
problems. The project starts with image denoising and will later extend to
optimization-based methods such as Tikhonov regularization and total variation
regularization. The goal is to build a clear, organized workflow for running
small experiments, saving results, and comparing methods.

## Current MVP

The current minimum viable experiment studies a simple image denoising pipeline:

```text
clean image -> add Gaussian noise -> apply Gaussian filter denoising -> evaluate PSNR and SSIM -> save figures and metrics
```

The experiment uses the built-in grayscale `camera` image from `skimage`. The
image is converted to floating-point values in `[0, 1]`, corrupted with
Gaussian noise, and then denoised using a Gaussian filter baseline. The result
is evaluated with PSNR and SSIM, two common image-quality metrics.

## Environment Setup

Install the required Python packages with:

```powershell
pip install -r requirements.txt
```

If using the project-local virtual environment, install packages through the
virtual environment's Python executable.

## How to Run

Run the MVP denoising experiment from the project root:

```powershell
.\.venv\Scripts\python.exe src\mvp_denoising.py
```

## Reproducing the Experiments

Run the main experiment scripts from the project root:

```powershell
.\.venv\Scripts\python.exe src\mvp_denoising.py
.\.venv\Scripts\python.exe src\noise_sensitivity.py
.\.venv\Scripts\python.exe src\filter_sigma_sensitivity.py
.\.venv\Scripts\python.exe src\tikhonov_lambda_extended.py
.\.venv\Scripts\python.exe src\tv_denoising.py
.\.venv\Scripts\python.exe src\nlm_denoising.py
.\.venv\Scripts\python.exe src\multi_image_nlm_denoising_comparison.py
.\.venv\Scripts\python.exe src\consolidated_denoising_comparison.py
.\.venv\Scripts\python.exe src\tikhonov_deblurring.py
.\.venv\Scripts\python.exe src\wiener_deblurring.py
.\.venv\Scripts\python.exe src\multi_image_deblurring_comparison.py
.\.venv\Scripts\python.exe src\richardson_lucy_deblurring.py
.\.venv\Scripts\python.exe src\degradation_robustness_study.py
.\.venv\Scripts\python.exe src\tiny_cnn_denoising.py
```

Each script saves CSV files in `results/` and figures in `figures/`.

## Outputs

The script generates the following files:

```text
figures/01_mvp_denoising_comparison.png
figures/01_mvp_error_map.png
results/01_mvp_denoising_metrics.csv
data/sample_images/camera_clean.png
data/sample_images/camera_noisy_sigma_0.10.png
data/sample_images/camera_denoised_gaussian.png
```

## First Results

| Method | PSNR | SSIM | Runtime seconds |
|---|---:|---:|---:|
| Noisy image | 20.421019 | 0.296393 | 0.000000 |
| Gaussian filter | 27.171126 | 0.638713 | 0.006478 |

## Interpretation

Gaussian filtering improves both PSNR and SSIM compared with the noisy image,
which shows that it removes a meaningful amount of random noise. However, the
filtered image also has some blurred edges and softened fine details. The error
map shows larger errors around edges and detailed regions. This motivates more
advanced regularization-based methods such as Tikhonov and total variation
denoising.

## Sensitivity Analysis

### Noise Level Sensitivity

Experiment design:

- Fixed method: Gaussian filter
- Fixed `filter_sigma = 1.0`
- Tested `noise_sigma` values: `0.05`, `0.10`, `0.15`, `0.20`
- Output files:
  - `results/02_noise_sensitivity_results.csv`
  - `figures/02_noise_sensitivity_psnr.png`
  - `figures/02_noise_sensitivity_ssim.png`
  - `figures/02_noise_sensitivity_visual_grid.png`

| noise_sigma | Method | PSNR | SSIM |
|---:|---|---:|---:|
| 0.05 | Noisy image | 26.157573 | 0.521413 |
| 0.05 | Gaussian filter | 28.842721 | 0.784190 |
| 0.10 | Noisy image | 20.421019 | 0.296393 |
| 0.10 | Gaussian filter | 27.171126 | 0.638713 |
| 0.15 | Noisy image | 17.222988 | 0.198411 |
| 0.15 | Gaussian filter | 25.318626 | 0.520214 |
| 0.20 | Noisy image | 15.038608 | 0.145189 |
| 0.20 | Gaussian filter | 23.561036 | 0.430733 |

As `noise_sigma` increases, the quality of the noisy image decreases. Gaussian
filtering improves PSNR and SSIM at all tested noise levels, but the final
reconstruction quality still decreases under stronger noise. This shows that
Gaussian filtering is helpful, but it cannot fully recover information lost
under severe noise.

### Filter Sigma Sensitivity

Experiment design:

- Fixed `noise_sigma = 0.10`
- Tested `filter_sigma` values: `0.25`, `0.5`, `1.0`, `1.5`, `2.0`, `3.0`
- Output files:
  - `results/03_filter_sigma_sensitivity_results.csv`
  - `figures/03_filter_sigma_sensitivity_psnr.png`
  - `figures/03_filter_sigma_sensitivity_ssim.png`
  - `figures/03_filter_sigma_sensitivity_visual_grid.png`

| Method | noise_sigma | filter_sigma | PSNR | SSIM |
|---|---:|---:|---:|---:|
| Noisy image | 0.10 | - | 20.421019 | 0.296393 |
| Gaussian filter | 0.10 | 0.25 | 20.432526 | 0.296724 |
| Gaussian filter | 0.10 | 0.5 | 24.021674 | 0.417645 |
| Gaussian filter | 0.10 | 1.0 | 27.171126 | 0.638713 |
| Gaussian filter | 0.10 | 1.5 | 26.408888 | 0.689625 |
| Gaussian filter | 0.10 | 2.0 | 25.421825 | 0.692033 |
| Gaussian filter | 0.10 | 3.0 | 23.930651 | 0.663370 |

Very small `filter_sigma` values do not remove much noise. Larger
`filter_sigma` values remove more noise, but they can also blur edges and fine
details. In this experiment, `filter_sigma = 1.0` gives the highest PSNR, while
`filter_sigma = 2.0` gives the highest SSIM. This shows that the best parameter
choice depends on the evaluation metric, and it motivates later study of
regularization parameters in Tikhonov and total variation methods.

## Tikhonov Regularization

Tikhonov denoising formulates image recovery as an optimization problem:

```text
min_x 0.5 * ||x - b||_2^2 + 0.5 * lambda * ||grad x||_2^2
```

Here, `b` is the noisy image and `x` is the reconstructed image. The first term
keeps the reconstruction close to the noisy observation. The second term
penalizes rapid image variation, encouraging a smoother result. The parameter
`lambda` controls the trade-off between data fidelity and smoothness. The
implementation uses an FFT-based solver with periodic boundary conditions.

### Extended Lambda Sensitivity

Experiment design:

- Fixed `noise_sigma = 0.10`
- Tested `lambda` values: `0.001`, `0.005`, `0.01`, `0.05`, `0.1`, `0.2`,
  `0.5`, `1.0`, `2.0`, `5.0`, `10.0`, `20.0`
- Compared against:
  - Noisy image
  - Gaussian filter baseline with `filter_sigma = 1.0`
- Output files:
  - `results/05_tikhonov_lambda_extended_results.csv`
  - `figures/05_tikhonov_lambda_extended_psnr.png`
  - `figures/05_tikhonov_lambda_extended_ssim.png`
  - `figures/05_tikhonov_lambda_extended_visual_grid.png`

| Method | Parameter | PSNR | SSIM |
|---|---:|---:|---:|
| Noisy image | - | 20.421019 | 0.296393 |
| Gaussian filter | sigma = 1.0 | 27.171126 | 0.638713 |
| Tikhonov | lambda = 0.001 | 20.455274 | 0.297377 |
| Tikhonov | lambda = 0.005 | 20.589752 | 0.301265 |
| Tikhonov | lambda = 0.01 | 20.752387 | 0.306017 |
| Tikhonov | lambda = 0.05 | 21.873123 | 0.340401 |
| Tikhonov | lambda = 0.1 | 22.945047 | 0.376432 |
| Tikhonov | lambda = 0.2 | 24.417737 | 0.433222 |
| Tikhonov | lambda = 0.5 | 26.324756 | 0.538782 |
| Tikhonov | lambda = 1.0 | 26.801759 | 0.622013 |
| Tikhonov | lambda = 2.0 | 26.204448 | 0.678147 |
| Tikhonov | lambda = 5.0 | 24.668893 | 0.689946 |
| Tikhonov | lambda = 10.0 | 23.452498 | 0.666887 |
| Tikhonov | lambda = 20.0 | 22.319030 | 0.635365 |

Small `lambda` values provide insufficient smoothing and leave visible noise.
Moderate `lambda` values improve reconstruction quality, while very large
`lambda` values cause over-smoothing and reduce quality. In this experiment,
the best Tikhonov PSNR occurs at `lambda = 1.0`, and the best Tikhonov SSIM
occurs at `lambda = 5.0`. The Gaussian filter gives higher PSNR than Tikhonov,
but Tikhonov with `lambda = 5.0` gives higher SSIM than the Gaussian filter
baseline. This shows that parameter choice depends on the evaluation metric.

## Total Variation Denoising

Total Variation (TV) denoising is a regularization-based method designed to
reduce noise while preserving edges. A common conceptual model is:

```text
min_x 0.5 * ||x - b||_2^2 + lambda * ||grad x||_1
```

The first term keeps the reconstruction close to the noisy observation. The TV
term penalizes total image variation. Compared with Tikhonov regularization,
TV is more edge-preserving because it penalizes the gradient through an
`L1`-type term rather than a squared gradient term. In scikit-image, the
regularization parameter is called `weight`. Larger `weight` values mean
stronger denoising, but values that are too large can produce over-smoothed or
cartoon-like results. The implementation uses
`skimage.restoration.denoise_tv_chambolle`.

### TV Weight Sensitivity

Experiment design:

- Fixed `noise_sigma = 0.10`
- Tested TV `weight` values: `0.02`, `0.05`, `0.1`, `0.2`, `0.4`, `0.8`
- Compared against:
  - Noisy image
  - Gaussian filter baseline with `filter_sigma = 1.0`
  - Tikhonov `lambda = 1.0`
  - Tikhonov `lambda = 5.0`
- Output files:
  - `results/06_tv_denoising_results.csv`
  - `figures/06_tv_weight_sensitivity_psnr.png`
  - `figures/06_tv_weight_sensitivity_ssim.png`
  - `figures/06_tv_denoising_visual_grid.png`
  - `figures/06_tv_error_maps.png`

| Method | Parameter | PSNR | SSIM |
|---|---:|---:|---:|
| Noisy image | - | 20.421019 | 0.296393 |
| Gaussian filter | sigma = 1.0 | 27.171126 | 0.638713 |
| Tikhonov | lambda = 1.0 | 26.801759 | 0.622013 |
| Tikhonov | lambda = 5.0 | 24.668893 | 0.689946 |
| TV Chambolle | weight = 0.02 | 23.549264 | 0.406829 |
| TV Chambolle | weight = 0.05 | 27.625561 | 0.643286 |
| TV Chambolle | weight = 0.1 | 28.302925 | 0.756463 |
| TV Chambolle | weight = 0.2 | 26.986203 | 0.725116 |
| TV Chambolle | weight = 0.4 | 25.026652 | 0.682579 |
| TV Chambolle | weight = 0.8 | 23.372204 | 0.643194 |

Very small TV `weight` values provide insufficient denoising. Moderate
`weight` values improve both PSNR and SSIM, with the best TV result occurring
at `weight = 0.1` for both metrics. In this experiment, TV with `weight = 0.1`
outperforms both Gaussian filtering and Tikhonov regularization. Large TV
`weight` values cause over-smoothing and can produce cartoon-like or
piecewise-constant artifacts. TV gives the best reconstruction quality in this
denoising experiment, but it is computationally more expensive than Gaussian
filtering and FFT-based Tikhonov denoising.

## Non-local Means Denoising Baseline

Non-local Means (NLM) is a patch-based denoising method. Unlike Gaussian
filtering, it does not rely only on spatial closeness. Instead, it compares
small image patches and averages pixels from similar patches, even when those
patches are not immediately adjacent. This gives NLM a non-local image prior
based on repeated or similar structures.

The key parameter is `h`, which controls filtering strength. Smaller `h`
values are more conservative and may leave noise. Larger `h` values produce
stronger denoising, but they may blur details if the averaging becomes too
aggressive. This experiment compares NLM against Gaussian filtering and TV
denoising on the `camera` image with `noise_sigma = 0.10`.

Output files:

- `results/12_nlm_denoising_results.csv`
- `figures/12_nlm_denoising_psnr.png`
- `figures/12_nlm_denoising_ssim.png`
- `figures/12_nlm_denoising_runtime.png`
- `figures/12_nlm_denoising_visual_grid.png`
- `figures/12_nlm_denoising_error_maps.png`

| Method | Parameter | PSNR | SSIM | Runtime seconds |
|---|---:|---:|---:|---:|
| Noisy image | - | 20.421019 | 0.296393 | 0.000000 |
| Gaussian filter | filter_sigma = 1.0 | 27.171126 | 0.638713 | 0.008483 |
| TV Chambolle | weight = 0.1 | 28.302925 | 0.756463 | 0.279427 |
| NLM denoising | h = 0.08 | 28.815080 | 0.730339 | 0.153047 |
| NLM denoising | h = 0.10 | 28.710546 | 0.742657 | 0.161924 |

NLM improves clearly over the noisy image and also improves over Gaussian
filtering in this experiment. The best NLM PSNR occurs at `h = 0.08`, while
the best NLM SSIM occurs at `h = 0.10`, so the preferred `h` value depends on
the metric. NLM achieves the highest PSNR among the tested methods, but TV
Chambolle remains best by SSIM. The `h` sensitivity is important: `h = 0.04`
is too conservative and leaves substantial residual noise. The trade-off is
runtime: NLM is slower than Gaussian filtering, although it is faster than TV
in this local run. This adds a non-local, patch-similarity prior to the
denoising part of the project, complementing local smoothing and variational
regularization.

## Multi-image Non-local Means Denoising Robustness

Phase 5A tested Non-local Means only on the `camera` image. This Phase 5B
experiment tests whether the NLM conclusion generalizes across multiple image
types. It uses `camera`, `coins`, `moon`, and `page`, and compares the noisy
image, Gaussian filtering, TV Chambolle, NLM `h = 0.08`, and NLM `h = 0.10`.
The two NLM `h` values are fixed from Phase 5A rather than re-tuned for each
image, so this is a fixed-parameter robustness test.

Output files:

- `results/13_multi_image_nlm_denoising_comparison.csv`
- `figures/13_multi_image_nlm_denoising_psnr_by_method.png`
- `figures/13_multi_image_nlm_denoising_ssim_by_method.png`
- `figures/13_multi_image_nlm_denoising_runtime_by_method.png`
- `figures/13_multi_image_nlm_denoising_visual_grid.png`
- `figures/13_multi_image_nlm_denoising_best_method_summary.png`

| Method | Average PSNR | Average SSIM | Average runtime seconds |
|---|---:|---:|---:|
| Noisy image | 20.257347 | 0.315270 | 0.000000 |
| Gaussian filter | 26.224359 | 0.643396 | 0.002819 |
| TV Chambolle | 28.667823 | 0.791593 | 0.166593 |
| NLM h = 0.08 | 29.073556 | 0.765391 | 0.122570 |
| NLM h = 0.10 | 29.284289 | 0.792058 | 0.161621 |

In this robustness comparison, NLM `h = 0.10` has the best average PSNR and
the best average SSIM, although its average SSIM is only marginally higher
than TV Chambolle. NLM improves clearly over Gaussian filtering across all
tested images. However, NLM does not universally outperform TV on every image:
TV Chambolle is best on `moon` for both PSNR and SSIM, and it is also best by
SSIM on `camera`. The ranking depends on both image type and evaluation
metric. Overall, `h = 0.08` often gives stronger PSNR on individual images,
while `h = 0.10` is better on average and stronger for SSIM. Gaussian filtering
remains the fastest restoration method by a large margin. This strengthens the
project by testing a non-local denoising prior beyond a single image.

## Consolidated Denoising Comparison

This section summarizes the main denoising methods under the same experimental
setting:

- Clean image: `skimage.data.camera()`
- `noise_sigma = 0.10`
- Random seed: `42`
- Metrics: PSNR, SSIM, and `runtime_seconds`

The purpose of this table is to compare reconstruction quality and
computational cost across the main methods.

Output files:

- `results/07_consolidated_denoising_comparison.csv`
- `figures/07_consolidated_psnr_comparison.png`
- `figures/07_consolidated_ssim_comparison.png`
- `figures/07_consolidated_runtime_comparison.png`
- `figures/07_consolidated_visual_comparison.png`

| Method | Parameter | PSNR | SSIM | Runtime seconds |
|---|---:|---:|---:|---:|
| Noisy image | - | 20.421019 | 0.296393 | 0.000000 |
| Gaussian filter | filter_sigma = 1.0 | 27.171126 | 0.638713 | 0.004875 |
| Tikhonov | lambda = 1.0 | 26.801759 | 0.622013 | 0.018835 |
| Tikhonov | lambda = 5.0 | 24.668893 | 0.689946 | 0.016929 |
| TV Chambolle | weight = 0.1 | 28.302925 | 0.756463 | 0.266987 |

The noisy image has the lowest PSNR and SSIM. Gaussian filtering is a strong
and very fast baseline. Tikhonov regularization provides a clear
optimization-based formulation, but it does not achieve the best reconstruction
quality in this denoising setting. Tikhonov with `lambda = 5.0` achieves higher
SSIM than Gaussian filtering, but lower PSNR. TV Chambolle with `weight = 0.1`
achieves the highest PSNR and SSIM among the tested methods. TV appears to
better preserve image structure and edges, but it is computationally more
expensive. Overall, TV gives the best reconstruction quality in this
experiment, while Gaussian filtering is the fastest restoration method.

## Multi-image Robustness Study

The goal of this study is to test whether the single-image denoising
conclusions generalize beyond `skimage.data.camera()`. The tested images are
`camera`, `coins`, `moon`, and `page`. The compared methods are the noisy image
baseline, Gaussian filtering with `filter_sigma = 1.0`, Tikhonov denoising with
`lambda = 1.0`, Tikhonov denoising with `lambda = 5.0`, and TV Chambolle
denoising with `weight = 0.1`.

Output files:

- `results/09_multi_image_denoising_comparison.csv`
- `figures/09_multi_image_psnr_by_method.png`
- `figures/09_multi_image_ssim_by_method.png`
- `figures/09_multi_image_runtime_by_method.png`
- `figures/09_multi_image_visual_grid.png`

| Method | Average PSNR | Average SSIM | Average runtime seconds |
|---|---:|---:|---:|
| Noisy image | 20.26 | 0.315 | 0.000 |
| Gaussian filter | 26.22 | 0.643 | 0.003 |
| Tikhonov lambda=1.0 | 26.03 | 0.622 | 0.015 |
| Tikhonov lambda=5.0 | 25.24 | 0.691 | 0.015 |
| TV Chambolle weight=0.1 | 28.67 | 0.792 | 0.171 |

The main finding is that TV Chambolle with `weight = 0.1` achieves the best
PSNR and SSIM on all tested images. Gaussian filtering remains the fastest
restoration method. Tikhonov with `lambda = 5.0` is not uniformly robust across
image types: it works very well on `moon` but performs poorly on `page` in
PSNR. This strengthens the project by addressing the previous limitation that
only one test image was used.

## Tikhonov Deblurring

Deblurring is a more typical image inverse problem than denoising because the
observation is generated by applying a blur operator to the clean image and
adding noise:

```text
b = Kx + noise
```

Here, `x` is the clean image, `K` is a blur operator, and `b` is the blurred
noisy observation. This experiment uses a Gaussian blur kernel with
`blur_sigma = 2.0` and then adds Gaussian noise with `noise_sigma = 0.01`.

The Tikhonov deblurring model is:

```text
min_x 0.5 * ||Kx - b||_2^2 + 0.5 * lambda * ||grad x||_2^2
```

The first term requires that re-blurring the reconstruction should match the
observation. The second term regularizes the reconstruction by penalizing rapid
image variation. The parameter `lambda` controls the trade-off between
aggressive deblurring and stability. Small `lambda` values may amplify noise
and create ringing artifacts, while large `lambda` values are more stable but
may leave the image overly smooth. The implementation uses an FFT-based solver
with periodic boundary conditions.

### Lambda Sensitivity for Deblurring

Experiment design:

- Clean image: `skimage.data.camera()`
- Gaussian blur kernel with `blur_sigma = 2.0`
- Gaussian noise with `noise_sigma = 0.01`
- Tested `lambda` values: `0.0001`, `0.0005`, `0.001`, `0.005`, `0.01`,
  `0.05`, `0.1`
- Output files:
  - `results/08_tikhonov_deblurring_results.csv`
  - `figures/08_tikhonov_deblurring_psnr.png`
  - `figures/08_tikhonov_deblurring_ssim.png`
  - `figures/08_tikhonov_deblurring_visual_grid.png`
  - `figures/08_tikhonov_deblurring_error_maps.png`

| Method | Parameter | PSNR | SSIM |
|---|---:|---:|---:|
| Blurred only | - | 25.568595 | 0.750070 |
| Blurred noisy | - | 25.419450 | 0.699460 |
| Tikhonov deblur | lambda = 0.0001 | 20.044275 | 0.242901 |
| Tikhonov deblur | lambda = 0.0005 | 24.731901 | 0.439101 |
| Tikhonov deblur | lambda = 0.001 | 26.158025 | 0.534194 |
| Tikhonov deblur | lambda = 0.005 | 27.684102 | 0.707010 |
| Tikhonov deblur | lambda = 0.01 | 27.757792 | 0.746953 |
| Tikhonov deblur | lambda = 0.05 | 27.244471 | 0.776191 |
| Tikhonov deblur | lambda = 0.1 | 26.839419 | 0.771727 |

The blurred noisy image is worse than the blurred-only image because noise
further degrades the observation. Very small `lambda` values produce unstable
deblurring and can amplify noise or ringing artifacts. Moderate `lambda`
values improve reconstruction quality. The best PSNR occurs at
`lambda = 0.01`, while the best SSIM occurs at `lambda = 0.05`. Tikhonov
deblurring improves both PSNR and SSIM compared with the blurred noisy
observation. This experiment demonstrates why regularization is important for
deblurring: direct or weakly regularized inversion can be unstable.

## Wiener Deblurring Baseline

Wiener deconvolution is a classical frequency-domain deblurring baseline. It
stabilizes direct inverse filtering by avoiding division by very small blur
frequencies, which would otherwise strongly amplify noise. The key parameter is
`balance`. Smaller `balance` values produce more aggressive deblurring, but
they also increase the risk of noise amplification and ringing artifacts.
Larger `balance` values give a more conservative and stable restoration, but
they may leave more blur in the result.

This experiment uses the same synthetic degradation setting as the Tikhonov
deblurring experiment: `skimage.data.camera()`, Gaussian blur with
`blur_sigma = 2.0`, and Gaussian noise with `noise_sigma = 0.01`. It compares
Wiener deconvolution with selected Tikhonov deblurring results.

Output files:

- `results/10_wiener_deblurring_results.csv`
- `figures/10_wiener_deblurring_psnr.png`
- `figures/10_wiener_deblurring_ssim.png`
- `figures/10_wiener_deblurring_visual_grid.png`
- `figures/10_wiener_deblurring_error_maps.png`

| Method | Parameter | PSNR | SSIM | Runtime seconds |
|---|---:|---:|---:|---:|
| Blurred noisy | - | 25.417081 | 0.698806 | 0.000000 |
| Tikhonov deblur | lambda = 0.01 | 27.752301 | 0.746024 | 0.033137 |
| Tikhonov deblur | lambda = 0.05 | 27.242932 | 0.775857 | 0.027800 |
| Wiener deblur | balance = 0.01 | 27.629829 | 0.735582 | 0.034590 |
| Wiener deblur | balance = 0.03 | 26.743149 | 0.772423 | 0.030849 |

Wiener deblurring improves substantially over the blurred noisy observation
for moderate `balance` values. The best Wiener PSNR occurs at
`balance = 0.01`, while the best Wiener SSIM occurs at `balance = 0.03`.
Wiener comes close to the selected Tikhonov results, but in this experiment it
does not exceed Tikhonov `lambda = 0.01` in PSNR or Tikhonov `lambda = 0.05`
in SSIM. Very small `balance` values are unstable and produce poor metrics,
while very large values are more conservative and leave stronger blur. This
adds a classical frequency-domain baseline to the deblurring part of the
project and reinforces that deblurring quality is parameter-sensitive.

## Multi-image Deblurring Robustness Study

The goal of this study is to test whether conclusions from the single-image
`camera` deblurring experiment generalize across multiple image structures.
The tested images are `camera`, `coins`, `moon`, and `page`. Each image is
degraded using Gaussian blur with `blur_sigma = 2.0` and Gaussian noise with
`noise_sigma = 0.01`.

The compared methods are:

- Blurred noisy baseline
- Tikhonov deblurring with `lambda = 0.01`
- Tikhonov deblurring with `lambda = 0.05`
- Wiener deblurring with `balance = 0.01`
- Wiener deblurring with `balance = 0.03`

This is a fixed-parameter robustness study. The parameters are selected from
the single-image deblurring experiments and are not tuned separately for each
image.

Output files:

- `results/11_multi_image_deblurring_comparison.csv`
- `figures/11_multi_image_deblurring_psnr_by_method.png`
- `figures/11_multi_image_deblurring_ssim_by_method.png`
- `figures/11_multi_image_deblurring_runtime_by_method.png`
- `figures/11_multi_image_deblurring_visual_grid.png`

| Method | Average PSNR | Average SSIM | Average runtime seconds |
|---|---:|---:|---:|
| Blurred noisy | 25.85 | 0.688 | 0.000 |
| Tikhonov lambda = 0.01 | 27.25 | 0.735 | 0.862 |
| Tikhonov lambda = 0.05 | 27.40 | 0.762 | 0.845 |
| Wiener balance = 0.01 | 27.04 | 0.726 | 0.913 |
| Wiener balance = 0.03 | 26.42 | 0.760 | 0.852 |

In this fixed-parameter study, Tikhonov deblurring with `lambda = 0.05` gives
the highest average PSNR and SSIM. The blurred noisy baseline is fastest
because no restoration is applied; among restoration methods, Tikhonov
`lambda = 0.05` is slightly fastest on average in this run. Wiener
deconvolution remains competitive on some images and metrics, especially with
`balance = 0.01` for PSNR on `page` and with `balance = 0.03` for SSIM on
`coins`, but it is not uniformly better than Tikhonov. The results are clearly
image-dependent, with `moon` behaving differently from more textured or
document-like images. This extends the deblurring part of the project from a
single-image comparison to a small robustness study across several standard
test images.

## Richardson-Lucy Deblurring Baseline

Richardson-Lucy deblurring is a classic iterative deconvolution method. It
starts from an image estimate and repeatedly updates it so that its blurred
version better matches the observed blurred image. The key parameter is
`num_iter`, the number of update steps. Fewer iterations may under-restore the
image, while too many iterations can sharpen noise or create artifacts.

This experiment compares Richardson-Lucy against fixed Tikhonov and Wiener
deblurring baselines on the `camera` image using the same synthetic
degradation: Gaussian blur with `blur_sigma = 2.0` and Gaussian noise with
`noise_sigma = 0.01`.

Output files:

- `results/14_richardson_lucy_deblurring_results.csv`
- `figures/14_richardson_lucy_deblurring_psnr.png`
- `figures/14_richardson_lucy_deblurring_ssim.png`
- `figures/14_richardson_lucy_deblurring_runtime.png`
- `figures/14_richardson_lucy_deblurring_visual_grid.png`
- `figures/14_richardson_lucy_deblurring_error_maps.png`

| Method | Parameter | PSNR | SSIM | Runtime seconds |
|---|---:|---:|---:|---:|
| Blurred noisy | - | 25.419450 | 0.699460 | 0.000000 |
| Tikhonov deblur | lambda = 0.01 | 27.757792 | 0.746953 | 0.019798 |
| Tikhonov deblur | lambda = 0.05 | 27.244471 | 0.776191 | 0.019327 |
| Wiener deblur | balance = 0.01 | 27.636507 | 0.736645 | 0.019391 |
| Wiener deblur | balance = 0.03 | 26.745217 | 0.772891 | 0.019019 |
| Richardson-Lucy | num_iter = 5 | 23.738545 | 0.738875 | 0.093209 |
| Richardson-Lucy | num_iter = 10 | 22.563644 | 0.745706 | 0.177121 |
| Richardson-Lucy | num_iter = 20 | 21.544518 | 0.741083 | 0.353720 |
| Richardson-Lucy | num_iter = 30 | 20.930593 | 0.731952 | 0.533768 |
| Richardson-Lucy | num_iter = 50 | 20.040791 | 0.712166 | 0.914123 |

Richardson-Lucy does not improve over the blurred noisy observation in PSNR,
but it does improve over the blurred noisy observation in SSIM for the tested
iteration counts. In this run, the best Richardson-Lucy PSNR occurs at
`num_iter = 5`, while the best Richardson-Lucy SSIM occurs at `num_iter = 10`.
The tested Richardson-Lucy settings do not outperform the best Tikhonov or
best Wiener baselines in this synthetic Gaussian blur plus noise experiment.
Too many iterations reduce both PSNR and SSIM, and runtime increases with
iteration count. This makes Richardson-Lucy a useful iterative deconvolution
baseline, but in this setting it is less stable than the selected Tikhonov and
Wiener methods and is sensitive to the number of iterations.

## Degradation Robustness Study

This phase tests whether earlier conclusions remain stable under different
degradation strengths. The denoising study uses `noise_sigma = 0.05, 0.10,
0.20`. The deblurring study uses `blur_sigma = 1.0, 2.0, 3.0` with fixed
`noise_sigma = 0.01`. Method parameters are fixed from earlier phases rather
than re-tuned for each degradation level, so this is a robustness study rather
than a per-condition optimization.

Output files:

- `results/15_degradation_robustness_denoising.csv`
- `results/15_degradation_robustness_deblurring.csv`
- `figures/15_degradation_robustness_denoising_psnr.png`
- `figures/15_degradation_robustness_denoising_ssim.png`
- `figures/15_degradation_robustness_denoising_runtime.png`
- `figures/15_degradation_robustness_deblurring_psnr.png`
- `figures/15_degradation_robustness_deblurring_ssim.png`
- `figures/15_degradation_robustness_deblurring_runtime.png`
- `figures/15_degradation_robustness_summary.png`

### Denoising Robustness Summary

| noise_sigma | Best PSNR method | Best PSNR | Best SSIM method | Best SSIM |
|---:|---|---:|---|---:|
| 0.05 | TV Chambolle | 29.256535 | TV Chambolle | 0.794296 |
| 0.10 | NLM h = 0.10 | 28.687700 | TV Chambolle | 0.756777 |
| 0.20 | TV Chambolle | 23.605989 | TV Chambolle | 0.473651 |

At mild noise (`noise_sigma = 0.05`), TV Chambolle is best by both PSNR and
SSIM. At moderate noise (`noise_sigma = 0.10`), NLM `h = 0.10` is best by
PSNR, while TV Chambolle remains best by SSIM. At stronger noise
(`noise_sigma = 0.20`), TV Chambolle is again best by both metrics. NLM is
competitive at mild and moderate noise, but with fixed `h = 0.10` it is not
robust at `noise_sigma = 0.20`. Gaussian filtering remains the fastest
restoration method. Overall, the method ranking changes with noise strength.

### Deblurring Robustness Summary

| blur_sigma | Best PSNR method | Best PSNR | Best SSIM method | Best SSIM |
|---:|---|---:|---|---:|
| 1.0 | Tikhonov lambda = 0.01 | 30.554318 | Tikhonov lambda = 0.05 | 0.855700 |
| 2.0 | Tikhonov lambda = 0.01 | 27.749095 | Tikhonov lambda = 0.05 | 0.775812 |
| 3.0 | Tikhonov lambda = 0.01 | 26.026181 | Tikhonov lambda = 0.05 | 0.717125 |

For deblurring, the ranking is more stable in this fixed grid. Tikhonov
`lambda = 0.01` is best by PSNR at all tested blur levels, while Tikhonov
`lambda = 0.05` is best by SSIM at all tested blur levels. Wiener deblurring
remains competitive, especially at stronger blur, but does not exceed the best
Tikhonov setting. Richardson-Lucy improves SSIM over the blurred noisy baseline
but does not improve PSNR and does not outperform Tikhonov or Wiener.

## Tiny CNN Denoising Baseline

Phase 8 adds a lightweight learning-based denoising baseline. A small
convolutional neural network is trained on noisy and clean image patches and
then evaluated on the `camera` test image with `noise_sigma = 0.10`. This is a
compact educational baseline with limited training data, not a
state-of-the-art deep learning method.

Training took `3.557896` seconds. Both training and validation loss decreased
steadily over eight epochs:

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 0.034897 | 0.022518 |
| 2 | 0.010471 | 0.007086 |
| 3 | 0.005626 | 0.005000 |
| 4 | 0.004589 | 0.004741 |
| 5 | 0.004336 | 0.004493 |
| 6 | 0.004039 | 0.004035 |
| 7 | 0.003892 | 0.004040 |
| 8 | 0.003826 | 0.003810 |

| Method | Parameter | PSNR | SSIM | Runtime seconds |
|---|---|---:|---:|---:|
| Noisy image | - | 20.421019 | 0.296393 | 0.000000 |
| Gaussian filter | filter_sigma = 1.0 | 27.171126 | 0.638712 | 0.005656 |
| TV Chambolle | weight = 0.1 | 28.302925 | 0.756461 | 0.218245 |
| NLM denoising | h = 0.1 | 28.710546 | 0.742656 | 0.178762 |
| Tiny CNN | trained model | 26.479463 | 0.627989 | 0.027087 |

The Tiny CNN clearly improves over the noisy image, but it does not outperform
Gaussian filtering, TV Chambolle, or NLM in PSNR or SSIM on this test. NLM
`h = 0.10` gives the best PSNR, while TV Chambolle gives the best SSIM. Tiny
CNN inference is faster than TV and NLM but slower than Gaussian filtering.
The result is useful because it shows that a small CNN trained on limited data
does not automatically outperform well-chosen classical image priors.

Output files:

- `results/16_tiny_cnn_denoising_results.csv`
- `results/16_tiny_cnn_training_history.csv`
- `results/16_tiny_cnn_metadata.txt`
- `models/16_tiny_cnn_denoising.pt`
- `figures/16_tiny_cnn_training_loss.png`
- `figures/16_tiny_cnn_denoising_psnr.png`
- `figures/16_tiny_cnn_denoising_ssim.png`
- `figures/16_tiny_cnn_denoising_runtime.png`
- `figures/16_tiny_cnn_denoising_visual_grid.png`
- `figures/16_tiny_cnn_denoising_error_maps.png`

## Next Steps

- Optionally implement TV deblurring or compare additional deblurring baselines.
- Prepare a short technical report.
- Polish the repository for GitHub presentation.
- Optionally add more test images to check whether the conclusions generalize.
