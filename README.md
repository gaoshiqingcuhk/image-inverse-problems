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
.\.venv\Scripts\python.exe src\consolidated_denoising_comparison.py
.\.venv\Scripts\python.exe src\tikhonov_deblurring.py
```

Each script saves CSV files in `results/` and figures in `figures/`.

## Outputs

The script generates the following files:

```text
figures/mvp_denoising_comparison.png
figures/mvp_error_map.png
results/mvp_denoising_metrics.csv
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
  - `results/noise_sensitivity_results.csv`
  - `figures/noise_sensitivity_psnr.png`
  - `figures/noise_sensitivity_ssim.png`
  - `figures/noise_sensitivity_visual_grid.png`

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
  - `results/filter_sigma_sensitivity_results.csv`
  - `figures/filter_sigma_sensitivity_psnr.png`
  - `figures/filter_sigma_sensitivity_ssim.png`
  - `figures/filter_sigma_sensitivity_visual_grid.png`

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
  - `results/tikhonov_lambda_extended_results.csv`
  - `figures/tikhonov_lambda_extended_psnr.png`
  - `figures/tikhonov_lambda_extended_ssim.png`
  - `figures/tikhonov_lambda_extended_visual_grid.png`

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
  - `results/tv_denoising_results.csv`
  - `figures/tv_weight_sensitivity_psnr.png`
  - `figures/tv_weight_sensitivity_ssim.png`
  - `figures/tv_denoising_visual_grid.png`
  - `figures/tv_error_maps.png`

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

- `results/consolidated_denoising_comparison.csv`
- `figures/consolidated_psnr_comparison.png`
- `figures/consolidated_ssim_comparison.png`
- `figures/consolidated_runtime_comparison.png`
- `figures/consolidated_visual_comparison.png`

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
  - `results/tikhonov_deblurring_results.csv`
  - `figures/tikhonov_deblurring_psnr.png`
  - `figures/tikhonov_deblurring_ssim.png`
  - `figures/tikhonov_deblurring_visual_grid.png`
  - `figures/tikhonov_deblurring_error_maps.png`

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

## Next Steps

- Optionally implement TV deblurring or compare additional deblurring baselines.
- Prepare a short technical report.
- Polish the repository for GitHub presentation.
- Optionally add more test images to check whether the conclusions generalize.
