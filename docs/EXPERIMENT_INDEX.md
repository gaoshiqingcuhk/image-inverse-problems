# Experiment Index

The numbered prefixes record the order in which the experiments were added.
Every entry below corresponds to an existing script.

| Prefix | Script | Description |
|---:|---|---|
| 01 | `src/mvp_denoising.py` | MVP Gaussian denoising |
| 02 | `src/noise_sensitivity.py` | Gaussian noise-level sensitivity |
| 03 | `src/filter_sigma_sensitivity.py` | Gaussian filter-parameter sensitivity |
| 04 | `src/tikhonov_denoising.py` | Initial Tikhonov denoising study |
| 05 | `src/tikhonov_lambda_extended.py` | Extended Tikhonov lambda sensitivity |
| 06 | `src/tv_denoising.py` | TV Chambolle denoising and weight sensitivity |
| 07 | `src/consolidated_denoising_comparison.py` | Consolidated classical denoising comparison |
| 08 | `src/tikhonov_deblurring.py` | Tikhonov deblurring study |
| 09 | `src/multi_image_denoising_comparison.py` | Multi-image denoising robustness |
| 10 | `src/wiener_deblurring.py` | Wiener deconvolution baseline |
| 11 | `src/multi_image_deblurring_comparison.py` | Multi-image deblurring robustness |
| 12 | `src/nlm_denoising.py` | Single-image Non-local Means study |
| 13 | `src/multi_image_nlm_denoising_comparison.py` | Multi-image NLM robustness |
| 14 | `src/richardson_lucy_deblurring.py` | Richardson-Lucy deconvolution baseline |
| 15 | `src/degradation_robustness_study.py` | Noise- and blur-strength robustness |
| 16 | `src/tiny_cnn_denoising.py` | Tiny CNN denoising baseline |

Additional utilities:

| Script | Purpose |
|---|---|
| `src/check_ssim_behavior.py` | Diagnostic check for the filter-sigma SSIM value |
| `src/test_environment.py` | Dependency import check |
