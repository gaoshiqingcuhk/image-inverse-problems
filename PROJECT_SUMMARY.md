# Project Summary

## Project Title

**Image Inverse Problems: Denoising and Deblurring**

## One-paragraph Summary

This project is a reproducible computational study of image denoising and
deblurring. It compares local filtering, variational regularization, non-local
patch-based denoising, frequency-domain deconvolution, iterative
deconvolution, and a lightweight learned baseline. The experiments progress
from a single standard image to multi-image and degradation-strength
robustness studies. Each phase saves quantitative results, visual comparisons,
and reproducible scripts. The project does not propose a new restoration
algorithm; its purpose is to study how different image priors, parameters,
metrics, and computational costs affect restoration behavior.

## Research/Training Motivation

Image inverse problems provide a practical setting for connecting applied
mathematics with scientific computing. Denoising estimates a clean image from
randomly corrupted data, while deblurring must additionally account for an
image formation operator that removes high-frequency information. Both tasks
illustrate why direct inversion can be unstable and why prior assumptions or
regularization are needed.

The project was organized as an undergraduate research-training exercise. It
emphasizes controlled comparisons, reproducibility, parameter sensitivity,
honest negative results, and clear technical communication.

## Methods Covered

- Gaussian filtering as a fast local denoising baseline.
- Tikhonov regularization for denoising and FFT-based deblurring.
- Total Variation denoising as an edge-preserving variational method.
- Non-local Means as a patch-similarity denoising method.
- Wiener deconvolution as a stabilized frequency-domain inverse filter.
- Richardson-Lucy as an iterative deconvolution baseline.
- A small PyTorch CNN as a lightweight learning-based denoiser.

## Experiments Completed

The completed work includes single-image denoising and deblurring, noise and
regularization-parameter sensitivity studies, consolidated method comparisons,
multi-image robustness experiments, a degradation-strength robustness study,
and a Tiny CNN training and evaluation experiment. Standard `scikit-image`
images are used so that the experiments can be reproduced without downloading
a private dataset.

Outputs are organized with numbered prefixes from `01` to `16`. Metrics and
metadata are stored in `results/`, plots and visual comparisons in `figures/`,
the CNN checkpoint in `models/`, and the full narrative in
`report/mini_report.md`.

## Main Findings

- TV is a strong and stable classical denoising baseline, particularly for
  structural similarity and stronger noise.
- NLM can produce strong PSNR and competitive SSIM, but its ranking depends on
  filtering strength, image content, and degradation level.
- The denoising experiments show that PSNR and SSIM may favor different
  methods and that rankings can change as noise becomes stronger.
- Tikhonov is the most stable deblurring approach in the tested synthetic
  Gaussian blur grid. Wiener remains competitive but is not consistently best.
- Richardson-Lucy is useful as an iterative baseline, but its performance
  depends strongly on iteration count and does not exceed the selected
  Tikhonov or Wiener settings.
- The Tiny CNN learns during its short training run and improves over the noisy
  image, but it does not outperform the stronger classical baselines. This
  demonstrates that a learned model needs sufficient data, architecture, and
  training design to provide a reliable advantage.

## Technical Skills Demonstrated

- Python scientific computing with NumPy, pandas, Matplotlib, and scikit-image.
- Image processing, degradation modeling, and quality evaluation with PSNR and
  SSIM.
- Optimization and regularization concepts for inverse problems.
- FFT-based numerical solvers and deconvolution methods.
- Robustness analysis across images and degradation strengths.
- Lightweight PyTorch model design, training, validation, and checkpointing.
- Reproducible experiment organization with scripts, CSV outputs, figures,
  documentation, and Git/GitHub workflow.

## Limitations

The study uses synthetic Gaussian noise and blur, a small set of standard
grayscale images, and fixed parameters in robustness experiments. It does not
include a real-world dataset, blind restoration, or large-scale deep learning.
The Tiny CNN is intentionally small and should not be interpreted as
representative of modern state-of-the-art image restoration systems. Runtime
measurements are also dependent on the local machine and implementation.

## Possible Next Steps

- Evaluate on larger and real-world datasets.
- Study motion blur, non-Gaussian noise, and unknown degradation operators.
- Compare alternative boundary conditions and blind restoration methods.
- Train stronger learned models while retaining the same controlled and
  reproducible evaluation design.
- Add repeated-run timing and uncertainty summaries.
