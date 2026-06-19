# Website Project Description

## Short Version

A reproducible Python study of image denoising and deblurring, comparing
classical filters, regularization, non-local methods, deconvolution algorithms,
robustness across degradation levels, and a lightweight CNN baseline.

## Medium Version

This project studies two foundational image inverse problems: recovering clean
images from noisy observations and recovering sharp images from blurred,
noisy observations. I implemented and compared Gaussian filtering, Tikhonov
regularization, Total Variation denoising, Non-local Means, Wiener
deconvolution, Richardson-Lucy deconvolution, and a small PyTorch CNN. The
study includes parameter sensitivity, multi-image tests, degradation-strength
robustness, quantitative evaluation with PSNR and SSIM, runtime comparisons,
and reproducible saved outputs. The focus is careful computational comparison
and honest analysis rather than a claim of a new algorithm.

## Technical Bullet Points

- Built reproducible denoising and deblurring experiments in Python.
- Implemented FFT-based Tikhonov solvers with explicit degradation models.
- Compared local, variational, non-local, frequency-domain, and iterative
  image priors.
- Evaluated PSNR, SSIM, runtime, parameter sensitivity, and visual error maps.
- Tested robustness across multiple images and degradation strengths.
- Trained and evaluated a lightweight CPU-friendly PyTorch denoising CNN.
- Organized scripts, metrics, figures, documentation, and model outputs using
  a versioned Git/GitHub workflow.

## GitHub Card Description

Reproducible image inverse problems project covering denoising, deblurring,
regularization, robustness analysis, and a lightweight CNN baseline in Python.
