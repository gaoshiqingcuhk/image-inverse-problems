# Optimization-Based and Regularization-Based Methods for Image Inverse Problems: A Reproducible Study

## Abstract

This report presents a reproducible computational mini-project on image inverse problems, focusing on image denoising and deblurring. The project studies how simple filtering and regularization-based methods recover a clean image from degraded observations. For denoising, Gaussian filtering, Tikhonov regularization, and Total Variation (TV) denoising are compared under controlled Gaussian noise. For deblurring, an FFT-based Tikhonov method is tested on images degraded by Gaussian blur and additive noise. The experiments are implemented in Python and evaluated using PSNR, SSIM, runtime, and visual comparisons. The denoising results show that Gaussian filtering is a strong and very fast baseline, while TV denoising achieves the best reconstruction quality in the tested setting. Tikhonov regularization provides an interpretable optimization-based formulation and illustrates the role of parameter sensitivity. The deblurring results show that Tikhonov regularization can improve blurred noisy observations, but weak regularization may amplify noise and artifacts. This work is intended as a reproducible research-training project rather than a novel algorithmic contribution.

## 1. Introduction

Image inverse problems arise when the goal is to estimate an unknown clean image from an observed degraded image. In many imaging systems, the measured image is not an exact copy of the underlying scene. It may be corrupted by sensor noise, optical blur, motion blur, compression artifacts, or incomplete measurements. The mathematical task is therefore to recover a plausible image from imperfect data.

This project studies two fundamental examples: denoising and deblurring. In denoising, the observed image is modeled as a clean image plus random noise. This is a natural starting point because the degradation is easy to simulate and the recovery task can be evaluated directly against the known clean image. In deblurring, the image is first transformed by a blur operator and then corrupted by noise. Deblurring is more challenging because blur weakens or removes high-frequency information such as edges and fine details. A naive inverse operation can therefore become unstable.

Regularization is a central tool for stabilizing inverse problems. It introduces prior assumptions about the desired reconstruction, such as smoothness or edge preservation. This project compares three types of methods: a simple Gaussian filtering baseline, Tikhonov regularization, and Total Variation denoising. These methods are not presented as new algorithms. Instead, they are used to build a small but organized computational study showing how mathematical models, numerical methods, parameter choices, and evaluation metrics interact in image restoration.

The main goal of the project is reproducibility. Each experiment produces saved figures, CSV result tables, and clear comparisons using PSNR, SSIM, runtime, and visual inspection. The project is designed as a research-training exercise suitable for developing practical experience in applied mathematics, scientific computing, optimization, and image processing.

## 2. Problem Formulation

For image denoising, the observation model is

```text
b = x + noise
```

where `x` denotes the clean image, `b` denotes the noisy observation, and `noise` denotes additive Gaussian noise. The goal is to construct an estimate of `x` using only the degraded observation `b`.

For image deblurring, the observation model is

```text
b = Kx + noise
```

where `K` is a blur operator. In this setting, the degradation is more severe because the operator `K` mixes neighboring pixel values and suppresses high-frequency image information. The recovery task is to estimate `x` from a blurred and noisy observation `b`.

The experiments use three main evaluation criteria. PSNR measures pixel-wise reconstruction quality and is derived from the mean squared error between the reconstruction and the clean image. A higher PSNR value indicates a smaller pixel-wise error. SSIM measures structural similarity and is designed to capture changes in image structure, contrast, and luminance. A higher SSIM value indicates stronger structural similarity to the clean image. Runtime measures the computational cost of the restoration method.

These metrics are complementary. PSNR may favor images with smaller pixel-wise errors, while SSIM may favor images that preserve visually meaningful structure. As shown in several experiments below, the best parameter according to PSNR is not always the best parameter according to SSIM.

## 3. Methods

### 3.1 Gaussian Filtering

Gaussian filtering is used as a simple smoothing baseline for denoising. The method replaces each pixel by a weighted average of nearby pixels, where closer pixels receive larger weights. This local averaging suppresses random fluctuations and often improves noisy images.

The key parameter is `filter_sigma`, which controls the spatial scale of smoothing. A small `filter_sigma` applies weak smoothing and may leave noise in the image. A large `filter_sigma` applies stronger smoothing but may blur edges, textures, and fine details.

Gaussian filtering is not formulated here as an explicit inverse problem solver. Its role is to provide a fast and interpretable baseline. Because it is simple and computationally cheap, it is useful for judging whether more mathematical methods actually provide a meaningful improvement.

### 3.2 Tikhonov Regularization for Denoising

Tikhonov denoising formulates image recovery as an optimization problem:

```text
min_x 0.5 * ||x - b||_2^2 + 0.5 * lambda * ||grad x||_2^2
```

The first term is the data fidelity term. It encourages the reconstructed image `x` to remain close to the noisy observation `b`. The second term is the regularization term. It penalizes large image gradients and therefore encourages smoothness. The parameter `lambda` controls the trade-off between these two goals.

When `lambda` is too small, the reconstruction remains close to the noisy image and may contain significant residual noise. When `lambda` is too large, the reconstruction becomes smoother but may lose edges and fine details. Thus, Tikhonov regularization makes the parameter trade-off explicit.

The implementation uses an FFT-based solver with periodic boundary conditions. This is computationally efficient and appropriate for a first reproducible implementation, although the periodic boundary assumption is a simplification that may introduce boundary artifacts.

### 3.3 Total Variation Denoising

Total Variation denoising is based on the conceptual model

```text
min_x 0.5 * ||x - b||_2^2 + lambda * ||grad x||_1
```

Like Tikhonov regularization, TV denoising balances fidelity to the noisy observation with a regularization term. The key difference is that TV uses an `L1`-type penalty on the image gradient instead of a squared `L2` penalty. This makes TV more edge-preserving: it tends to remove small oscillatory noise while allowing sharper jumps at important edges.

In the implementation, the `scikit-image` function `denoise_tv_chambolle` is used. Its regularization parameter is called `weight`. A larger `weight` produces stronger denoising, but if the value is too large, the image can become over-smoothed or cartoon-like, with piecewise-constant regions and reduced texture.

TV denoising is especially relevant for image inverse problems because it illustrates a different type of prior assumption. Instead of simply assuming smoothness everywhere, it favors images that are mostly smooth but may contain sharp edges.

### 3.4 Tikhonov Deblurring

For deblurring, the Tikhonov model becomes

```text
min_x 0.5 * ||Kx - b||_2^2 + 0.5 * lambda * ||grad x||_2^2
```

Here, the data fidelity term is different from denoising. It does not require the reconstruction `x` itself to look like the blurred observation `b`. Instead, it requires that applying the blur operator `K` to the reconstruction should reproduce the observation. This is the core inverse problem structure: the unknown clean image is estimated indirectly through the degradation model.

Deblurring is more unstable than denoising because blur suppresses high-frequency components. If regularization is too weak, the numerical inverse may amplify noise and produce ringing artifacts. If regularization is too strong, the reconstruction may remain overly smooth. The parameter `lambda` again controls the balance between aggressive inversion and stability.

The deblurring implementation also uses an FFT-based solver with periodic boundary conditions. The blur operator is represented in the frequency domain, and Tikhonov regularization stabilizes the division by small frequency components.

## 4. Experimental Setup

All experiments use the built-in grayscale image `skimage.data.camera()`. Pixel values are converted to floating-point values in `[0, 1]`. Randomness is controlled by fixed seeds to make the results reproducible.

For denoising experiments, Gaussian noise is added with

```text
noise_sigma = 0.10
random seed = 42
```

For deblurring experiments, a Gaussian blur kernel is used with

```text
blur_sigma = 2.0
noise_sigma = 0.01
random seed = 42
```

Each experiment saves quantitative results to CSV files and visual results to figure files. The main quantitative metrics are PSNR, SSIM, and runtime. Visual comparisons and error maps are also used because image restoration quality cannot be fully understood from scalar metrics alone.

## 5. Results

### 5.1 MVP Denoising Baseline

The first experiment compares the noisy image with a Gaussian-filtered image.

| Method | PSNR | SSIM | Runtime seconds |
|---|---:|---:|---:|
| Noisy image | 20.421019 | 0.296393 | 0.000000 |
| Gaussian filter | 27.171126 | 0.638713 | 0.006478 |

Gaussian filtering substantially improves both PSNR and SSIM compared with the noisy image. This confirms that simple smoothing can remove a meaningful amount of additive Gaussian noise. However, visual inspection shows that the restored image is also smoother than the original, especially near edges and fine details. This motivates the study of regularization-based methods.

![MVP denoising comparison](../figures/01_mvp_denoising_comparison.png)

### 5.2 Sensitivity Analysis

Two sensitivity experiments were performed for the Gaussian filtering baseline. The first varied the noise level while keeping `filter_sigma = 1.0` fixed. The tested noise levels were `noise_sigma = 0.05, 0.10, 0.15, 0.20`. As the noise level increases, the quality of the noisy image decreases. Gaussian filtering improves the metrics at all tested noise levels, but the final reconstruction quality still becomes worse under stronger noise. This shows that smoothing helps but cannot fully recover information lost under severe noise.

![Noise sensitivity PSNR](../figures/02_noise_sensitivity_psnr.png)

The second sensitivity experiment fixed `noise_sigma = 0.10` and varied `filter_sigma = 0.25, 0.5, 1.0, 1.5, 2.0, 3.0`. The PSNR optimum occurs at `filter_sigma = 1.0`, while the SSIM optimum occurs at `filter_sigma = 2.0`. This difference shows that parameter selection depends on the evaluation metric. A parameter value that minimizes pixel-wise error may not be the same value that maximizes structural similarity.

![Filter sigma sensitivity PSNR](../figures/03_filter_sigma_sensitivity_psnr.png)

The Tikhonov denoising lambda sensitivity experiment shows a similar pattern: parameter choice strongly affects performance. Small `lambda` values provide insufficient smoothing, while very large values can over-smooth the reconstruction.

![Tikhonov lambda sensitivity PSNR](../figures/05_tikhonov_lambda_extended_psnr.png)

### 5.3 Consolidated Denoising Comparison

The main denoising methods were compared under the same setting: the same clean image, the same Gaussian noise level, and the same random seed.

| Method | Parameter | PSNR | SSIM | Runtime seconds |
|---|---:|---:|---:|---:|
| Noisy image | - | 20.421019 | 0.296393 | 0.000000 |
| Gaussian filter | filter_sigma = 1.0 | 27.171126 | 0.638713 | 0.004875 |
| Tikhonov | lambda = 1.0 | 26.801759 | 0.622013 | 0.018835 |
| Tikhonov | lambda = 5.0 | 24.668893 | 0.689946 | 0.016929 |
| TV Chambolle | weight = 0.1 | 28.302925 | 0.756463 | 0.266987 |

The noisy image has the lowest PSNR and SSIM, as expected. Gaussian filtering is a strong baseline and is also the fastest restoration method among the tested algorithms. Tikhonov regularization provides a clear optimization-based formulation, but it does not achieve the best reconstruction quality in this denoising setting. Tikhonov with `lambda = 5.0` achieves higher SSIM than Gaussian filtering, but its PSNR is lower.

TV Chambolle with `weight = 0.1` achieves the highest PSNR and SSIM among all tested denoising methods. This suggests that TV regularization better preserves image structure and edges in this experiment. The trade-off is computational cost: TV is noticeably slower than both Gaussian filtering and FFT-based Tikhonov denoising.

![Consolidated visual comparison](../figures/07_consolidated_visual_comparison.png)

![Consolidated PSNR comparison](../figures/07_consolidated_psnr_comparison.png)

![Consolidated SSIM comparison](../figures/07_consolidated_ssim_comparison.png)

![Consolidated runtime comparison](../figures/07_consolidated_runtime_comparison.png)

TV denoising is also illustrated visually below.

![TV denoising visual grid](../figures/06_tv_denoising_visual_grid.png)

As a Phase 2 robustness check, the main denoising methods were also evaluated
on four built-in `scikit-image` images: `camera`, `coins`, `moon`, and `page`.
The same noise level and method parameters were used for all images. In this
multi-image study, TV Chambolle with `weight = 0.1` achieved the best PSNR and
SSIM on all tested images, while Gaussian filtering remained the fastest
restoration method. This supports the main single-image conclusion while also
showing that Tikhonov `lambda = 5.0` is not uniformly robust across image
types.

### 5.4 Tikhonov Deblurring Results

The deblurring experiment applies Gaussian blur and small Gaussian noise, then solves a Tikhonov deblurring problem over several values of `lambda`.

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

The blurred noisy image is worse than the blurred-only image because noise further degrades the observation. Very small `lambda` values produce unstable deblurring: they attempt to invert the blur too aggressively and may amplify noise or ringing artifacts. Moderate `lambda` values improve reconstruction quality. The best PSNR occurs at `lambda = 0.01`, while the best SSIM occurs at `lambda = 0.05`.

This result demonstrates why regularization is important for deblurring. Direct or weakly regularized inversion can be unstable, while a moderate regularization level can improve both pixel-wise and structural reconstruction quality.

![Tikhonov deblurring visual grid](../figures/08_tikhonov_deblurring_visual_grid.png)

![Tikhonov deblurring PSNR](../figures/08_tikhonov_deblurring_psnr.png)

![Tikhonov deblurring SSIM](../figures/08_tikhonov_deblurring_ssim.png)

## 6. Discussion

The experiments show several consistent patterns. First, Gaussian filtering is a strong baseline for additive Gaussian noise. It is easy to implement, very fast, and gives a large improvement over the noisy image. However, because it is a generic smoothing method, it can blur edges and fine structures.

Second, Tikhonov regularization is valuable because it introduces an explicit optimization-based formulation. It makes the trade-off between data fidelity and smoothness mathematically visible. In denoising, Tikhonov did not outperform TV or the best Gaussian filter result in terms of PSNR, but it provided a clear framework for studying regularization parameters. In deblurring, Tikhonov was especially useful because it stabilized an otherwise ill-conditioned inverse problem.

Third, TV denoising achieved the best denoising quality in the tested setting. Its edge-preserving regularization produced the highest PSNR and SSIM among the compared denoising methods. This supports the intuition that an `L1`-type gradient penalty can better preserve edges than the squared-gradient penalty used in Tikhonov regularization. The cost is runtime: TV denoising is slower than both Gaussian filtering and FFT-based Tikhonov denoising.

The multi-image robustness experiment supports the conclusion that TV gives
the best denoising quality in the tested setting, while Gaussian filtering
remains the fastest restoration method.

A recurring theme is that optimal parameters depend on the chosen metric. In Gaussian filtering, Tikhonov denoising, and Tikhonov deblurring, the parameter that maximized PSNR was not always the parameter that maximized SSIM. This is important because PSNR and SSIM measure different aspects of image quality. A well-designed inverse problem experiment should therefore report multiple metrics and include visual comparisons rather than relying on a single number.

The deblurring experiment also highlights the instability of inverse problems. When `lambda` is too small, the method attempts to invert the blur too aggressively and can amplify noise. When `lambda` is moderate, the method improves both PSNR and SSIM. This illustrates the practical role of regularization in stabilizing inverse problems.

## 7. Limitations

This project has several limitations. Although a small multi-image robustness
check was added, only a few standard grayscale test images were used, so the
numerical conclusions may not generalize to broader image types. The FFT-based
solvers use periodic boundary conditions, which are mathematically convenient
but may not match real image boundaries. The blur model is synthetic and uses a
Gaussian kernel rather than real camera or motion blur.

The project also does not include learning-based or deep learning methods. No real-world image dataset was used. TV deblurring was not implemented, and no plug-and-play or learned priors were tested. More images and real-world datasets are still needed to test whether the conclusions generalize. The experiments therefore should be interpreted as a controlled computational study rather than a comprehensive benchmark.

Future work should test more images, multiple noise and blur settings, alternative boundary conditions, TV deblurring, and simple learning-based baselines. These extensions would help determine whether the observed conclusions remain stable across broader image restoration tasks.

## 8. Conclusion

This project builds a reproducible workflow for studying image inverse problems. It implements denoising and deblurring experiments, compares simple and regularization-based methods, and evaluates reconstruction quality using PSNR, SSIM, runtime, and visual inspection.

For denoising, Gaussian filtering is fast and effective, but TV denoising gives the best reconstruction quality in the tested setting. Tikhonov regularization provides a clear optimization-based model and demonstrates the importance of regularization parameter selection. For deblurring, Tikhonov regularization improves blurred noisy observations but requires careful `lambda` selection to avoid unstable inversion.

Overall, the project demonstrates the value of regularization, parameter sensitivity analysis, and reproducible numerical experiments in image inverse problems. It provides a compact foundation for further study, including more test images, TV deblurring, plug-and-play methods, and learning-based restoration baselines.

## References

[1] A. N. Tikhonov and V. Y. Arsenin, *Solutions of Ill-Posed Problems*. Washington, DC: Winston, 1977.

[2] L. I. Rudin, S. Osher, and E. Fatemi, "Nonlinear total variation based noise removal algorithms," *Physica D: Nonlinear Phenomena*, vol. 60, no. 1â€?, pp. 259â€?68, 1992.

[3] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600â€?12, 2004.

[4] scikit-image contributors, "scikit-image: Image processing in Python," documentation for `skimage.filters`, `skimage.restoration`, and `skimage.metrics`.
