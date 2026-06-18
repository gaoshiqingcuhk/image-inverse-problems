from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, filters, img_as_float, img_as_ubyte, io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# This script is intentionally simple: one image, one noise level, and one
# Gaussian-filter baseline. It is the first reproducible MVP experiment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "sample_images"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

NOISE_SIGMA = 0.10
FILTER_SIGMA = 1.0
RANDOM_SEED = 42


def ensure_output_folders() -> None:
    """Create all folders used by this experiment."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_grayscale_png(image: np.ndarray, path: Path) -> None:
    """Save a float image in [0, 1] as an 8-bit grayscale PNG."""
    io.imsave(path, img_as_ubyte(np.clip(image, 0.0, 1.0)))


def compute_metrics(clean: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM against the clean reference image."""
    psnr = peak_signal_noise_ratio(clean, test, data_range=1.0)
    ssim = structural_similarity(clean, test, data_range=1.0)
    return psnr, ssim


def save_comparison_figure(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    path: Path,
) -> None:
    """Save a side-by-side visual comparison."""
    images = [clean, noisy, denoised]
    titles = ["Clean image", "Noisy image", "Gaussian denoising"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_error_map(clean: np.ndarray, denoised: np.ndarray, path: Path) -> None:
    """Save the absolute error between the clean and denoised images."""
    error = np.abs(clean - denoised)

    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(error, cmap="inferno", vmin=0.0)
    axis.set_title("Absolute error: clean vs denoised")
    axis.axis("off")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_folders()

    # 1. Load a built-in grayscale test image and convert it to [0, 1].
    clean = img_as_float(data.camera())

    # 2. Add reproducible Gaussian noise.
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=clean.shape)
    noisy = np.clip(clean + noise, 0.0, 1.0)

    # 3. Denoise using a simple Gaussian filter baseline and time it.
    start_time = perf_counter()
    denoised = filters.gaussian(noisy, sigma=FILTER_SIGMA)
    runtime_seconds = perf_counter() - start_time

    # 4. Compute quantitative image-quality metrics.
    noisy_psnr, noisy_ssim = compute_metrics(clean, noisy)
    denoised_psnr, denoised_ssim = compute_metrics(clean, denoised)

    metrics = pd.DataFrame(
        [
            {
                "method": "noisy",
                "noise_sigma": NOISE_SIGMA,
                "filter_sigma": "",
                "PSNR": noisy_psnr,
                "SSIM": noisy_ssim,
                "runtime_seconds": 0.0,
            },
            {
                "method": "gaussian_filter",
                "noise_sigma": NOISE_SIGMA,
                "filter_sigma": FILTER_SIGMA,
                "PSNR": denoised_psnr,
                "SSIM": denoised_ssim,
                "runtime_seconds": runtime_seconds,
            },
        ]
    )

    # 5. Save metrics, images, and figures.
    metrics_path = RESULTS_DIR / "01_mvp_denoising_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    save_grayscale_png(clean, DATA_DIR / "camera_clean.png")
    save_grayscale_png(noisy, DATA_DIR / "camera_noisy_sigma_0.10.png")
    save_grayscale_png(denoised, DATA_DIR / "camera_denoised_gaussian.png")

    save_comparison_figure(
        clean,
        noisy,
        denoised,
        FIGURES_DIR / "01_mvp_denoising_comparison.png",
    )
    save_error_map(clean, denoised, FIGURES_DIR / "01_mvp_error_map.png")

    print("MVP denoising experiment completed successfully.")
    print(f"Metrics saved to: {metrics_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
