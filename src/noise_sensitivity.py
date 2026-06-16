from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, filters, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# This script studies one simple question:
# how does the Gaussian noise level affect denoising performance?
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

NOISE_SIGMAS = [0.05, 0.10, 0.15, 0.20]
FILTER_SIGMA = 1.0
RANDOM_SEED = 42


def ensure_output_folders() -> None:
    """Create folders for saved results and figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def add_gaussian_noise(image: np.ndarray, noise_sigma: float) -> np.ndarray:
    """Add reproducible Gaussian noise and clip the result to [0, 1]."""
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(loc=0.0, scale=noise_sigma, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def compute_metrics(clean: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM using the clean image as reference."""
    psnr = peak_signal_noise_ratio(clean, test, data_range=1.0)
    ssim = structural_similarity(clean, test, data_range=1.0)
    return psnr, ssim


def run_experiment(clean: np.ndarray) -> tuple[pd.DataFrame, dict[float, dict[str, np.ndarray]]]:
    """Run the noise sensitivity experiment and return metrics plus images."""
    rows = []
    images_by_sigma = {}

    for noise_sigma in NOISE_SIGMAS:
        noisy = add_gaussian_noise(clean, noise_sigma)

        start_time = perf_counter()
        denoised = filters.gaussian(noisy, sigma=FILTER_SIGMA)
        runtime_seconds = perf_counter() - start_time

        noisy_psnr, noisy_ssim = compute_metrics(clean, noisy)
        denoised_psnr, denoised_ssim = compute_metrics(clean, denoised)

        rows.append(
            {
                "noise_sigma": noise_sigma,
                "method": "noisy_image",
                "filter_sigma": "",
                "PSNR": noisy_psnr,
                "SSIM": noisy_ssim,
                "runtime_seconds": 0.0,
            }
        )
        rows.append(
            {
                "noise_sigma": noise_sigma,
                "method": "gaussian_filter",
                "filter_sigma": FILTER_SIGMA,
                "PSNR": denoised_psnr,
                "SSIM": denoised_ssim,
                "runtime_seconds": runtime_seconds,
            }
        )

        images_by_sigma[noise_sigma] = {
            "noisy_image": noisy,
            "gaussian_filter": denoised,
        }

    return pd.DataFrame(rows), images_by_sigma


def save_metric_curve(
    metrics: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    path: Path,
) -> None:
    """Save a line plot for PSNR or SSIM across noise levels."""
    fig, axis = plt.subplots(figsize=(6, 4))

    for method, label in [
        ("noisy_image", "Noisy image"),
        ("gaussian_filter", "Gaussian filter"),
    ]:
        method_data = metrics[metrics["method"] == method]
        axis.plot(
            method_data["noise_sigma"],
            method_data[metric_name],
            marker="o",
            linewidth=2,
            label=label,
        )

    axis.set_xlabel("Noise sigma")
    axis.set_ylabel(ylabel)
    axis.set_title(f"Noise sensitivity: {ylabel}")
    axis.grid(True, alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_visual_grid(images_by_sigma: dict[float, dict[str, np.ndarray]], path: Path) -> None:
    """Save a grid showing noisy and denoised images for each noise level."""
    fig, axes = plt.subplots(
        nrows=len(NOISE_SIGMAS),
        ncols=2,
        figsize=(7, 10),
    )

    for row_index, noise_sigma in enumerate(NOISE_SIGMAS):
        noisy = images_by_sigma[noise_sigma]["noisy_image"]
        denoised = images_by_sigma[noise_sigma]["gaussian_filter"]

        axes[row_index, 0].imshow(noisy, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 0].set_title(f"Noisy, sigma={noise_sigma:.2f}")
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(denoised, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 1].set_title(f"Gaussian filter, sigma={noise_sigma:.2f}")
        axes[row_index, 1].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_folders()

    clean = img_as_float(data.camera())
    metrics, images_by_sigma = run_experiment(clean)

    results_path = RESULTS_DIR / "noise_sensitivity_results.csv"
    metrics.to_csv(results_path, index=False)

    save_metric_curve(
        metrics,
        metric_name="PSNR",
        ylabel="PSNR",
        path=FIGURES_DIR / "noise_sensitivity_psnr.png",
    )
    save_metric_curve(
        metrics,
        metric_name="SSIM",
        ylabel="SSIM",
        path=FIGURES_DIR / "noise_sensitivity_ssim.png",
    )
    save_visual_grid(
        images_by_sigma,
        path=FIGURES_DIR / "noise_sensitivity_visual_grid.png",
    )

    print("Noise sensitivity experiment completed successfully.")
    print(f"Results saved to: {results_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
