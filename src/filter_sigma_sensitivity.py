from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, filters, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# This script studies how the Gaussian filter parameter affects denoising.
# The noisy image is generated once and reused for every filter sigma.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

NOISE_SIGMA = 0.10
FILTER_SIGMAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
RANDOM_SEED = 42


def ensure_output_folders() -> None:
    """Create folders for saved results and figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    """Add one reproducible Gaussian noise realization and clip to [0, 1]."""
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def compute_metrics(clean: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM using the clean image as the reference."""
    psnr = peak_signal_noise_ratio(clean, test, data_range=1.0)
    ssim = structural_similarity(clean, test, data_range=1.0)
    return psnr, ssim


def run_experiment(clean: np.ndarray, noisy: np.ndarray) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Run the filter sigma sensitivity experiment."""
    rows = []
    denoised_images = {}

    noisy_psnr, noisy_ssim = compute_metrics(clean, noisy)
    rows.append(
        {
            "method": "noisy_image",
            "noise_sigma": NOISE_SIGMA,
            "filter_sigma": "",
            "PSNR": noisy_psnr,
            "SSIM": noisy_ssim,
            "runtime_seconds": 0.0,
        }
    )

    for filter_sigma in FILTER_SIGMAS:
        start_time = perf_counter()
        denoised = filters.gaussian(noisy, sigma=filter_sigma)
        runtime_seconds = perf_counter() - start_time

        psnr, ssim = compute_metrics(clean, denoised)
        rows.append(
            {
                "method": "gaussian_filter",
                "noise_sigma": NOISE_SIGMA,
                "filter_sigma": filter_sigma,
                "PSNR": psnr,
                "SSIM": ssim,
                "runtime_seconds": runtime_seconds,
            }
        )
        denoised_images[filter_sigma] = denoised

    return pd.DataFrame(rows), denoised_images


def save_metric_curve(
    metrics: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    baseline_value: float,
    path: Path,
) -> None:
    """Save a metric curve across filter sigma values with the noisy baseline."""
    filter_results = metrics[metrics["method"] == "gaussian_filter"]

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        filter_results["filter_sigma"],
        filter_results[metric_name],
        marker="o",
        linewidth=2,
        label="Gaussian filter",
    )
    axis.axhline(
        baseline_value,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Noisy image baseline",
    )

    axis.set_xlabel("Filter sigma")
    axis.set_ylabel(ylabel)
    axis.set_title(f"Filter sigma sensitivity: {ylabel}")
    axis.grid(True, alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_visual_grid(
    noisy: np.ndarray,
    denoised_images: dict[float, np.ndarray],
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    """Save the noisy image and all denoised images in one visual grid."""
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    flat_axes = axes.ravel()

    noisy_row = metrics[metrics["method"] == "noisy_image"].iloc[0]
    flat_axes[0].imshow(noisy, cmap="gray", vmin=0.0, vmax=1.0)
    flat_axes[0].set_title(
        f"Noisy image\nPSNR={noisy_row['PSNR']:.2f}, SSIM={noisy_row['SSIM']:.3f}"
    )
    flat_axes[0].axis("off")

    filter_results = metrics[metrics["method"] == "gaussian_filter"]
    for axis, (_, row) in zip(flat_axes[1:], filter_results.iterrows()):
        filter_sigma = row["filter_sigma"]
        denoised = denoised_images[filter_sigma]
        axis.imshow(denoised, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(
            f"filter_sigma={filter_sigma:.2f}\nPSNR={row['PSNR']:.2f}, SSIM={row['SSIM']:.3f}"
        )
        axis.axis("off")

    for axis in flat_axes[1 + len(FILTER_SIGMAS) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_folders()

    clean = img_as_float(data.camera())
    noisy = add_gaussian_noise(clean)

    metrics, denoised_images = run_experiment(clean, noisy)

    results_path = RESULTS_DIR / "filter_sigma_sensitivity_results.csv"
    metrics.to_csv(results_path, index=False)

    noisy_baseline = metrics[metrics["method"] == "noisy_image"].iloc[0]
    save_metric_curve(
        metrics,
        metric_name="PSNR",
        ylabel="PSNR",
        baseline_value=noisy_baseline["PSNR"],
        path=FIGURES_DIR / "filter_sigma_sensitivity_psnr.png",
    )
    save_metric_curve(
        metrics,
        metric_name="SSIM",
        ylabel="SSIM",
        baseline_value=noisy_baseline["SSIM"],
        path=FIGURES_DIR / "filter_sigma_sensitivity_ssim.png",
    )
    save_visual_grid(
        noisy,
        denoised_images,
        metrics,
        path=FIGURES_DIR / "filter_sigma_sensitivity_visual_grid.png",
    )

    print("Filter sigma sensitivity experiment completed successfully.")
    print(f"Results saved to: {results_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
