from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, filters, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle


# Consolidated denoising comparison under one shared experimental setting.
# This script compares the main methods developed so far:
# Gaussian filtering, Tikhonov regularization, and Total Variation denoising.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

NOISE_SIGMA = 0.10
RANDOM_SEED = 42
GAUSSIAN_FILTER_SIGMA = 1.0
TIKHONOV_LAMBDAS = [1.0, 5.0]
TV_WEIGHT = 0.1


METHOD_LABELS = {
    "noisy_image": "Noisy image",
    "gaussian_filter": "Gaussian filter",
    "tikhonov_lambda_1.0": "Tikhonov lambda=1.0",
    "tikhonov_lambda_5.0": "Tikhonov lambda=5.0",
    "tv_chambolle_weight_0.1": "TV weight=0.1",
}


def ensure_output_folders() -> None:
    """Create folders for saved results and figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    """Add one reproducible Gaussian noise realization and clip to [0, 1]."""
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def tikhonov_denoise_fft(noisy: np.ndarray, lam: float) -> np.ndarray:
    """Denoise using FFT-based Tikhonov regularization.

    This solves

        min_x 0.5 * ||x - noisy||_2^2 + 0.5 * lam * ||grad x||_2^2

    with periodic boundary conditions. The standard periodic discrete
    Laplacian eigenvalues are

        eig = 4 - 2*cos(2*pi*k/m) - 2*cos(2*pi*l/n).

    The Fourier-domain solution is

        x_hat = noisy_hat / (1 + lam * eig).
    """
    rows, cols = noisy.shape

    row_freq = np.arange(rows).reshape(rows, 1)
    col_freq = np.arange(cols).reshape(1, cols)
    eig = (
        4.0
        - 2.0 * np.cos(2.0 * np.pi * row_freq / rows)
        - 2.0 * np.cos(2.0 * np.pi * col_freq / cols)
    )

    noisy_hat = np.fft.fft2(noisy)
    denoised_hat = noisy_hat / (1.0 + lam * eig)
    denoised = np.fft.ifft2(denoised_hat).real

    return np.clip(denoised, 0.0, 1.0)


def compute_metrics(clean: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM using the clean image as reference."""
    psnr = peak_signal_noise_ratio(clean, test, data_range=1.0)
    ssim = structural_similarity(clean, test, data_range=1.0)
    return psnr, ssim


def add_result_row(
    rows: list[dict[str, float | str]],
    clean: np.ndarray,
    image: np.ndarray,
    method: str,
    parameter: str,
    runtime_seconds: float,
    notes: str,
) -> None:
    """Compute metrics and append one row to the comparison table."""
    psnr, ssim = compute_metrics(clean, image)
    rows.append(
        {
            "method": method,
            "parameter": parameter,
            "PSNR": psnr,
            "SSIM": ssim,
            "runtime_seconds": runtime_seconds,
            "notes": notes,
        }
    )


def run_comparison(clean: np.ndarray, noisy: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run all main denoising methods under the same input setting."""
    rows: list[dict[str, float | str]] = []
    images = {
        "clean": clean,
        "noisy_image": noisy,
    }

    add_result_row(
        rows,
        clean,
        noisy,
        method="noisy_image",
        parameter="-",
        runtime_seconds=0.0,
        notes="degraded observation",
    )

    start_time = perf_counter()
    gaussian_result = filters.gaussian(
        noisy,
        sigma=GAUSSIAN_FILTER_SIGMA,
        preserve_range=True,
    )
    gaussian_runtime = perf_counter() - start_time
    images["gaussian_filter"] = gaussian_result
    add_result_row(
        rows,
        clean,
        gaussian_result,
        method="gaussian_filter",
        parameter=f"filter_sigma={GAUSSIAN_FILTER_SIGMA}",
        runtime_seconds=gaussian_runtime,
        notes="simple smoothing baseline",
    )

    for lam in TIKHONOV_LAMBDAS:
        start_time = perf_counter()
        tikhonov_result = tikhonov_denoise_fft(noisy, lam)
        runtime_seconds = perf_counter() - start_time

        method = f"tikhonov_lambda_{lam:.1f}"
        images[method] = tikhonov_result
        note = (
            "best Tikhonov PSNR setting from lambda sensitivity"
            if lam == 1.0
            else "best Tikhonov SSIM setting from lambda sensitivity"
        )
        add_result_row(
            rows,
            clean,
            tikhonov_result,
            method=method,
            parameter=f"lambda={lam}",
            runtime_seconds=runtime_seconds,
            notes=note,
        )

    start_time = perf_counter()
    tv_result = denoise_tv_chambolle(
        noisy,
        weight=TV_WEIGHT,
        channel_axis=None,
    )
    tv_runtime = perf_counter() - start_time
    tv_result = np.clip(tv_result, 0.0, 1.0)
    images["tv_chambolle_weight_0.1"] = tv_result
    add_result_row(
        rows,
        clean,
        tv_result,
        method="tv_chambolle_weight_0.1",
        parameter=f"weight={TV_WEIGHT}",
        runtime_seconds=tv_runtime,
        notes="best TV setting by both PSNR and SSIM",
    )

    return pd.DataFrame(rows), images


def save_bar_chart(
    metrics: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    path: Path,
    use_log_scale: bool = False,
) -> None:
    """Save a bar chart for one metric."""
    labels = [METHOD_LABELS[method] for method in metrics["method"]]

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, metrics[metric_name])
    axis.set_ylabel(ylabel)
    axis.set_title(f"Consolidated denoising comparison: {ylabel}")
    axis.tick_params(axis="x", labelrotation=25)

    if use_log_scale:
        axis.set_yscale("log")

    for index, value in enumerate(metrics[metric_name]):
        axis.text(
            index,
            value,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def format_title(method: str, psnr: float, ssim: float) -> str:
    """Make compact subplot titles for the visual comparison."""
    return f"{METHOD_LABELS[method]}\nPSNR={psnr:.2f}, SSIM={ssim:.3f}"


def save_visual_comparison(
    clean: np.ndarray,
    images: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    """Save clean, noisy, and denoised results side by side."""
    ordered_methods = [
        "clean",
        "noisy_image",
        "gaussian_filter",
        "tikhonov_lambda_1.0",
        "tikhonov_lambda_5.0",
        "tv_chambolle_weight_0.1",
    ]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 7))
    flat_axes = axes.ravel()

    for axis, method in zip(flat_axes, ordered_methods):
        axis.imshow(images[method], cmap="gray", vmin=0.0, vmax=1.0)
        if method == "clean":
            axis.set_title("Clean image")
        else:
            row = metrics[metrics["method"] == method].iloc[0]
            axis.set_title(format_title(method, row["PSNR"], row["SSIM"]))
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_folders()

    clean = img_as_float(data.camera())
    noisy = add_gaussian_noise(clean)

    metrics, images = run_comparison(clean, noisy)

    results_path = RESULTS_DIR / "07_consolidated_denoising_comparison.csv"
    metrics.to_csv(results_path, index=False)

    save_bar_chart(
        metrics,
        metric_name="PSNR",
        ylabel="PSNR",
        path=FIGURES_DIR / "07_consolidated_psnr_comparison.png",
    )
    save_bar_chart(
        metrics,
        metric_name="SSIM",
        ylabel="SSIM",
        path=FIGURES_DIR / "07_consolidated_ssim_comparison.png",
    )
    save_bar_chart(
        metrics,
        metric_name="runtime_seconds",
        ylabel="Runtime seconds",
        path=FIGURES_DIR / "07_consolidated_runtime_comparison.png",
        use_log_scale=True,
    )
    save_visual_comparison(
        clean,
        images,
        metrics,
        path=FIGURES_DIR / "07_consolidated_visual_comparison.png",
    )

    print("Consolidated denoising comparison completed successfully.")
    print(f"Results saved to: {results_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
