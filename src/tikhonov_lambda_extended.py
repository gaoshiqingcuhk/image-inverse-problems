from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, filters, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# Extended lambda sensitivity experiment for Tikhonov image denoising.
#
# The FFT implementation uses periodic boundary conditions. Under this
# boundary model, the discrete Laplacian is diagonal in the Fourier domain,
# which gives a simple closed-form solver for each lambda value.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

NOISE_SIGMA = 0.10
RANDOM_SEED = 42
GAUSSIAN_FILTER_SIGMA = 1.0

LAMBDAS = [
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
]
SELECTED_LAMBDAS_FOR_GRID = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


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
    """Denoise an image using FFT-based Tikhonov regularization.

    This solves the optimization problem

        min_x 0.5 * ||x - noisy||_2^2 + 0.5 * lam * ||grad x||_2^2

    with periodic boundary conditions. For an image with shape (m, n), the
    standard periodic discrete Laplacian eigenvalues are

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
    method: str,
    parameter_name: str,
    parameter_value: float | str,
    psnr: float,
    ssim: float,
    runtime_seconds: float,
) -> None:
    """Add one row to the results table."""
    rows.append(
        {
            "method": method,
            "noise_sigma": NOISE_SIGMA,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "PSNR": psnr,
            "SSIM": ssim,
            "runtime_seconds": runtime_seconds,
        }
    )


def run_experiment(
    clean: np.ndarray,
    noisy: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, dict[float, np.ndarray]]:
    """Run noisy, Gaussian filter, and extended Tikhonov comparisons."""
    rows: list[dict[str, float | str]] = []
    tikhonov_images = {}

    noisy_psnr, noisy_ssim = compute_metrics(clean, noisy)
    add_result_row(
        rows,
        method="noisy_image",
        parameter_name="",
        parameter_value="",
        psnr=noisy_psnr,
        ssim=noisy_ssim,
        runtime_seconds=0.0,
    )

    start_time = perf_counter()
    gaussian_result = filters.gaussian(noisy, sigma=GAUSSIAN_FILTER_SIGMA)
    gaussian_runtime = perf_counter() - start_time
    gaussian_psnr, gaussian_ssim = compute_metrics(clean, gaussian_result)
    add_result_row(
        rows,
        method="gaussian_filter",
        parameter_name="filter_sigma",
        parameter_value=GAUSSIAN_FILTER_SIGMA,
        psnr=gaussian_psnr,
        ssim=gaussian_ssim,
        runtime_seconds=gaussian_runtime,
    )

    for lam in LAMBDAS:
        start_time = perf_counter()
        tikhonov_result = tikhonov_denoise_fft(noisy, lam)
        runtime_seconds = perf_counter() - start_time

        psnr, ssim = compute_metrics(clean, tikhonov_result)
        add_result_row(
            rows,
            method="tikhonov",
            parameter_name="lambda",
            parameter_value=lam,
            psnr=psnr,
            ssim=ssim,
            runtime_seconds=runtime_seconds,
        )
        tikhonov_images[lam] = tikhonov_result

    return pd.DataFrame(rows), gaussian_result, tikhonov_images


def save_metric_curve(
    metrics: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    path: Path,
) -> None:
    """Save a lambda sensitivity curve with reference baselines."""
    tikhonov_results = metrics[metrics["method"] == "tikhonov"].copy()
    tikhonov_results["parameter_value"] = tikhonov_results["parameter_value"].astype(float)

    noisy_value = metrics.loc[metrics["method"] == "noisy_image", metric_name].iloc[0]
    gaussian_value = metrics.loc[metrics["method"] == "gaussian_filter", metric_name].iloc[0]

    best_index = tikhonov_results[metric_name].idxmax()
    best_row = tikhonov_results.loc[best_index]

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        tikhonov_results["parameter_value"],
        tikhonov_results[metric_name],
        marker="o",
        linewidth=2,
        label="Tikhonov",
    )
    axis.scatter(
        [best_row["parameter_value"]],
        [best_row[metric_name]],
        s=70,
        color="tab:red",
        zorder=3,
        label=f"Best Tikhonov {ylabel}",
    )
    axis.axhline(
        noisy_value,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Noisy image",
    )
    axis.axhline(
        gaussian_value,
        color="tab:green",
        linestyle="--",
        linewidth=1.5,
        label="Gaussian filter",
    )

    axis.set_xscale("log")
    axis.set_xlabel("lambda")
    axis.set_ylabel(ylabel)
    axis.set_title(f"Extended Tikhonov lambda sensitivity: {ylabel}")
    axis.grid(True, alpha=0.3, which="both")
    axis.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def format_title(method: str, psnr: float, ssim: float) -> str:
    """Make a compact subplot title."""
    return f"{method}\nPSNR={psnr:.2f}, SSIM={ssim:.3f}"


def save_visual_grid(
    clean: np.ndarray,
    noisy: np.ndarray,
    gaussian_result: np.ndarray,
    tikhonov_images: dict[float, np.ndarray],
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    """Save visual examples for baselines and selected Tikhonov lambdas."""
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    flat_axes = axes.ravel()

    noisy_row = metrics[metrics["method"] == "noisy_image"].iloc[0]
    gaussian_row = metrics[metrics["method"] == "gaussian_filter"].iloc[0]

    images_and_titles = [
        (clean, "Clean image"),
        (
            noisy,
            format_title("Noisy image", noisy_row["PSNR"], noisy_row["SSIM"]),
        ),
        (
            gaussian_result,
            format_title("Gaussian filter", gaussian_row["PSNR"], gaussian_row["SSIM"]),
        ),
    ]

    tikhonov_rows = metrics[metrics["method"] == "tikhonov"].copy()
    tikhonov_rows["parameter_value"] = tikhonov_rows["parameter_value"].astype(float)

    for lam in SELECTED_LAMBDAS_FOR_GRID:
        row = tikhonov_rows[np.isclose(tikhonov_rows["parameter_value"], lam)].iloc[0]
        images_and_titles.append(
            (
                tikhonov_images[lam],
                format_title(f"Tikhonov lambda={lam:g}", row["PSNR"], row["SSIM"]),
            )
        )

    for axis, (image, title) in zip(flat_axes, images_and_titles):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_folders()

    clean = img_as_float(data.camera())
    noisy = add_gaussian_noise(clean)

    metrics, gaussian_result, tikhonov_images = run_experiment(clean, noisy)

    results_path = RESULTS_DIR / "tikhonov_lambda_extended_results.csv"
    metrics.to_csv(results_path, index=False)

    save_metric_curve(
        metrics,
        metric_name="PSNR",
        ylabel="PSNR",
        path=FIGURES_DIR / "tikhonov_lambda_extended_psnr.png",
    )
    save_metric_curve(
        metrics,
        metric_name="SSIM",
        ylabel="SSIM",
        path=FIGURES_DIR / "tikhonov_lambda_extended_ssim.png",
    )
    save_visual_grid(
        clean,
        noisy,
        gaussian_result,
        tikhonov_images,
        metrics,
        path=FIGURES_DIR / "tikhonov_lambda_extended_visual_grid.png",
    )

    print("Extended Tikhonov lambda experiment completed successfully.")
    print(f"Results saved to: {results_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
