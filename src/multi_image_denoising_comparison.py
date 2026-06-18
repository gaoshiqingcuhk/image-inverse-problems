from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, data, filters, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle


# Phase 2: robustness check across several built-in skimage test images.
# This script intentionally reuses the same denoising methods and parameters
# from the main single-image comparison so that results are easy to compare.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

IMAGE_LOADERS = [
    ("camera", data.camera),
    ("coins", data.coins),
    ("moon", data.moon),
    ("page", data.page),
]

NOISE_SIGMA = 0.10
BASE_RANDOM_SEED = 42
GAUSSIAN_FILTER_SIGMA = 1.0
TIKHONOV_LAMBDAS = [1.0, 5.0]
TV_WEIGHT = 0.1

METHOD_ORDER = [
    "noisy_image",
    "gaussian_filter",
    "tikhonov_lambda_1.0",
    "tikhonov_lambda_5.0",
    "tv_chambolle_weight_0.1",
]

METHOD_LABELS = {
    "noisy_image": "Noisy",
    "gaussian_filter": "Gaussian",
    "tikhonov_lambda_1.0": "Tikhonov 1.0",
    "tikhonov_lambda_5.0": "Tikhonov 5.0",
    "tv_chambolle_weight_0.1": "TV 0.1",
}


def ensure_output_folders() -> None:
    """Create folders for saved results and figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_grayscale_float(loader) -> np.ndarray:
    """Load an image, convert to grayscale if needed, and scale to [0, 1]."""
    image = loader()
    image = img_as_float(image)

    if image.ndim == 3:
        image = color.rgb2gray(image)

    return np.clip(image, 0.0, 1.0)


def add_gaussian_noise(image: np.ndarray, seed: int) -> np.ndarray:
    """Add reproducible Gaussian noise and clip to [0, 1]."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def tikhonov_denoise_fft(noisy: np.ndarray, lam: float) -> np.ndarray:
    """Denoise using FFT-based Tikhonov regularization.

    The solver uses periodic boundary conditions. With this boundary model,
    the discrete Laplacian is diagonal in the Fourier domain.
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
    image_name: str,
    method: str,
    parameter: str,
    clean: np.ndarray,
    result: np.ndarray,
    runtime_seconds: float,
) -> None:
    """Compute metrics and append one row to the output table."""
    psnr, ssim = compute_metrics(clean, result)
    rows.append(
        {
            "image_name": image_name,
            "method": method,
            "parameter": parameter,
            "PSNR": psnr,
            "SSIM": ssim,
            "runtime_seconds": runtime_seconds,
        }
    )


def run_methods_for_image(
    image_name: str,
    clean: np.ndarray,
    noisy: np.ndarray,
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    """Run all denoising methods for one image."""
    rows: list[dict[str, float | str]] = []
    images = {
        "clean": clean,
        "noisy_image": noisy,
    }

    add_result_row(rows, image_name, "noisy_image", "-", clean, noisy, 0.0)

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
        image_name,
        "gaussian_filter",
        f"filter_sigma={GAUSSIAN_FILTER_SIGMA}",
        clean,
        gaussian_result,
        gaussian_runtime,
    )

    for lam in TIKHONOV_LAMBDAS:
        start_time = perf_counter()
        tikhonov_result = tikhonov_denoise_fft(noisy, lam)
        runtime_seconds = perf_counter() - start_time

        method = f"tikhonov_lambda_{lam:.1f}"
        images[method] = tikhonov_result
        add_result_row(
            rows,
            image_name,
            method,
            f"lambda={lam}",
            clean,
            tikhonov_result,
            runtime_seconds,
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
        image_name,
        "tv_chambolle_weight_0.1",
        f"weight={TV_WEIGHT}",
        clean,
        tv_result,
        tv_runtime,
    )

    return rows, images


def run_experiment() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Run the multi-image comparison and return metrics plus images."""
    all_rows = []
    all_images = {}

    for image_index, (image_name, loader) in enumerate(IMAGE_LOADERS):
        clean = load_grayscale_float(loader)
        noisy = add_gaussian_noise(clean, seed=BASE_RANDOM_SEED + image_index)
        rows, images = run_methods_for_image(image_name, clean, noisy)
        all_rows.extend(rows)
        all_images[image_name] = images

    return pd.DataFrame(all_rows), all_images


def save_metric_by_method_figure(
    metrics: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    path: Path,
    include_noisy: bool = True,
    use_log_scale: bool = False,
) -> None:
    """Save grouped bar chart comparing methods across images."""
    methods = METHOD_ORDER if include_noisy else METHOD_ORDER[1:]
    image_names = [name for name, _ in IMAGE_LOADERS]
    x = np.arange(len(image_names))
    width = 0.14 if include_noisy else 0.17

    fig, axis = plt.subplots(figsize=(9, 4.8))

    for method_index, method in enumerate(methods):
        values = []
        for image_name in image_names:
            row = metrics[
                (metrics["image_name"] == image_name)
                & (metrics["method"] == method)
            ].iloc[0]
            values.append(row[metric_name])

        offset = (method_index - (len(methods) - 1) / 2.0) * width
        axis.bar(x + offset, values, width=width, label=METHOD_LABELS[method])

    axis.set_xticks(x)
    axis.set_xticklabels(image_names)
    axis.set_ylabel(ylabel)
    axis.set_title(f"Multi-image denoising comparison: {ylabel}")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    if use_log_scale:
        axis.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def format_title(method: str, psnr: float | None = None, ssim: float | None = None) -> str:
    """Create compact subplot titles."""
    if psnr is None or ssim is None:
        return "Clean"
    return f"{METHOD_LABELS[method]}\nPSNR={psnr:.2f}, SSIM={ssim:.3f}"


def save_visual_grid(
    metrics: pd.DataFrame,
    all_images: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    """Save one row per image and one column per method."""
    image_names = [name for name, _ in IMAGE_LOADERS]
    columns = ["clean"] + METHOD_ORDER

    fig, axes = plt.subplots(
        nrows=len(image_names),
        ncols=len(columns),
        figsize=(15, 10),
    )

    for row_index, image_name in enumerate(image_names):
        for col_index, column in enumerate(columns):
            axis = axes[row_index, col_index]
            axis.imshow(all_images[image_name][column], cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")

            if column == "clean":
                axis.set_title(f"{image_name}\nClean")
            else:
                metric_row = metrics[
                    (metrics["image_name"] == image_name)
                    & (metrics["method"] == column)
                ].iloc[0]
                axis.set_title(
                    format_title(column, metric_row["PSNR"], metric_row["SSIM"]),
                    fontsize=9,
                )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_folders()

    metrics, all_images = run_experiment()

    results_path = RESULTS_DIR / "09_multi_image_denoising_comparison.csv"
    metrics.to_csv(results_path, index=False)

    save_metric_by_method_figure(
        metrics,
        metric_name="PSNR",
        ylabel="PSNR",
        path=FIGURES_DIR / "09_multi_image_psnr_by_method.png",
    )
    save_metric_by_method_figure(
        metrics,
        metric_name="SSIM",
        ylabel="SSIM",
        path=FIGURES_DIR / "09_multi_image_ssim_by_method.png",
    )
    save_metric_by_method_figure(
        metrics,
        metric_name="runtime_seconds",
        ylabel="Runtime seconds",
        path=FIGURES_DIR / "09_multi_image_runtime_by_method.png",
        include_noisy=False,
        use_log_scale=True,
    )
    save_visual_grid(
        metrics,
        all_images,
        path=FIGURES_DIR / "09_multi_image_visual_grid.png",
    )

    print("Multi-image denoising comparison completed successfully.")
    print(f"Results saved to: {results_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
