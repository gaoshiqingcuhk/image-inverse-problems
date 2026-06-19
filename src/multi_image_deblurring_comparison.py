from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# Phase 4: multi-image deblurring robustness study.
#
# This script reuses the FFT-based circular-convolution degradation model from
# the single-image Tikhonov and Wiener deblurring experiments. The fixed
# parameters are chosen from the Phase 3 single-image study and are not tuned
# separately for each image.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

IMAGE_LOADERS = [
    ("camera", data.camera),
    ("coins", data.coins),
    ("moon", data.moon),
    ("page", data.page),
]

BLUR_KERNEL_SIZE = 21
BLUR_SIGMA = 2.0
NOISE_SIGMA = 0.01
BASE_RANDOM_SEED = 42

TIKHONOV_LAMBDAS = [0.01, 0.05]
WIENER_BALANCES = [0.01, 0.03]

METHOD_ORDER = [
    "blurred_noisy",
    "tikhonov_deblur_lambda_0.01",
    "tikhonov_deblur_lambda_0.05",
    "wiener_deblur_balance_0.01",
    "wiener_deblur_balance_0.03",
]

METHOD_LABELS = {
    "blurred_noisy": "Blurred + noise",
    "tikhonov_deblur_lambda_0.01": "Tikhonov 0.01",
    "tikhonov_deblur_lambda_0.05": "Tikhonov 0.05",
    "wiener_deblur_balance_0.01": "Wiener 0.01",
    "wiener_deblur_balance_0.03": "Wiener 0.03",
}


def ensure_output_folders() -> None:
    """Create folders used by this experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_float_gray(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale float values in [0, 1]."""
    image = img_as_float(image)
    if image.ndim == 3:
        image = color.rgb2gray(image)
    return np.clip(image, 0.0, 1.0)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a normalized 2D Gaussian blur kernel."""
    coordinates = np.arange(size) - size // 2
    xx, yy = np.meshgrid(coordinates, coordinates)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def psf_to_otf(kernel: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Convert a small blur kernel into an image-sized FFT response."""
    padded = np.zeros(image_shape, dtype=float)
    kernel_rows, kernel_cols = kernel.shape
    padded[:kernel_rows, :kernel_cols] = kernel
    padded = np.roll(padded, shift=-(kernel_rows // 2), axis=0)
    padded = np.roll(padded, shift=-(kernel_cols // 2), axis=1)
    return np.fft.fft2(padded)


def apply_blur_fft(clean: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blur an image using FFT-based circular convolution."""
    H = psf_to_otf(kernel, clean.shape)
    blurred = np.fft.ifft2(np.fft.fft2(clean) * H).real
    return np.clip(blurred, 0.0, 1.0), H


def add_gaussian_noise(image: np.ndarray, seed: int) -> np.ndarray:
    """Add deterministic Gaussian noise and clip to [0, 1]."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def laplacian_eigenvalues(image_shape: tuple[int, int]) -> np.ndarray:
    """Return periodic discrete Laplacian eigenvalues."""
    rows, cols = image_shape
    row_freq = np.arange(rows).reshape(rows, 1)
    col_freq = np.arange(cols).reshape(1, cols)
    return (
        4.0
        - 2.0 * np.cos(2.0 * np.pi * row_freq / rows)
        - 2.0 * np.cos(2.0 * np.pi * col_freq / cols)
    )


def tikhonov_deblur_fft(observed: np.ndarray, H: np.ndarray, lam: float) -> np.ndarray:
    """Deblur using FFT-based Tikhonov regularization."""
    b_hat = np.fft.fft2(observed)
    eig_lap = laplacian_eigenvalues(observed.shape)
    denominator = np.abs(H) ** 2 + lam * eig_lap
    x_hat = np.conj(H) * b_hat / denominator
    restored = np.fft.ifft2(x_hat).real
    return np.clip(restored, 0.0, 1.0)


def wiener_deblur_fft(observed: np.ndarray, H: np.ndarray, balance: float) -> np.ndarray:
    """Deblur using the FFT Wiener inverse-filter baseline."""
    b_hat = np.fft.fft2(observed)
    denominator = np.abs(H) ** 2 + balance
    x_hat = np.conj(H) * b_hat / denominator
    restored = np.fft.ifft2(x_hat).real
    return np.clip(restored, 0.0, 1.0)


def compute_metrics(clean: np.ndarray, restored: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM using the clean image as reference."""
    psnr = peak_signal_noise_ratio(clean, restored, data_range=1.0)
    ssim = structural_similarity(clean, restored, data_range=1.0)
    return psnr, ssim


def add_result_row(
    rows: list[dict[str, float | str]],
    image_name: str,
    method: str,
    parameter: str,
    clean: np.ndarray,
    restored: np.ndarray,
    runtime_seconds: float,
) -> None:
    """Compute metrics and append one result row."""
    psnr, ssim = compute_metrics(clean, restored)
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
    kernel: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    """Run all fixed-parameter deblurring methods for one image."""
    rows: list[dict[str, float | str]] = []
    images: dict[str, np.ndarray] = {"clean": clean}

    blurred, H = apply_blur_fft(clean, kernel)
    blurred_noisy = add_gaussian_noise(blurred, seed=seed)
    images["blurred_noisy"] = blurred_noisy

    add_result_row(rows, image_name, "blurred_noisy", "-", clean, blurred_noisy, 0.0)

    for lam in TIKHONOV_LAMBDAS:
        start_time = perf_counter()
        restored = tikhonov_deblur_fft(blurred_noisy, H, lam)
        runtime_seconds = perf_counter() - start_time
        key = f"tikhonov_deblur_lambda_{lam:g}"
        images[key] = restored
        add_result_row(
            rows,
            image_name,
            "tikhonov_deblur",
            f"lambda={lam:g}",
            clean,
            restored,
            runtime_seconds,
        )

    for balance in WIENER_BALANCES:
        start_time = perf_counter()
        restored = wiener_deblur_fft(blurred_noisy, H, balance)
        runtime_seconds = perf_counter() - start_time
        key = f"wiener_deblur_balance_{balance:g}"
        images[key] = restored
        add_result_row(
            rows,
            image_name,
            "wiener_deblur",
            f"balance={balance:g}",
            clean,
            restored,
            runtime_seconds,
        )

    return rows, images


def run_experiment() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Run the multi-image deblurring comparison."""
    all_rows = []
    all_images = {}
    kernel = gaussian_kernel(BLUR_KERNEL_SIZE, BLUR_SIGMA)

    for image_index, (image_name, loader) in enumerate(IMAGE_LOADERS):
        clean = ensure_float_gray(loader())
        rows, images = run_methods_for_image(
            image_name=image_name,
            clean=clean,
            kernel=kernel,
            seed=BASE_RANDOM_SEED + image_index,
        )
        all_rows.extend(rows)
        all_images[image_name] = images

    return pd.DataFrame(all_rows), all_images


def method_key(method: str, parameter: str) -> str:
    """Convert method and parameter columns into a plotting key."""
    if method == "blurred_noisy":
        return "blurred_noisy"
    value = parameter.split("=", maxsplit=1)[1]
    return f"{method}_{parameter.split('=', maxsplit=1)[0]}_{value}"


def save_metric_by_method_figure(
    metrics: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    path: Path,
    include_blurred_noisy: bool = True,
    use_log_scale: bool = False,
) -> None:
    """Save grouped bar chart comparing methods across images."""
    methods = METHOD_ORDER if include_blurred_noisy else METHOD_ORDER[1:]
    image_names = [name for name, _ in IMAGE_LOADERS]
    x = np.arange(len(image_names))
    width = 0.14 if include_blurred_noisy else 0.17

    fig, axis = plt.subplots(figsize=(9.5, 4.8), facecolor="white")

    for method_index, method in enumerate(methods):
        values = []
        for image_name in image_names:
            image_rows = metrics[metrics["image_name"] == image_name].copy()
            image_rows["method_key"] = image_rows.apply(
                lambda row: method_key(row["method"], row["parameter"]),
                axis=1,
            )
            values.append(image_rows[image_rows["method_key"] == method][metric_name].iloc[0])

        offset = (method_index - (len(methods) - 1) / 2.0) * width
        axis.bar(x + offset, values, width=width, label=METHOD_LABELS[method])

    axis.set_xticks(x)
    axis.set_xticklabels(image_names)
    axis.set_ylabel(ylabel)
    axis.set_title(f"Multi-image deblurring comparison: {ylabel}")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    if use_log_scale:
        axis.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def format_title(label: str, psnr: float | None = None, ssim: float | None = None) -> str:
    """Create readable subplot titles."""
    if psnr is None or ssim is None:
        return label
    return f"{label}\nPSNR={psnr:.2f}, SSIM={ssim:.3f}"


def save_visual_grid(
    metrics: pd.DataFrame,
    all_images: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    """Save visual grid with one image per row and one result per column."""
    image_names = [name for name, _ in IMAGE_LOADERS]
    columns = ["clean"] + METHOD_ORDER

    fig, axes = plt.subplots(
        nrows=len(image_names),
        ncols=len(columns),
        figsize=(16, 10),
        facecolor="white",
    )

    for row_index, image_name in enumerate(image_names):
        image_rows = metrics[metrics["image_name"] == image_name].copy()
        image_rows["method_key"] = image_rows.apply(
            lambda row: method_key(row["method"], row["parameter"]),
            axis=1,
        )

        for col_index, column in enumerate(columns):
            axis = axes[row_index, col_index]
            axis.imshow(all_images[image_name][column], cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")

            if column == "clean":
                axis.set_title(f"{image_name}\nClean", fontsize=9)
            else:
                row = image_rows[image_rows["method_key"] == column].iloc[0]
                axis.set_title(
                    format_title(METHOD_LABELS[column], row["PSNR"], row["SSIM"]),
                    fontsize=8,
                )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_folders()

    metrics, all_images = run_experiment()

    results_path = RESULTS_DIR / "11_multi_image_deblurring_comparison.csv"
    metrics.to_csv(results_path, index=False)

    save_metric_by_method_figure(
        metrics,
        metric_name="PSNR",
        ylabel="PSNR",
        path=FIGURES_DIR / "11_multi_image_deblurring_psnr_by_method.png",
    )
    save_metric_by_method_figure(
        metrics,
        metric_name="SSIM",
        ylabel="SSIM",
        path=FIGURES_DIR / "11_multi_image_deblurring_ssim_by_method.png",
    )
    save_metric_by_method_figure(
        metrics,
        metric_name="runtime_seconds",
        ylabel="Runtime seconds",
        path=FIGURES_DIR / "11_multi_image_deblurring_runtime_by_method.png",
        include_blurred_noisy=False,
        use_log_scale=True,
    )
    save_visual_grid(
        metrics,
        all_images,
        path=FIGURES_DIR / "11_multi_image_deblurring_visual_grid.png",
    )

    print("Multi-image deblurring comparison completed successfully.")
    print(f"Results saved to: {results_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
