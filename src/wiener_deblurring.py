from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# Phase 3: Wiener deblurring baseline.
#
# The blur model, Tikhonov comparison solver, and Wiener solver all use FFTs
# with circular convolution. This corresponds to periodic boundary conditions
# and keeps the degradation setting consistent with tikhonov_deblurring.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

BLUR_KERNEL_SIZE = 21
BLUR_SIGMA = 2.0
NOISE_SIGMA = 0.01
RANDOM_SEED = 42
TIKHONOV_LAMBDAS = [0.01, 0.05]
WIENER_BALANCES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def ensure_output_folders() -> None:
    """Create folders used by this experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a normalized 2D Gaussian blur kernel."""
    coordinates = np.arange(size) - size // 2
    xx, yy = np.meshgrid(coordinates, coordinates)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def psf_to_otf(kernel: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Convert a small spatial blur kernel into an FFT frequency response."""
    padded = np.zeros(image_shape, dtype=float)
    kernel_rows, kernel_cols = kernel.shape
    padded[:kernel_rows, :kernel_cols] = kernel
    padded = np.roll(padded, shift=-(kernel_rows // 2), axis=0)
    padded = np.roll(padded, shift=-(kernel_cols // 2), axis=1)
    return np.fft.fft2(padded)


def circular_convolve_fft(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply circular convolution using the blur frequency response H."""
    blurred = np.fft.ifft2(np.fft.fft2(image) * H).real
    return np.clip(blurred, 0.0, 1.0)


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    """Add reproducible Gaussian noise and clip the result to [0, 1]."""
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def laplacian_eigenvalues(image_shape: tuple[int, int]) -> np.ndarray:
    """Return periodic discrete Laplacian eigenvalues for an image shape."""
    rows, cols = image_shape
    row_freq = np.arange(rows).reshape(rows, 1)
    col_freq = np.arange(cols).reshape(1, cols)
    return (
        4.0
        - 2.0 * np.cos(2.0 * np.pi * row_freq / rows)
        - 2.0 * np.cos(2.0 * np.pi * col_freq / cols)
    )


def tikhonov_deblur_fft(blurred_noisy: np.ndarray, H: np.ndarray, lam: float) -> np.ndarray:
    """Deblur with FFT-based Tikhonov regularization."""
    b_hat = np.fft.fft2(blurred_noisy)
    eig_lap = laplacian_eigenvalues(blurred_noisy.shape)
    denominator = np.abs(H) ** 2 + lam * eig_lap
    x_hat = np.conj(H) * b_hat / denominator
    deblurred = np.fft.ifft2(x_hat).real
    return np.clip(deblurred, 0.0, 1.0)


def wiener_deblur_fft(blurred_noisy: np.ndarray, H: np.ndarray, balance: float) -> np.ndarray:
    """Deblur with a simple FFT Wiener inverse filter.

    The formula

        X_hat = conj(H) * B / (abs(H)^2 + balance)

    stabilizes direct inverse filtering by avoiding division by very small
    blur frequencies. Smaller balance values are more aggressive but can
    amplify noise and ringing artifacts.
    """
    b_hat = np.fft.fft2(blurred_noisy)
    denominator = np.abs(H) ** 2 + balance
    x_hat = np.conj(H) * b_hat / denominator
    deblurred = np.fft.ifft2(x_hat).real
    return np.clip(deblurred, 0.0, 1.0)


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
) -> None:
    """Compute metrics and append one CSV result row."""
    psnr, ssim = compute_metrics(clean, image)
    rows.append(
        {
            "method": method,
            "parameter": parameter,
            "PSNR": psnr,
            "SSIM": ssim,
            "runtime_seconds": runtime_seconds,
        }
    )


def run_experiment(clean: np.ndarray, H: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Create degradation, run baselines, and collect metrics."""
    rows: list[dict[str, float | str]] = []
    images: dict[str, np.ndarray] = {"clean": clean}

    blurred = circular_convolve_fft(clean, H)
    blurred_noisy = add_gaussian_noise(blurred)
    images["blurred_only"] = blurred
    images["blurred_noisy"] = blurred_noisy

    add_result_row(rows, clean, blurred, "blurred_only", "-", 0.0)
    add_result_row(rows, clean, blurred_noisy, "blurred_noisy", "-", 0.0)

    for lam in TIKHONOV_LAMBDAS:
        start_time = perf_counter()
        restored = tikhonov_deblur_fft(blurred_noisy, H, lam)
        runtime_seconds = perf_counter() - start_time
        key = f"tikhonov_lambda_{lam:g}"
        images[key] = restored
        add_result_row(
            rows,
            clean,
            restored,
            "tikhonov_deblur",
            f"lambda={lam:g}",
            runtime_seconds,
        )

    for balance in WIENER_BALANCES:
        start_time = perf_counter()
        restored = wiener_deblur_fft(blurred_noisy, H, balance)
        runtime_seconds = perf_counter() - start_time
        key = f"wiener_balance_{balance:g}"
        images[key] = restored
        add_result_row(
            rows,
            clean,
            restored,
            "wiener_deblur",
            f"balance={balance:g}",
            runtime_seconds,
        )

    return pd.DataFrame(rows), images


def parse_parameter_value(parameter: str) -> float:
    """Extract the numeric value from strings like 'balance=0.001'."""
    return float(parameter.split("=", maxsplit=1)[1])


def save_metric_curve(metrics: pd.DataFrame, metric_name: str, path: Path) -> None:
    """Save a Wiener balance sensitivity curve with Tikhonov references."""
    wiener_rows = metrics[metrics["method"] == "wiener_deblur"].copy()
    wiener_rows["balance"] = wiener_rows["parameter"].map(parse_parameter_value)
    tikhonov_rows = metrics[metrics["method"] == "tikhonov_deblur"].copy()
    blurred_noisy_value = metrics.loc[metrics["method"] == "blurred_noisy", metric_name].iloc[0]

    fig, axis = plt.subplots(figsize=(6.5, 4.2), facecolor="white")
    axis.plot(
        wiener_rows["balance"],
        wiener_rows[metric_name],
        marker="o",
        linewidth=2,
        label="Wiener deblur",
    )
    axis.axhline(
        blurred_noisy_value,
        color="gray",
        linestyle="--",
        linewidth=1.4,
        label="Blurred + noise",
    )

    for _, row in tikhonov_rows.iterrows():
        axis.axhline(
            row[metric_name],
            linestyle=":",
            linewidth=1.6,
            label=f"Tikhonov {row['parameter']}",
        )

    axis.set_xscale("log")
    axis.set_xlabel("Wiener balance")
    axis.set_ylabel(metric_name)
    axis.set_title(f"Wiener deblurring: {metric_name}")
    axis.grid(True, alpha=0.3, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def format_title(title: str, psnr: float, ssim: float) -> str:
    """Make compact subplot titles."""
    return f"{title}\nPSNR={psnr:.2f}, SSIM={ssim:.3f}"


def metric_row(metrics: pd.DataFrame, method: str, parameter: str | None = None) -> pd.Series:
    """Select a metric row by method and optional parameter string."""
    rows = metrics[metrics["method"] == method]
    if parameter is not None:
        rows = rows[rows["parameter"] == parameter]
    return rows.iloc[0]


def best_wiener_rows(metrics: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the best Wiener rows by PSNR and SSIM."""
    wiener_rows = metrics[metrics["method"] == "wiener_deblur"]
    best_psnr = wiener_rows.loc[wiener_rows["PSNR"].idxmax()]
    best_ssim = wiener_rows.loc[wiener_rows["SSIM"].idxmax()]
    return best_psnr, best_ssim


def save_visual_grid(metrics: pd.DataFrame, images: dict[str, np.ndarray], path: Path) -> None:
    """Save a visual comparison of degraded, Tikhonov, and Wiener results."""
    best_psnr_row, best_ssim_row = best_wiener_rows(metrics)
    best_items = [
        (f"Best Wiener PSNR\n{best_psnr_row['parameter']}", best_psnr_row["parameter"]),
    ]
    if best_ssim_row["parameter"] != best_psnr_row["parameter"]:
        best_items.append((f"Best Wiener SSIM\n{best_ssim_row['parameter']}", best_ssim_row["parameter"]))
    else:
        best_items[0] = (
            f"Best Wiener PSNR/SSIM\n{best_psnr_row['parameter']}",
            best_psnr_row["parameter"],
        )

    items: list[tuple[str, str, str, str | None]] = [
        ("clean", "Clean image", "clean", None),
        ("blurred_noisy", "Blurred + noise", "blurred_noisy", None),
        ("tikhonov_lambda_0.01", "Tikhonov lambda=0.01", "tikhonov_deblur", "lambda=0.01"),
        ("tikhonov_lambda_0.05", "Tikhonov lambda=0.05", "tikhonov_deblur", "lambda=0.05"),
    ]
    for title, parameter in best_items:
        balance = parse_parameter_value(parameter)
        items.append((f"wiener_balance_{balance:g}", title, "wiener_deblur", parameter))

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 7), facecolor="white")
    flat_axes = axes.ravel()

    for axis, (image_key, title, method, parameter) in zip(flat_axes, items):
        axis.imshow(images[image_key], cmap="gray", vmin=0.0, vmax=1.0)
        if method == "clean":
            axis.set_title(title)
        else:
            row = metric_row(metrics, method, parameter)
            axis.set_title(format_title(title, row["PSNR"], row["SSIM"]))
        axis.axis("off")

    for axis in flat_axes[len(items) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_error_maps(clean: np.ndarray, metrics: pd.DataFrame, images: dict[str, np.ndarray], path: Path) -> None:
    """Save absolute error maps for degraded, Tikhonov, and best Wiener results."""
    best_psnr_row, best_ssim_row = best_wiener_rows(metrics)
    items = [
        ("Blurred + noise", "blurred_noisy"),
        ("Tikhonov\nlambda=0.01", "tikhonov_lambda_0.01"),
        ("Tikhonov\nlambda=0.05", "tikhonov_lambda_0.05"),
    ]

    best_psnr_balance = parse_parameter_value(best_psnr_row["parameter"])
    items.append((f"Best Wiener PSNR\n{best_psnr_row['parameter']}", f"wiener_balance_{best_psnr_balance:g}"))
    if best_ssim_row["parameter"] != best_psnr_row["parameter"]:
        best_ssim_balance = parse_parameter_value(best_ssim_row["parameter"])
        items.append((f"Best Wiener SSIM\n{best_ssim_row['parameter']}", f"wiener_balance_{best_ssim_balance:g}"))

    errors = [np.abs(clean - images[key]) for _, key in items]
    vmax = max(float(error.max()) for error in errors)

    fig, axes = plt.subplots(nrows=1, ncols=len(items), figsize=(3.2 * len(items), 3.5), facecolor="white")
    if len(items) == 1:
        axes = np.array([axes])

    for axis, (title, _), error in zip(axes, items, errors):
        image = axis.imshow(error, cmap="inferno", vmin=0.0, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")

    fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_folders()

    clean = img_as_float(data.camera())
    kernel = make_gaussian_kernel(BLUR_KERNEL_SIZE, BLUR_SIGMA)
    H = psf_to_otf(kernel, clean.shape)

    metrics, images = run_experiment(clean, H)

    results_path = RESULTS_DIR / "10_wiener_deblurring_results.csv"
    metrics.to_csv(results_path, index=False)

    save_metric_curve(metrics, "PSNR", FIGURES_DIR / "10_wiener_deblurring_psnr.png")
    save_metric_curve(metrics, "SSIM", FIGURES_DIR / "10_wiener_deblurring_ssim.png")
    save_visual_grid(metrics, images, FIGURES_DIR / "10_wiener_deblurring_visual_grid.png")
    save_error_maps(clean, metrics, images, FIGURES_DIR / "10_wiener_deblurring_error_maps.png")

    best_psnr_row, best_ssim_row = best_wiener_rows(metrics)

    print("Wiener deblurring experiment completed successfully.")
    print(f"Results saved to: {results_path}")
    print(f"Best Wiener by PSNR: {best_psnr_row['parameter']} ({best_psnr_row['PSNR']:.6f})")
    print(f"Best Wiener by SSIM: {best_ssim_row['parameter']} ({best_ssim_row['SSIM']:.6f})")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
