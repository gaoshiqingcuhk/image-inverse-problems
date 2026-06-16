from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, img_as_float, img_as_ubyte, io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# First deblurring MVP: FFT-based Tikhonov deblurring.
#
# The blur and the inverse solver both use circular convolution, which
# corresponds to periodic boundary conditions. This keeps the experiment simple
# and makes the blur operator diagonal in the Fourier domain.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "sample_images"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

BLUR_KERNEL_SIZE = 21
BLUR_SIGMA = 2.0
NOISE_SIGMA = 0.01
RANDOM_SEED = 42
LAMBDAS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]


def ensure_output_folders() -> None:
    """Create folders used by this experiment."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a normalized 2D Gaussian blur kernel."""
    coordinates = np.arange(size) - size // 2
    xx, yy = np.meshgrid(coordinates, coordinates)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def psf_to_otf(kernel: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Convert a spatial blur kernel into an FFT frequency response.

    The small point spread function is placed into an image-sized array. Then
    it is rolled so the kernel center is at the FFT origin, which gives the
    frequency response for circular convolution.
    """
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
    """Add reproducible Gaussian noise and clip to [0, 1]."""
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
    """Deblur using FFT-based Tikhonov regularization.

    This solves

        min_x 0.5 * ||Kx - b||_2^2 + 0.5 * lam * ||grad x||_2^2

    with periodic boundary conditions. In the Fourier domain:

        x_hat = conj(H) * b_hat / (abs(H)**2 + lam * eig_lap).
    """
    b_hat = np.fft.fft2(blurred_noisy)
    eig_lap = laplacian_eigenvalues(blurred_noisy.shape)
    denominator = np.abs(H) ** 2 + lam * eig_lap
    x_hat = np.conj(H) * b_hat / denominator
    deblurred = np.fft.ifft2(x_hat).real
    return np.clip(deblurred, 0.0, 1.0)


def compute_metrics(clean: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM using the clean image as reference."""
    psnr = peak_signal_noise_ratio(clean, test, data_range=1.0)
    ssim = structural_similarity(clean, test, data_range=1.0)
    return psnr, ssim


def save_grayscale_png(image: np.ndarray, path: Path) -> None:
    """Save a float image in [0, 1] as an 8-bit grayscale PNG."""
    io.imsave(path, img_as_ubyte(np.clip(image, 0.0, 1.0)))


def add_result_row(
    rows: list[dict[str, float | str]],
    clean: np.ndarray,
    image: np.ndarray,
    method: str,
    parameter_name: str,
    parameter_value: float | str,
    runtime_seconds: float,
) -> None:
    """Compute metrics and append one result row."""
    psnr, ssim = compute_metrics(clean, image)
    rows.append(
        {
            "method": method,
            "blur_sigma": BLUR_SIGMA,
            "noise_sigma": NOISE_SIGMA,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "PSNR": psnr,
            "SSIM": ssim,
            "runtime_seconds": runtime_seconds,
        }
    )


def run_experiment(clean: np.ndarray, H: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Generate degradation, run Tikhonov deblurring, and collect results."""
    rows: list[dict[str, float | str]] = []
    images = {"clean": clean}

    blurred = circular_convolve_fft(clean, H)
    blurred_noisy = add_gaussian_noise(blurred)
    images["blurred_only"] = blurred
    images["blurred_noisy"] = blurred_noisy

    add_result_row(rows, clean, blurred, "blurred_only", "", "", 0.0)
    add_result_row(rows, clean, blurred_noisy, "blurred_noisy", "", "", 0.0)

    for lam in LAMBDAS:
        start_time = perf_counter()
        deblurred = tikhonov_deblur_fft(blurred_noisy, H, lam)
        runtime_seconds = perf_counter() - start_time
        images[f"tikhonov_lambda_{lam:g}"] = deblurred
        add_result_row(
            rows,
            clean,
            deblurred,
            "tikhonov_deblur",
            "lambda",
            lam,
            runtime_seconds,
        )

    return pd.DataFrame(rows), images


def save_metric_curve(metrics: pd.DataFrame, metric_name: str, ylabel: str, path: Path) -> None:
    """Save a lambda sensitivity curve with degraded-image baselines."""
    tikhonov_rows = metrics[metrics["method"] == "tikhonov_deblur"].copy()
    tikhonov_rows["parameter_value"] = tikhonov_rows["parameter_value"].astype(float)
    blurred_value = metrics.loc[metrics["method"] == "blurred_only", metric_name].iloc[0]
    blurred_noisy_value = metrics.loc[metrics["method"] == "blurred_noisy", metric_name].iloc[0]

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        tikhonov_rows["parameter_value"],
        tikhonov_rows[metric_name],
        marker="o",
        linewidth=2,
        label="Tikhonov deblur",
    )
    axis.axhline(blurred_value, color="gray", linestyle="--", linewidth=1.4, label="Blurred only")
    axis.axhline(
        blurred_noisy_value,
        color="tab:red",
        linestyle="--",
        linewidth=1.4,
        label="Blurred + noise",
    )
    axis.set_xscale("log")
    axis.set_xlabel("lambda")
    axis.set_ylabel(ylabel)
    axis.set_title(f"Tikhonov deblurring: {ylabel}")
    axis.grid(True, alpha=0.3, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def format_title(title: str, psnr: float, ssim: float) -> str:
    """Make compact subplot titles."""
    return f"{title}\nPSNR={psnr:.2f}, SSIM={ssim:.3f}"


def metric_row(metrics: pd.DataFrame, method: str, parameter_value: float | None = None) -> pd.Series:
    """Select a metric row by method and optional parameter value."""
    rows = metrics[metrics["method"] == method].copy()
    if parameter_value is not None:
        rows["parameter_value"] = rows["parameter_value"].astype(float)
        rows = rows[np.isclose(rows["parameter_value"], parameter_value)]
    return rows.iloc[0]


def save_visual_grid(metrics: pd.DataFrame, images: dict[str, np.ndarray], path: Path) -> None:
    """Save clean, degraded, and all Tikhonov deblurred results."""
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    flat_axes = axes.ravel()

    items = [
        ("clean", "Clean image", None),
        ("blurred_only", "Blurred", None),
        ("blurred_noisy", "Blurred + noise", None),
    ]
    items.extend((f"tikhonov_lambda_{lam:g}", f"Tikhonov lambda={lam:g}", lam) for lam in LAMBDAS)

    for axis, (image_key, title, lam) in zip(flat_axes, items):
        axis.imshow(images[image_key], cmap="gray", vmin=0.0, vmax=1.0)
        if image_key == "clean":
            axis.set_title(title)
        elif lam is None:
            row = metric_row(metrics, image_key)
            axis.set_title(format_title(title, row["PSNR"], row["SSIM"]))
        else:
            row = metric_row(metrics, "tikhonov_deblur", lam)
            axis.set_title(format_title(title, row["PSNR"], row["SSIM"]))
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_error_maps(clean: np.ndarray, metrics: pd.DataFrame, images: dict[str, np.ndarray], path: Path) -> None:
    """Save error maps for blurred noisy and best Tikhonov results."""
    tikhonov_rows = metrics[metrics["method"] == "tikhonov_deblur"].copy()
    tikhonov_rows["parameter_value"] = tikhonov_rows["parameter_value"].astype(float)
    best_psnr_row = tikhonov_rows.loc[tikhonov_rows["PSNR"].idxmax()]
    best_ssim_row = tikhonov_rows.loc[tikhonov_rows["SSIM"].idxmax()]
    best_psnr_lam = best_psnr_row["parameter_value"]
    best_ssim_lam = best_ssim_row["parameter_value"]

    items = [
        ("Blurred + noise", "blurred_noisy"),
        (f"Best PSNR\nlambda={best_psnr_lam:g}", f"tikhonov_lambda_{best_psnr_lam:g}"),
        (f"Best SSIM\nlambda={best_ssim_lam:g}", f"tikhonov_lambda_{best_ssim_lam:g}"),
    ]
    errors = [np.abs(clean - images[key]) for _, key in items]
    vmax = max(float(error.max()) for error in errors)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
    for axis, (title, _), error in zip(axes, items, errors):
        image = axis.imshow(error, cmap="inferno", vmin=0.0, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")

    fig.colorbar(image, ax=axes, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_folders()

    clean = img_as_float(data.camera())
    kernel = make_gaussian_kernel(BLUR_KERNEL_SIZE, BLUR_SIGMA)
    H = psf_to_otf(kernel, clean.shape)

    metrics, images = run_experiment(clean, H)

    results_path = RESULTS_DIR / "tikhonov_deblurring_results.csv"
    metrics.to_csv(results_path, index=False)

    save_grayscale_png(clean, DATA_DIR / "deblur_clean.png")
    save_grayscale_png(images["blurred_only"], DATA_DIR / "deblur_blurred_sigma_2.0.png")
    save_grayscale_png(
        images["blurred_noisy"],
        DATA_DIR / "deblur_blurred_noisy_sigma_0.01.png",
    )

    save_metric_curve(metrics, "PSNR", "PSNR", FIGURES_DIR / "tikhonov_deblurring_psnr.png")
    save_metric_curve(metrics, "SSIM", "SSIM", FIGURES_DIR / "tikhonov_deblurring_ssim.png")
    save_visual_grid(metrics, images, FIGURES_DIR / "tikhonov_deblurring_visual_grid.png")
    save_error_maps(clean, metrics, images, FIGURES_DIR / "tikhonov_deblurring_error_maps.png")

    print("Tikhonov deblurring experiment completed successfully.")
    print(f"Results saved to: {results_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
