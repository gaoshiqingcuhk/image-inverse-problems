from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import richardson_lucy


# Phase 6: Richardson-Lucy deblurring baseline.
#
# This experiment uses the same synthetic blur/noise setting as the previous
# single-image deblurring experiments. Tikhonov and Wiener baselines are kept
# fixed, while Richardson-Lucy is tested over several iteration counts.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

BLUR_SIGMA = 2.0
PSF_SIZE = 21
NOISE_SIGMA = 0.01
RANDOM_SEED = 42

TIKHONOV_LAMBDAS = [0.01, 0.05]
WIENER_BALANCES = [0.01, 0.03]
RL_ITERATIONS = [5, 10, 20, 30, 50]

METHOD_LABELS = {
    "blurred_noisy": "Blurred + noise",
    "tikhonov_deblur_lambda_0.01": "Tikhonov 0.01",
    "tikhonov_deblur_lambda_0.05": "Tikhonov 0.05",
    "wiener_deblur_balance_0.01": "Wiener 0.01",
    "wiener_deblur_balance_0.03": "Wiener 0.03",
}


def ensure_output_folders() -> None:
    """Create folders for saved CSV files and figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """Create a normalized 2D Gaussian point spread function."""
    coordinates = np.arange(size) - size // 2
    xx, yy = np.meshgrid(coordinates, coordinates)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return psf / psf.sum()


def psf_to_otf(psf: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Convert a small PSF into an FFT response for circular convolution."""
    padded = np.zeros(image_shape, dtype=float)
    rows, cols = psf.shape
    padded[:rows, :cols] = psf
    padded = np.roll(padded, shift=-(rows // 2), axis=0)
    padded = np.roll(padded, shift=-(cols // 2), axis=1)
    return np.fft.fft2(padded)


def circular_convolve_fft(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply circular convolution using the FFT frequency response H."""
    blurred = np.fft.ifft2(np.fft.fft2(image) * H).real
    return np.clip(blurred, 0.0, 1.0)


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    """Add deterministic Gaussian noise and clip to [0, 1]."""
    rng = np.random.default_rng(RANDOM_SEED)
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
    """Deblur with FFT-based Tikhonov regularization."""
    b_hat = np.fft.fft2(observed)
    eig_lap = laplacian_eigenvalues(observed.shape)
    denominator = np.abs(H) ** 2 + lam * eig_lap
    x_hat = np.conj(H) * b_hat / denominator
    restored = np.fft.ifft2(x_hat).real
    return np.clip(restored, 0.0, 1.0)


def wiener_deblur_fft(observed: np.ndarray, H: np.ndarray, balance: float) -> np.ndarray:
    """Deblur with an FFT Wiener inverse-filter baseline."""
    b_hat = np.fft.fft2(observed)
    denominator = np.abs(H) ** 2 + balance
    x_hat = np.conj(H) * b_hat / denominator
    restored = np.fft.ifft2(x_hat).real
    return np.clip(restored, 0.0, 1.0)


def richardson_lucy_compat(
    observed: np.ndarray,
    psf: np.ndarray,
    num_iter: int,
) -> np.ndarray:
    """Run Richardson-Lucy with compatibility for scikit-image API versions."""
    try:
        restored = richardson_lucy(
            observed,
            psf,
            num_iter=num_iter,
            clip=False,
        )
    except TypeError:
        restored = richardson_lucy(
            observed,
            psf,
            iterations=num_iter,
            clip=False,
        )
    return np.clip(restored, 0.0, 1.0)


def compute_metrics(clean: np.ndarray, restored: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM using the clean image as reference."""
    psnr = peak_signal_noise_ratio(clean, restored, data_range=1.0)
    ssim = structural_similarity(clean, restored, data_range=1.0)
    return psnr, ssim


def add_result_row(
    rows: list[dict[str, float | str]],
    clean: np.ndarray,
    restored: np.ndarray,
    method: str,
    parameter: str,
    runtime_seconds: float,
) -> None:
    """Compute metrics and append one CSV row."""
    psnr, ssim = compute_metrics(clean, restored)
    rows.append(
        {
            "method": method,
            "parameter": parameter,
            "PSNR": psnr,
            "SSIM": ssim,
            "runtime_seconds": runtime_seconds,
        }
    )


def run_experiment(
    clean: np.ndarray,
    psf: np.ndarray,
    H: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Create degradation, run baselines, and collect results."""
    rows: list[dict[str, float | str]] = []
    images: dict[str, np.ndarray] = {"clean": clean}

    blurred = circular_convolve_fft(clean, H)
    blurred_noisy = add_gaussian_noise(blurred)
    images["blurred_noisy"] = blurred_noisy
    add_result_row(rows, clean, blurred_noisy, "blurred_noisy", "-", 0.0)

    for lam in TIKHONOV_LAMBDAS:
        start_time = perf_counter()
        restored = tikhonov_deblur_fft(blurred_noisy, H, lam)
        runtime_seconds = perf_counter() - start_time
        key = f"tikhonov_deblur_lambda_{lam:g}"
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
        key = f"wiener_deblur_balance_{balance:g}"
        images[key] = restored
        add_result_row(
            rows,
            clean,
            restored,
            "wiener_deblur",
            f"balance={balance:g}",
            runtime_seconds,
        )

    for num_iter in RL_ITERATIONS:
        start_time = perf_counter()
        restored = richardson_lucy_compat(blurred_noisy, psf, num_iter)
        runtime_seconds = perf_counter() - start_time
        key = f"richardson_lucy_iter_{num_iter}"
        images[key] = restored
        add_result_row(
            rows,
            clean,
            restored,
            "richardson_lucy",
            f"num_iter={num_iter}",
            runtime_seconds,
        )

    return pd.DataFrame(rows), images


def label_for_row(row: pd.Series) -> str:
    """Create a compact x-axis label for a result row."""
    if row["method"] == "blurred_noisy":
        return "Blurred\nnoisy"
    if row["method"] == "tikhonov_deblur":
        return row["parameter"].replace("lambda=", "Tik\n")
    if row["method"] == "wiener_deblur":
        return row["parameter"].replace("balance=", "Wie\n")
    return row["parameter"].replace("num_iter=", "RL\n")


def save_bar_figure(
    metrics: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    path: Path,
    include_degraded: bool = True,
    use_log_scale: bool = False,
) -> None:
    """Save a bar chart for PSNR, SSIM, or runtime."""
    plot_rows = metrics.copy()
    if not include_degraded:
        plot_rows = plot_rows[plot_rows["method"] != "blurred_noisy"]

    labels = [label_for_row(row) for _, row in plot_rows.iterrows()]
    values = plot_rows[metric_name].to_numpy()

    fig, axis = plt.subplots(figsize=(9, 4.6), facecolor="white")
    axis.bar(labels, values)
    axis.set_ylabel(ylabel)
    axis.set_title(f"Richardson-Lucy deblurring comparison: {ylabel}")
    axis.grid(True, axis="y", alpha=0.25)
    axis.tick_params(axis="x", labelrotation=0)

    if use_log_scale:
        axis.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def row_for(metrics: pd.DataFrame, method: str, parameter: str | None = None) -> pd.Series:
    """Select one metric row by method and optional parameter."""
    rows = metrics[metrics["method"] == method]
    if parameter is not None:
        rows = rows[rows["parameter"] == parameter]
    return rows.iloc[0]


def best_rl_rows(metrics: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return best Richardson-Lucy rows by PSNR and SSIM."""
    rl_rows = metrics[metrics["method"] == "richardson_lucy"]
    best_psnr = rl_rows.loc[rl_rows["PSNR"].idxmax()]
    best_ssim = rl_rows.loc[rl_rows["SSIM"].idxmax()]
    return best_psnr, best_ssim


def format_title(title: str, psnr: float, ssim: float) -> str:
    """Create compact subplot titles."""
    return f"{title}\nPSNR={psnr:.2f}, SSIM={ssim:.3f}"


def image_key_for(method: str, parameter: str) -> str:
    """Return the image dictionary key for a method/parameter pair."""
    if method == "blurred_noisy":
        return "blurred_noisy"
    value = parameter.split("=", maxsplit=1)[1]
    if method == "tikhonov_deblur":
        return f"tikhonov_deblur_lambda_{value}"
    if method == "wiener_deblur":
        return f"wiener_deblur_balance_{value}"
    return f"richardson_lucy_iter_{value}"


def save_visual_grid(metrics: pd.DataFrame, images: dict[str, np.ndarray], path: Path) -> None:
    """Save clean, degraded, baseline, and best RL visual results."""
    best_psnr_row, best_ssim_row = best_rl_rows(metrics)

    items: list[tuple[str, str, str, str | None]] = [
        ("clean", "Clean image", "clean", None),
        ("blurred_noisy", "Blurred + noise", "blurred_noisy", "-"),
        ("tikhonov_deblur_lambda_0.01", "Tikhonov lambda=0.01", "tikhonov_deblur", "lambda=0.01"),
        ("tikhonov_deblur_lambda_0.05", "Tikhonov lambda=0.05", "tikhonov_deblur", "lambda=0.05"),
        ("wiener_deblur_balance_0.01", "Wiener balance=0.01", "wiener_deblur", "balance=0.01"),
        ("wiener_deblur_balance_0.03", "Wiener balance=0.03", "wiener_deblur", "balance=0.03"),
    ]

    best_key = image_key_for("richardson_lucy", best_psnr_row["parameter"])
    title = f"Best RL PSNR\n{best_psnr_row['parameter']}"
    if best_psnr_row["parameter"] == best_ssim_row["parameter"]:
        title = f"Best RL PSNR/SSIM\n{best_psnr_row['parameter']}"
    items.append((best_key, title, "richardson_lucy", best_psnr_row["parameter"]))

    if best_psnr_row["parameter"] != best_ssim_row["parameter"]:
        best_key = image_key_for("richardson_lucy", best_ssim_row["parameter"])
        items.append(
            (
                best_key,
                f"Best RL SSIM\n{best_ssim_row['parameter']}",
                "richardson_lucy",
                best_ssim_row["parameter"],
            )
        )

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 7), facecolor="white")
    flat_axes = axes.ravel()

    for axis, (image_key, title, method, parameter) in zip(flat_axes, items):
        axis.imshow(images[image_key], cmap="gray", vmin=0.0, vmax=1.0)
        if method == "clean":
            axis.set_title(title)
        else:
            row = row_for(metrics, method, parameter)
            axis.set_title(format_title(title, row["PSNR"], row["SSIM"]), fontsize=9)
        axis.axis("off")

    for axis in flat_axes[len(items) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_error_maps(clean: np.ndarray, metrics: pd.DataFrame, images: dict[str, np.ndarray], path: Path) -> None:
    """Save absolute error maps for degraded, baselines, and best RL results."""
    best_psnr_row, best_ssim_row = best_rl_rows(metrics)
    rows = [
        row_for(metrics, "blurred_noisy", "-"),
        row_for(metrics, "tikhonov_deblur", "lambda=0.01"),
        row_for(metrics, "tikhonov_deblur", "lambda=0.05"),
        row_for(metrics, "wiener_deblur", "balance=0.01"),
        row_for(metrics, "wiener_deblur", "balance=0.03"),
        best_psnr_row,
    ]
    titles = [
        "Blurred + noise",
        "Tikhonov\nlambda=0.01",
        "Tikhonov\nlambda=0.05",
        "Wiener\nbalance=0.01",
        "Wiener\nbalance=0.03",
        f"Best RL PSNR\n{best_psnr_row['parameter']}",
    ]
    if best_psnr_row["parameter"] != best_ssim_row["parameter"]:
        rows.append(best_ssim_row)
        titles.append(f"Best RL SSIM\n{best_ssim_row['parameter']}")

    keys = [image_key_for(row["method"], row["parameter"]) for row in rows]
    errors = [np.abs(clean - images[key]) for key in keys]
    vmax = max(float(error.max()) for error in errors)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(errors),
        figsize=(3.0 * len(errors), 3.4),
        facecolor="white",
    )
    if len(errors) == 1:
        axes = np.array([axes])

    for axis, title, error in zip(axes, titles, errors):
        image = axis.imshow(error, cmap="inferno", vmin=0.0, vmax=vmax)
        axis.set_title(title, fontsize=9)
        axis.axis("off")

    fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_folders()

    clean = img_as_float(data.camera())
    psf = gaussian_psf(PSF_SIZE, BLUR_SIGMA)
    H = psf_to_otf(psf, clean.shape)
    metrics, images = run_experiment(clean, psf, H)

    results_path = RESULTS_DIR / "14_richardson_lucy_deblurring_results.csv"
    metrics.to_csv(results_path, index=False)

    save_bar_figure(
        metrics,
        "PSNR",
        "PSNR",
        FIGURES_DIR / "14_richardson_lucy_deblurring_psnr.png",
    )
    save_bar_figure(
        metrics,
        "SSIM",
        "SSIM",
        FIGURES_DIR / "14_richardson_lucy_deblurring_ssim.png",
    )
    save_bar_figure(
        metrics,
        "runtime_seconds",
        "Runtime seconds",
        FIGURES_DIR / "14_richardson_lucy_deblurring_runtime.png",
        include_degraded=False,
        use_log_scale=True,
    )
    save_visual_grid(
        metrics,
        images,
        FIGURES_DIR / "14_richardson_lucy_deblurring_visual_grid.png",
    )
    save_error_maps(
        clean,
        metrics,
        images,
        FIGURES_DIR / "14_richardson_lucy_deblurring_error_maps.png",
    )

    best_psnr_row, best_ssim_row = best_rl_rows(metrics)

    print("Richardson-Lucy deblurring experiment completed successfully.")
    print(f"Results saved to: {results_path}")
    print(f"Best RL by PSNR: {best_psnr_row['parameter']} ({best_psnr_row['PSNR']:.6f})")
    print(f"Best RL by SSIM: {best_ssim_row['parameter']} ({best_ssim_row['SSIM']:.6f})")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
