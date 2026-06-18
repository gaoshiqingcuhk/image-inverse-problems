from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, filters, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle


# Total Variation denoising experiment.
#
# TV denoising balances closeness to the noisy observation with a penalty on
# image variation. In report notation, we can describe it as:
#
#     min_x 0.5 * ||x - b||_2^2 + lambda * ||grad x||_1
#
# In scikit-image's denoise_tv_chambolle implementation, the regularization
# parameter is named "weight". Larger weight values produce stronger denoising
# but lower fidelity to the noisy input image.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

NOISE_SIGMA = 0.10
RANDOM_SEED = 42
GAUSSIAN_FILTER_SIGMA = 1.0
TIKHONOV_LAMBDAS = [1.0, 5.0]
TV_WEIGHTS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]


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

    This uses periodic boundary conditions, where the standard discrete
    Laplacian eigenvalues are

        eig = 4 - 2*cos(2*pi*k/m) - 2*cos(2*pi*l/n).

    The Fourier-domain solution is

        x_hat = noisy_hat / (1 + lambda * eig).
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
    """Append one row to the experiment table."""
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


def evaluate_result(
    rows: list[dict[str, float | str]],
    clean: np.ndarray,
    image: np.ndarray,
    method: str,
    parameter_name: str,
    parameter_value: float | str,
    runtime_seconds: float,
) -> None:
    """Compute metrics for one result and append them to the table."""
    psnr, ssim = compute_metrics(clean, image)
    add_result_row(
        rows,
        method=method,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
        psnr=psnr,
        ssim=ssim,
        runtime_seconds=runtime_seconds,
    )


def run_experiment(clean: np.ndarray, noisy: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run baselines and TV-Chambolle denoising over several weights."""
    rows: list[dict[str, float | str]] = []
    images = {
        "clean": clean,
        "noisy_image": noisy,
    }

    evaluate_result(
        rows,
        clean,
        noisy,
        method="noisy_image",
        parameter_name="",
        parameter_value="",
        runtime_seconds=0.0,
    )

    start_time = perf_counter()
    gaussian_result = filters.gaussian(noisy, sigma=GAUSSIAN_FILTER_SIGMA)
    gaussian_runtime = perf_counter() - start_time
    images["gaussian_filter"] = gaussian_result
    evaluate_result(
        rows,
        clean,
        gaussian_result,
        method="gaussian_filter",
        parameter_name="filter_sigma",
        parameter_value=GAUSSIAN_FILTER_SIGMA,
        runtime_seconds=gaussian_runtime,
    )

    for lam in TIKHONOV_LAMBDAS:
        start_time = perf_counter()
        tikhonov_result = tikhonov_denoise_fft(noisy, lam)
        runtime_seconds = perf_counter() - start_time

        method = f"tikhonov_lambda_{lam:.1f}"
        images[method] = tikhonov_result
        evaluate_result(
            rows,
            clean,
            tikhonov_result,
            method=method,
            parameter_name="lambda",
            parameter_value=lam,
            runtime_seconds=runtime_seconds,
        )

    for weight in TV_WEIGHTS:
        start_time = perf_counter()
        tv_result = denoise_tv_chambolle(
            noisy,
            weight=weight,
            channel_axis=None,
        )
        runtime_seconds = perf_counter() - start_time
        tv_result = np.clip(tv_result, 0.0, 1.0)

        method = f"tv_chambolle_weight_{weight:g}"
        images[method] = tv_result
        evaluate_result(
            rows,
            clean,
            tv_result,
            method="tv_chambolle",
            parameter_name="weight",
            parameter_value=weight,
            runtime_seconds=runtime_seconds,
        )

    return pd.DataFrame(rows), images


def save_tv_metric_curve(
    metrics: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    path: Path,
) -> None:
    """Save a TV weight sensitivity curve with reference baselines."""
    tv_results = metrics[metrics["method"] == "tv_chambolle"].copy()
    tv_results["parameter_value"] = tv_results["parameter_value"].astype(float)

    references = [
        ("noisy_image", "Noisy image", "gray"),
        ("gaussian_filter", "Gaussian filter", "tab:green"),
        ("tikhonov_lambda_1.0", "Tikhonov lambda=1.0", "tab:orange"),
        ("tikhonov_lambda_5.0", "Tikhonov lambda=5.0", "tab:purple"),
    ]

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        tv_results["parameter_value"],
        tv_results[metric_name],
        marker="o",
        linewidth=2,
        label="TV-Chambolle",
    )

    for method, label, color in references:
        value = metrics.loc[metrics["method"] == method, metric_name].iloc[0]
        axis.axhline(
            value,
            linestyle="--",
            linewidth=1.4,
            color=color,
            label=label,
        )

    axis.set_xlabel("TV weight")
    axis.set_ylabel(ylabel)
    axis.set_title(f"TV weight sensitivity: {ylabel}")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def row_for_method(metrics: pd.DataFrame, method: str) -> pd.Series:
    """Return the first row matching a method name."""
    return metrics[metrics["method"] == method].iloc[0]


def row_for_tv_weight(metrics: pd.DataFrame, weight: float) -> pd.Series:
    """Return the TV row for a specific weight."""
    tv_rows = metrics[metrics["method"] == "tv_chambolle"].copy()
    tv_rows["parameter_value"] = tv_rows["parameter_value"].astype(float)
    return tv_rows[np.isclose(tv_rows["parameter_value"], weight)].iloc[0]


def format_title(title: str, psnr: float, ssim: float) -> str:
    """Make a compact subplot title."""
    return f"{title}\nPSNR={psnr:.2f}, SSIM={ssim:.3f}"


def save_visual_grid(
    metrics: pd.DataFrame,
    images: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Save clean, baselines, Tikhonov, and all TV results."""
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 10))
    flat_axes = axes.ravel()

    noisy_row = row_for_method(metrics, "noisy_image")
    gaussian_row = row_for_method(metrics, "gaussian_filter")
    tikhonov_1_row = row_for_method(metrics, "tikhonov_lambda_1.0")
    tikhonov_5_row = row_for_method(metrics, "tikhonov_lambda_5.0")

    images_and_titles = [
        (images["clean"], "Clean image"),
        (
            images["noisy_image"],
            format_title("Noisy image", noisy_row["PSNR"], noisy_row["SSIM"]),
        ),
        (
            images["gaussian_filter"],
            format_title("Gaussian filter", gaussian_row["PSNR"], gaussian_row["SSIM"]),
        ),
        (
            images["tikhonov_lambda_1.0"],
            format_title("Tikhonov lambda=1.0", tikhonov_1_row["PSNR"], tikhonov_1_row["SSIM"]),
        ),
        (
            images["tikhonov_lambda_5.0"],
            format_title("Tikhonov lambda=5.0", tikhonov_5_row["PSNR"], tikhonov_5_row["SSIM"]),
        ),
    ]

    for weight in TV_WEIGHTS:
        row = row_for_tv_weight(metrics, weight)
        images_and_titles.append(
            (
                images[f"tv_chambolle_weight_{weight:g}"],
                format_title(f"TV weight={weight:g}", row["PSNR"], row["SSIM"]),
            )
        )

    for axis, (image, title) in zip(flat_axes, images_and_titles):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    for axis in flat_axes[len(images_and_titles) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_error_maps(
    clean: np.ndarray,
    metrics: pd.DataFrame,
    images: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Save error maps for key baseline and best TV results."""
    tv_results = metrics[metrics["method"] == "tv_chambolle"].copy()
    tv_results["parameter_value"] = tv_results["parameter_value"].astype(float)

    best_psnr_row = tv_results.loc[tv_results["PSNR"].idxmax()]
    best_ssim_row = tv_results.loc[tv_results["SSIM"].idxmax()]
    best_psnr_weight = best_psnr_row["parameter_value"]
    best_ssim_weight = best_ssim_row["parameter_value"]

    error_items = [
        ("Gaussian filter", "gaussian_filter"),
        ("Tikhonov lambda=1.0", "tikhonov_lambda_1.0"),
        ("Tikhonov lambda=5.0", "tikhonov_lambda_5.0"),
        (
            f"Best TV by PSNR\nweight={best_psnr_weight:g}",
            f"tv_chambolle_weight_{best_psnr_weight:g}",
        ),
        (
            f"Best TV by SSIM\nweight={best_ssim_weight:g}",
            f"tv_chambolle_weight_{best_ssim_weight:g}",
        ),
    ]

    error_maps = [np.abs(clean - images[key]) for _, key in error_items]
    vmax = max(float(error.max()) for error in error_maps)

    fig, axes = plt.subplots(nrows=1, ncols=len(error_items), figsize=(15, 3.5))
    for axis, (title, _), error in zip(axes, error_items, error_maps):
        image = axis.imshow(error, cmap="inferno", vmin=0.0, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")

    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_folders()

    clean = img_as_float(data.camera())
    noisy = add_gaussian_noise(clean)

    metrics, images = run_experiment(clean, noisy)

    results_path = RESULTS_DIR / "06_tv_denoising_results.csv"
    metrics.to_csv(results_path, index=False)

    save_tv_metric_curve(
        metrics,
        metric_name="PSNR",
        ylabel="PSNR",
        path=FIGURES_DIR / "06_tv_weight_sensitivity_psnr.png",
    )
    save_tv_metric_curve(
        metrics,
        metric_name="SSIM",
        ylabel="SSIM",
        path=FIGURES_DIR / "06_tv_weight_sensitivity_ssim.png",
    )
    save_visual_grid(
        metrics,
        images,
        path=FIGURES_DIR / "06_tv_denoising_visual_grid.png",
    )
    save_error_maps(
        clean,
        metrics,
        images,
        path=FIGURES_DIR / "06_tv_error_maps.png",
    )

    print("Total Variation denoising experiment completed successfully.")
    print(f"Results saved to: {results_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
