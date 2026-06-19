from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, filters, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, richardson_lucy


# Phase 7 / v0.8: degradation robustness study.
#
# This script does not introduce new algorithms. It reuses fixed restoration
# settings from earlier phases and tests whether method rankings change when
# the degradation strength changes.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

DENOISING_NOISE_SIGMAS = [0.05, 0.10, 0.20]
DEBLURRING_BLUR_SIGMAS = [1.0, 2.0, 3.0]
DEBLURRING_NOISE_SIGMA = 0.01

GAUSSIAN_FILTER_SIGMA = 1.0
TIKHONOV_DENOISING_LAMBDA = 1.0
TV_WEIGHT = 0.1
NLM_H = 0.10
NLM_PATCH_SIZE = 5
NLM_PATCH_DISTANCE = 6
NLM_FAST_MODE = True

PSF_SIZE = 21
TIKHONOV_DEBLUR_LAMBDAS = [0.01, 0.05]
WIENER_BALANCES = [0.01, 0.03]
RL_ITERATIONS = [5, 10]


def ensure_output_folders() -> None:
    """Create folders for saved results and figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Add deterministic Gaussian noise and clip to [0, 1]."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def compute_metrics(clean: np.ndarray, restored: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM with a fixed data range."""
    psnr = peak_signal_noise_ratio(clean, restored, data_range=1.0)
    ssim = structural_similarity(clean, restored, data_range=1.0)
    return psnr, ssim


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


def tikhonov_denoise_fft(noisy: np.ndarray, lam: float) -> np.ndarray:
    """Denoise with FFT-based Tikhonov regularization."""
    eig = laplacian_eigenvalues(noisy.shape)
    noisy_hat = np.fft.fft2(noisy)
    restored_hat = noisy_hat / (1.0 + lam * eig)
    restored = np.fft.ifft2(restored_hat).real
    return np.clip(restored, 0.0, 1.0)


def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """Create a normalized Gaussian point spread function."""
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
    """Apply circular convolution using an FFT frequency response."""
    blurred = np.fft.ifft2(np.fft.fft2(image) * H).real
    return np.clip(blurred, 0.0, 1.0)


def tikhonov_deblur_fft(observed: np.ndarray, H: np.ndarray, lam: float) -> np.ndarray:
    """Deblur with FFT-based Tikhonov regularization."""
    b_hat = np.fft.fft2(observed)
    eig_lap = laplacian_eigenvalues(observed.shape)
    denominator = np.abs(H) ** 2 + lam * eig_lap
    x_hat = np.conj(H) * b_hat / denominator
    restored = np.fft.ifft2(x_hat).real
    return np.clip(restored, 0.0, 1.0)


def wiener_deblur_fft(observed: np.ndarray, H: np.ndarray, balance: float) -> np.ndarray:
    """Deblur with the FFT Wiener inverse-filter baseline."""
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


def append_denoising_row(
    rows: list[dict[str, float | str]],
    noise_sigma: float,
    method: str,
    parameter: str,
    clean: np.ndarray,
    restored: np.ndarray,
    runtime_seconds: float,
) -> None:
    """Compute metrics and append one denoising result row."""
    psnr, ssim = compute_metrics(clean, restored)
    rows.append(
        {
            "noise_sigma": noise_sigma,
            "method": method,
            "parameter": parameter,
            "PSNR": psnr,
            "SSIM": ssim,
            "runtime_seconds": runtime_seconds,
        }
    )


def append_deblurring_row(
    rows: list[dict[str, float | str]],
    blur_sigma: float,
    noise_sigma: float,
    method: str,
    parameter: str,
    clean: np.ndarray,
    restored: np.ndarray,
    runtime_seconds: float,
) -> None:
    """Compute metrics and append one deblurring result row."""
    psnr, ssim = compute_metrics(clean, restored)
    rows.append(
        {
            "blur_sigma": blur_sigma,
            "noise_sigma": noise_sigma,
            "method": method,
            "parameter": parameter,
            "PSNR": psnr,
            "SSIM": ssim,
            "runtime_seconds": runtime_seconds,
        }
    )


def run_denoising_robustness(clean: np.ndarray) -> pd.DataFrame:
    """Run fixed denoising methods under multiple noise levels."""
    rows: list[dict[str, float | str]] = []

    for index, noise_sigma in enumerate(DENOISING_NOISE_SIGMAS):
        noisy = add_gaussian_noise(clean, sigma=noise_sigma, seed=100 + index)
        append_denoising_row(rows, noise_sigma, "noisy_image", "-", clean, noisy, 0.0)

        start_time = perf_counter()
        restored = filters.gaussian(
            noisy,
            sigma=GAUSSIAN_FILTER_SIGMA,
            preserve_range=True,
        )
        runtime_seconds = perf_counter() - start_time
        append_denoising_row(
            rows,
            noise_sigma,
            "gaussian_filter",
            f"filter_sigma={GAUSSIAN_FILTER_SIGMA}",
            clean,
            np.clip(restored, 0.0, 1.0),
            runtime_seconds,
        )

        start_time = perf_counter()
        restored = tikhonov_denoise_fft(noisy, TIKHONOV_DENOISING_LAMBDA)
        runtime_seconds = perf_counter() - start_time
        append_denoising_row(
            rows,
            noise_sigma,
            "tikhonov_denoising",
            f"lambda={TIKHONOV_DENOISING_LAMBDA}",
            clean,
            restored,
            runtime_seconds,
        )

        start_time = perf_counter()
        restored = denoise_tv_chambolle(noisy, weight=TV_WEIGHT, channel_axis=None)
        runtime_seconds = perf_counter() - start_time
        append_denoising_row(
            rows,
            noise_sigma,
            "tv_chambolle",
            f"weight={TV_WEIGHT}",
            clean,
            np.clip(restored, 0.0, 1.0),
            runtime_seconds,
        )

        start_time = perf_counter()
        restored = denoise_nl_means(
            noisy,
            h=NLM_H,
            patch_size=NLM_PATCH_SIZE,
            patch_distance=NLM_PATCH_DISTANCE,
            fast_mode=NLM_FAST_MODE,
            channel_axis=None,
        )
        runtime_seconds = perf_counter() - start_time
        append_denoising_row(
            rows,
            noise_sigma,
            "nlm_denoising",
            f"h={NLM_H}",
            clean,
            np.clip(restored, 0.0, 1.0),
            runtime_seconds,
        )

    return pd.DataFrame(rows)


def run_deblurring_robustness(clean: np.ndarray) -> pd.DataFrame:
    """Run fixed deblurring methods under multiple blur levels."""
    rows: list[dict[str, float | str]] = []

    for index, blur_sigma in enumerate(DEBLURRING_BLUR_SIGMAS):
        psf = gaussian_psf(PSF_SIZE, blur_sigma)
        H = psf_to_otf(psf, clean.shape)
        blurred = circular_convolve_fft(clean, H)
        blurred_noisy = add_gaussian_noise(
            blurred,
            sigma=DEBLURRING_NOISE_SIGMA,
            seed=200 + index,
        )
        append_deblurring_row(
            rows,
            blur_sigma,
            DEBLURRING_NOISE_SIGMA,
            "blurred_noisy",
            "-",
            clean,
            blurred_noisy,
            0.0,
        )

        for lam in TIKHONOV_DEBLUR_LAMBDAS:
            start_time = perf_counter()
            restored = tikhonov_deblur_fft(blurred_noisy, H, lam)
            runtime_seconds = perf_counter() - start_time
            append_deblurring_row(
                rows,
                blur_sigma,
                DEBLURRING_NOISE_SIGMA,
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
            append_deblurring_row(
                rows,
                blur_sigma,
                DEBLURRING_NOISE_SIGMA,
                "wiener_deblur",
                f"balance={balance:g}",
                clean,
                restored,
                runtime_seconds,
            )

        for num_iter in RL_ITERATIONS:
            start_time = perf_counter()
            restored = richardson_lucy_compat(blurred_noisy, psf, num_iter)
            runtime_seconds = perf_counter() - start_time
            append_deblurring_row(
                rows,
                blur_sigma,
                DEBLURRING_NOISE_SIGMA,
                "richardson_lucy",
                f"num_iter={num_iter}",
                clean,
                restored,
                runtime_seconds,
            )

    return pd.DataFrame(rows)


def save_line_plot(
    table: pd.DataFrame,
    x_column: str,
    metric: str,
    ylabel: str,
    path: Path,
    exclude_baseline_from_runtime: bool = False,
) -> None:
    """Save a simple line plot for one metric across degradation strengths."""
    plot_table = table.copy()
    if exclude_baseline_from_runtime and metric == "runtime_seconds":
        plot_table = plot_table[
            ~plot_table["method"].isin(["noisy_image", "blurred_noisy"])
        ]

    fig, axis = plt.subplots(figsize=(7.0, 4.5), facecolor="white")
    for (method, parameter), group in plot_table.groupby(["method", "parameter"]):
        label = method if parameter == "-" else f"{method}, {parameter}"
        group = group.sort_values(x_column)
        axis.plot(
            group[x_column],
            group[metric],
            marker="o",
            linewidth=1.8,
            label=label,
        )

    axis.set_xlabel(x_column)
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel} vs {x_column}")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def best_method_summary(
    denoising: pd.DataFrame,
    deblurring: pd.DataFrame,
) -> list[list[str]]:
    """Create summary rows for a table-like best-method figure."""
    rows: list[list[str]] = []

    for noise_sigma, group in denoising.groupby("noise_sigma"):
        best_psnr = group.loc[group["PSNR"].idxmax()]
        best_ssim = group.loc[group["SSIM"].idxmax()]
        rows.append(
            [
                f"denoise noise={noise_sigma:g}",
                f"{best_psnr['method']} {best_psnr['parameter']}",
                f"{best_ssim['method']} {best_ssim['parameter']}",
            ]
        )

    for blur_sigma, group in deblurring.groupby("blur_sigma"):
        best_psnr = group.loc[group["PSNR"].idxmax()]
        best_ssim = group.loc[group["SSIM"].idxmax()]
        rows.append(
            [
                f"deblur blur={blur_sigma:g}",
                f"{best_psnr['method']} {best_psnr['parameter']}",
                f"{best_ssim['method']} {best_ssim['parameter']}",
            ]
        )

    return rows


def save_summary_figure(denoising: pd.DataFrame, deblurring: pd.DataFrame, path: Path) -> None:
    """Save a table-like summary of best methods by degradation level."""
    rows = best_method_summary(denoising, deblurring)
    fig, axis = plt.subplots(figsize=(9, 3.2), facecolor="white")
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Condition", "Best PSNR", "Best SSIM"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    axis.set_title("Degradation robustness best-method summary")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_all_figures(denoising: pd.DataFrame, deblurring: pd.DataFrame) -> None:
    """Save all requested Phase 7 figures."""
    save_line_plot(
        denoising,
        "noise_sigma",
        "PSNR",
        "PSNR",
        FIGURES_DIR / "15_degradation_robustness_denoising_psnr.png",
    )
    save_line_plot(
        denoising,
        "noise_sigma",
        "SSIM",
        "SSIM",
        FIGURES_DIR / "15_degradation_robustness_denoising_ssim.png",
    )
    save_line_plot(
        denoising,
        "noise_sigma",
        "runtime_seconds",
        "Runtime seconds",
        FIGURES_DIR / "15_degradation_robustness_denoising_runtime.png",
        exclude_baseline_from_runtime=True,
    )
    save_line_plot(
        deblurring,
        "blur_sigma",
        "PSNR",
        "PSNR",
        FIGURES_DIR / "15_degradation_robustness_deblurring_psnr.png",
    )
    save_line_plot(
        deblurring,
        "blur_sigma",
        "SSIM",
        "SSIM",
        FIGURES_DIR / "15_degradation_robustness_deblurring_ssim.png",
    )
    save_line_plot(
        deblurring,
        "blur_sigma",
        "runtime_seconds",
        "Runtime seconds",
        FIGURES_DIR / "15_degradation_robustness_deblurring_runtime.png",
        exclude_baseline_from_runtime=True,
    )
    save_summary_figure(
        denoising,
        deblurring,
        FIGURES_DIR / "15_degradation_robustness_summary.png",
    )


def main() -> None:
    ensure_output_folders()

    clean = img_as_float(data.camera())
    denoising = run_denoising_robustness(clean)
    deblurring = run_deblurring_robustness(clean)

    denoising_path = RESULTS_DIR / "15_degradation_robustness_denoising.csv"
    deblurring_path = RESULTS_DIR / "15_degradation_robustness_deblurring.csv"
    denoising.to_csv(denoising_path, index=False)
    deblurring.to_csv(deblurring_path, index=False)

    save_all_figures(denoising, deblurring)

    print("Degradation robustness study completed successfully.")
    print(f"Denoising results saved to: {denoising_path}")
    print(f"Deblurring results saved to: {deblurring_path}")
    print("\nDenoising results:")
    print(denoising.to_string(index=False))
    print("\nDeblurring results:")
    print(deblurring.to_string(index=False))


if __name__ == "__main__":
    main()
