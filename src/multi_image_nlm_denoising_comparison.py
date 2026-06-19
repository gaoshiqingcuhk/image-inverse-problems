from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage import color, data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle


# Phase 5B: multi-image Non-local Means denoising robustness.
#
# The two NLM h values are fixed from the Phase 5A camera-image experiment.
# This script does not tune h separately for each image; it tests whether those
# settings transfer across several standard images.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

IMAGE_LOADERS = [
    ("camera", data.camera, 42),
    ("coins", data.coins, 43),
    ("moon", data.moon, 44),
    ("page", data.page, 45),
]

NOISE_SIGMA = 0.10
GAUSSIAN_FILTER_SIGMA = 1.0
TV_WEIGHT = 0.1
NLM_H_VALUES = [0.08, 0.10]
NLM_PATCH_SIZE = 5
NLM_PATCH_DISTANCE = 6
NLM_FAST_MODE = True

METHOD_ORDER = [
    "noisy_image",
    "gaussian_filter",
    "tv_chambolle",
    "nlm_denoising_h_0.08",
    "nlm_denoising_h_0.10",
]

METHOD_LABELS = {
    "noisy_image": "Noisy",
    "gaussian_filter": "Gaussian",
    "tv_chambolle": "TV 0.1",
    "nlm_denoising_h_0.08": "NLM h=0.08",
    "nlm_denoising_h_0.10": "NLM h=0.10",
}


def ensure_output_folders() -> None:
    """Create folders for saved results and figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_grayscale_float(loader) -> np.ndarray:
    """Load a skimage image, convert RGB to grayscale if needed, and clip."""
    image = img_as_float(loader())
    if image.ndim == 3:
        image = color.rgb2gray(image)
    return np.clip(image, 0.0, 1.0)


def add_gaussian_noise(image: np.ndarray, seed: int) -> np.ndarray:
    """Add reproducible Gaussian noise and clip the degraded image."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def compute_metrics(clean: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM with the clean image as reference."""
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
    """Compute metrics and append one row to the result table."""
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
    gaussian_result = gaussian_filter(noisy, sigma=GAUSSIAN_FILTER_SIGMA)
    gaussian_runtime = perf_counter() - start_time
    gaussian_result = np.clip(gaussian_result, 0.0, 1.0)
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

    start_time = perf_counter()
    tv_result = denoise_tv_chambolle(
        noisy,
        weight=TV_WEIGHT,
        channel_axis=None,
    )
    tv_runtime = perf_counter() - start_time
    tv_result = np.clip(tv_result, 0.0, 1.0)
    images["tv_chambolle"] = tv_result
    add_result_row(
        rows,
        image_name,
        "tv_chambolle",
        f"weight={TV_WEIGHT}",
        clean,
        tv_result,
        tv_runtime,
    )

    for h_value in NLM_H_VALUES:
        start_time = perf_counter()
        nlm_result = denoise_nl_means(
            noisy,
            h=h_value,
            patch_size=NLM_PATCH_SIZE,
            patch_distance=NLM_PATCH_DISTANCE,
            fast_mode=NLM_FAST_MODE,
            channel_axis=None,
        )
        nlm_runtime = perf_counter() - start_time
        nlm_result = np.clip(nlm_result, 0.0, 1.0)

        method = f"nlm_denoising_h_{h_value:.2f}"
        images[method] = nlm_result
        add_result_row(
            rows,
            image_name,
            method,
            f"h={h_value:.2f}",
            clean,
            nlm_result,
            nlm_runtime,
        )

    return rows, images


def run_experiment() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Run the full multi-image NLM robustness experiment."""
    all_rows = []
    all_images = {}

    for image_name, loader, seed in IMAGE_LOADERS:
        clean = load_grayscale_float(loader)
        noisy = add_gaussian_noise(clean, seed)
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
    """Save grouped bar charts comparing methods across images."""
    methods = METHOD_ORDER if include_noisy else METHOD_ORDER[1:]
    image_names = [name for name, _, _ in IMAGE_LOADERS]
    x = np.arange(len(image_names))
    width = 0.14 if include_noisy else 0.17

    fig, axis = plt.subplots(figsize=(9.5, 4.8), facecolor="white")

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
    axis.set_title(f"Multi-image NLM denoising robustness: {ylabel}")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    if use_log_scale:
        axis.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_visual_grid(
    metrics: pd.DataFrame,
    all_images: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    """Save rows as images and columns as clean/noisy/restored outputs."""
    image_names = [name for name, _, _ in IMAGE_LOADERS]
    columns = ["clean"] + METHOD_ORDER

    fig, axes = plt.subplots(
        nrows=len(image_names),
        ncols=len(columns),
        figsize=(16, 10),
        facecolor="white",
    )

    for row_index, image_name in enumerate(image_names):
        for col_index, column in enumerate(columns):
            axis = axes[row_index, col_index]
            axis.imshow(all_images[image_name][column], cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")

            if column == "clean":
                title = f"{image_name}\nClean"
            else:
                metric_row = metrics[
                    (metrics["image_name"] == image_name)
                    & (metrics["method"] == column)
                ].iloc[0]
                title = (
                    f"{METHOD_LABELS[column]}\n"
                    f"PSNR={metric_row['PSNR']:.2f}, SSIM={metric_row['SSIM']:.3f}"
                )
            axis.set_title(title, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_best_method_summary(metrics: pd.DataFrame, path: Path) -> None:
    """Save a compact table-like figure showing best PSNR and SSIM methods."""
    rows = []
    for image_name, _, _ in IMAGE_LOADERS:
        image_rows = metrics[metrics["image_name"] == image_name]
        best_psnr = image_rows.loc[image_rows["PSNR"].idxmax()]
        best_ssim = image_rows.loc[image_rows["SSIM"].idxmax()]
        rows.append(
            [
                image_name,
                f"{METHOD_LABELS[best_psnr['method']]}\n{best_psnr['PSNR']:.2f}",
                f"{METHOD_LABELS[best_ssim['method']]}\n{best_ssim['SSIM']:.3f}",
            ]
        )

    fig, axis = plt.subplots(figsize=(8, 2.8), facecolor="white")
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Image", "Best PSNR method", "Best SSIM method"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)
    axis.set_title("Best method summary for multi-image NLM denoising")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return average metrics and per-image best-method summaries."""
    averages = (
        metrics.groupby("method", as_index=False)
        .agg(
            average_PSNR=("PSNR", "mean"),
            average_SSIM=("SSIM", "mean"),
            average_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values("method")
    )

    best_rows = []
    for image_name, _, _ in IMAGE_LOADERS:
        image_rows = metrics[metrics["image_name"] == image_name]
        best_psnr = image_rows.loc[image_rows["PSNR"].idxmax()]
        best_ssim = image_rows.loc[image_rows["SSIM"].idxmax()]
        best_rows.append(
            {
                "image_name": image_name,
                "best_PSNR_method": best_psnr["method"],
                "best_PSNR": best_psnr["PSNR"],
                "best_SSIM_method": best_ssim["method"],
                "best_SSIM": best_ssim["SSIM"],
            }
        )

    return averages, pd.DataFrame(best_rows)


def main() -> None:
    ensure_output_folders()

    metrics, all_images = run_experiment()

    results_path = RESULTS_DIR / "13_multi_image_nlm_denoising_comparison.csv"
    metrics.to_csv(results_path, index=False)

    save_metric_by_method_figure(
        metrics,
        metric_name="PSNR",
        ylabel="PSNR",
        path=FIGURES_DIR / "13_multi_image_nlm_denoising_psnr_by_method.png",
    )
    save_metric_by_method_figure(
        metrics,
        metric_name="SSIM",
        ylabel="SSIM",
        path=FIGURES_DIR / "13_multi_image_nlm_denoising_ssim_by_method.png",
    )
    save_metric_by_method_figure(
        metrics,
        metric_name="runtime_seconds",
        ylabel="Runtime seconds",
        path=FIGURES_DIR / "13_multi_image_nlm_denoising_runtime_by_method.png",
        include_noisy=False,
        use_log_scale=True,
    )
    save_visual_grid(
        metrics,
        all_images,
        path=FIGURES_DIR / "13_multi_image_nlm_denoising_visual_grid.png",
    )
    save_best_method_summary(
        metrics,
        path=FIGURES_DIR / "13_multi_image_nlm_denoising_best_method_summary.png",
    )

    averages, best_by_image = summarize_metrics(metrics)

    print("Multi-image NLM denoising robustness experiment completed successfully.")
    print(f"Results saved to: {results_path}")
    print("\nFull results:")
    print(metrics.to_string(index=False))
    print("\nAverage metrics by method:")
    print(averages.to_string(index=False))
    print("\nBest methods by image:")
    print(best_by_image.to_string(index=False))


if __name__ == "__main__":
    main()
