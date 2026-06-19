from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, filters, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle


# Phase 5A: Non-local Means denoising baseline.
#
# This experiment keeps the same single-image denoising setting used earlier:
# skimage.data.camera(), Gaussian noise with sigma=0.10, and random seed 42.
# NLM adds a patch-similarity / non-local image prior baseline to the local and
# variational denoising methods already in the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

NOISE_SIGMA = 0.10
RANDOM_SEED = 42
GAUSSIAN_FILTER_SIGMA = 1.0
TV_WEIGHT = 0.1
NLM_H_VALUES = [0.04, 0.06, 0.08, 0.10, 0.12]
NLM_PATCH_SIZE = 5
NLM_PATCH_DISTANCE = 6
NLM_FAST_MODE = True


def ensure_output_folders() -> None:
    """Create folders used by this experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    """Add reproducible Gaussian noise and clip the result to [0, 1]."""
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


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
    """Compute metrics and append one output row."""
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


def run_experiment(clean: np.ndarray, noisy: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run noisy baseline, Gaussian, TV, and NLM denoising."""
    rows: list[dict[str, float | str]] = []
    images = {
        "clean": clean,
        "noisy_image": noisy,
    }

    add_result_row(rows, clean, noisy, "noisy_image", "-", 0.0)

    start_time = perf_counter()
    gaussian_result = filters.gaussian(
        noisy,
        sigma=GAUSSIAN_FILTER_SIGMA,
        preserve_range=True,
    )
    gaussian_runtime = perf_counter() - start_time
    gaussian_result = np.clip(gaussian_result, 0.0, 1.0)
    images["gaussian_filter"] = gaussian_result
    add_result_row(
        rows,
        clean,
        gaussian_result,
        "gaussian_filter",
        f"filter_sigma={GAUSSIAN_FILTER_SIGMA}",
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
        clean,
        tv_result,
        "tv_chambolle",
        f"weight={TV_WEIGHT}",
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
        images[f"nlm_h_{h_value:g}"] = nlm_result
        add_result_row(
            rows,
            clean,
            nlm_result,
            "nlm_denoising",
            f"h={h_value:g}",
            nlm_runtime,
        )

    return pd.DataFrame(rows), images


def parse_parameter_value(parameter: str) -> float:
    """Extract the numeric value from strings like 'h=0.08'."""
    return float(parameter.split("=", maxsplit=1)[1])


def save_metric_curve(metrics: pd.DataFrame, metric_name: str, path: Path) -> None:
    """Save an NLM h sensitivity curve with reference baselines."""
    nlm_rows = metrics[metrics["method"] == "nlm_denoising"].copy()
    nlm_rows["h"] = nlm_rows["parameter"].map(parse_parameter_value)

    fig, axis = plt.subplots(figsize=(6.5, 4.2), facecolor="white")
    axis.plot(
        nlm_rows["h"],
        nlm_rows[metric_name],
        marker="o",
        linewidth=2,
        label="NLM denoising",
    )

    references = [
        ("noisy_image", "Noisy image", "gray"),
        ("gaussian_filter", "Gaussian filter", "tab:green"),
        ("tv_chambolle", "TV Chambolle", "tab:purple"),
    ]
    for method, label, color in references:
        value = metrics.loc[metrics["method"] == method, metric_name].iloc[0]
        axis.axhline(
            value,
            linestyle="--",
            linewidth=1.4,
            color=color,
            label=label,
        )

    axis.set_xlabel("NLM h")
    axis.set_ylabel(metric_name)
    axis.set_title(f"Non-local Means denoising: {metric_name}")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_runtime_chart(metrics: pd.DataFrame, path: Path) -> None:
    """Save runtime comparison excluding the zero-cost noisy baseline."""
    plot_rows = metrics[metrics["method"] != "noisy_image"].copy()
    labels = plot_rows.apply(lambda row: f"{row['method']}\n{row['parameter']}", axis=1)

    fig, axis = plt.subplots(figsize=(8, 4.5), facecolor="white")
    axis.bar(labels, plot_rows["runtime_seconds"])
    axis.set_ylabel("Runtime seconds")
    axis.set_title("Non-local Means denoising: runtime")
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def format_title(title: str, psnr: float, ssim: float) -> str:
    """Make compact subplot titles."""
    return f"{title}\nPSNR={psnr:.2f}, SSIM={ssim:.3f}"


def row_for(metrics: pd.DataFrame, method: str, parameter: str | None = None) -> pd.Series:
    """Select one metric row by method and optional parameter string."""
    rows = metrics[metrics["method"] == method]
    if parameter is not None:
        rows = rows[rows["parameter"] == parameter]
    return rows.iloc[0]


def best_nlm_rows(metrics: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return best NLM rows by PSNR and SSIM."""
    nlm_rows = metrics[metrics["method"] == "nlm_denoising"]
    best_psnr = nlm_rows.loc[nlm_rows["PSNR"].idxmax()]
    best_ssim = nlm_rows.loc[nlm_rows["SSIM"].idxmax()]
    return best_psnr, best_ssim


def save_visual_grid(metrics: pd.DataFrame, images: dict[str, np.ndarray], path: Path) -> None:
    """Save a visual comparison of key denoising outputs."""
    best_psnr_row, best_ssim_row = best_nlm_rows(metrics)
    best_items = [
        (f"Best NLM PSNR\n{best_psnr_row['parameter']}", best_psnr_row["parameter"]),
    ]
    if best_ssim_row["parameter"] != best_psnr_row["parameter"]:
        best_items.append((f"Best NLM SSIM\n{best_ssim_row['parameter']}", best_ssim_row["parameter"]))
    else:
        best_items[0] = (
            f"Best NLM PSNR/SSIM\n{best_psnr_row['parameter']}",
            best_psnr_row["parameter"],
        )

    items: list[tuple[str, str, str, str | None]] = [
        ("clean", "Clean image", "clean", None),
        ("noisy_image", "Noisy image", "noisy_image", None),
        ("gaussian_filter", "Gaussian filter", "gaussian_filter", None),
        ("tv_chambolle", "TV Chambolle", "tv_chambolle", None),
    ]
    for title, parameter in best_items:
        h_value = parse_parameter_value(parameter)
        items.append((f"nlm_h_{h_value:g}", title, "nlm_denoising", parameter))

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 7), facecolor="white")
    flat_axes = axes.ravel()

    for axis, (image_key, title, method, parameter) in zip(flat_axes, items):
        axis.imshow(images[image_key], cmap="gray", vmin=0.0, vmax=1.0)
        if method == "clean":
            axis.set_title(title)
        else:
            row = row_for(metrics, method, parameter)
            axis.set_title(format_title(title, row["PSNR"], row["SSIM"]))
        axis.axis("off")

    for axis in flat_axes[len(items) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_error_maps(clean: np.ndarray, metrics: pd.DataFrame, images: dict[str, np.ndarray], path: Path) -> None:
    """Save absolute error maps for noisy, baselines, and best NLM results."""
    best_psnr_row, best_ssim_row = best_nlm_rows(metrics)
    items = [
        ("Noisy image", "noisy_image"),
        ("Gaussian filter", "gaussian_filter"),
        ("TV Chambolle", "tv_chambolle"),
    ]

    best_psnr_h = parse_parameter_value(best_psnr_row["parameter"])
    items.append((f"Best NLM PSNR\n{best_psnr_row['parameter']}", f"nlm_h_{best_psnr_h:g}"))
    if best_ssim_row["parameter"] != best_psnr_row["parameter"]:
        best_ssim_h = parse_parameter_value(best_ssim_row["parameter"])
        items.append((f"Best NLM SSIM\n{best_ssim_row['parameter']}", f"nlm_h_{best_ssim_h:g}"))

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
    noisy = add_gaussian_noise(clean)
    metrics, images = run_experiment(clean, noisy)

    results_path = RESULTS_DIR / "12_nlm_denoising_results.csv"
    metrics.to_csv(results_path, index=False)

    save_metric_curve(metrics, "PSNR", FIGURES_DIR / "12_nlm_denoising_psnr.png")
    save_metric_curve(metrics, "SSIM", FIGURES_DIR / "12_nlm_denoising_ssim.png")
    save_runtime_chart(metrics, FIGURES_DIR / "12_nlm_denoising_runtime.png")
    save_visual_grid(metrics, images, FIGURES_DIR / "12_nlm_denoising_visual_grid.png")
    save_error_maps(clean, metrics, images, FIGURES_DIR / "12_nlm_denoising_error_maps.png")

    best_psnr_row, best_ssim_row = best_nlm_rows(metrics)

    print("Non-local Means denoising experiment completed successfully.")
    print(f"Results saved to: {results_path}")
    print(f"Best NLM by PSNR: {best_psnr_row['parameter']} ({best_psnr_row['PSNR']:.6f})")
    print(f"Best NLM by SSIM: {best_ssim_row['parameter']} ({best_ssim_row['SSIM']:.6f})")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
