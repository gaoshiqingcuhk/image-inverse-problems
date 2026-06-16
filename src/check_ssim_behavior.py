from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, filters, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# Diagnostic script only:
# check whether the unusual SSIM value at filter_sigma = 0.5 is reproducible.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"

NOISE_SIGMA = 0.10
RANDOM_SEED = 42
FILTER_SIGMAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    """Add one reproducible Gaussian noise realization and clip to [0, 1]."""
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def compute_metrics(clean: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM with an explicit data range."""
    psnr = peak_signal_noise_ratio(clean, test, data_range=1.0)
    ssim = structural_similarity(clean, test, data_range=1.0)
    return psnr, ssim


def print_image_check(label: str, clean: np.ndarray, image: np.ndarray) -> None:
    """Print basic numerical checks and quality metrics for one image."""
    psnr, ssim = compute_metrics(clean, image)
    print(label)
    print(f"  min_pixel_value: {image.min():.12f}")
    print(f"  max_pixel_value: {image.max():.12f}")
    print(f"  has_nan: {np.isnan(image).any()}")
    print(f"  PSNR: {psnr:.6f}")
    print(f"  SSIM: {ssim:.6f}")


def save_visual_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised_05: np.ndarray,
    denoised_10: np.ndarray,
    denoised_20: np.ndarray,
    path: Path,
) -> None:
    """Save a visual comparison for the suspicious and reference cases."""
    images = [clean, noisy, denoised_05, denoised_10, denoised_20]
    titles = [
        "Clean image",
        "Noisy image",
        "filter_sigma=0.5",
        "filter_sigma=1.0",
        "filter_sigma=2.0",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    clean = img_as_float(data.camera())
    noisy = add_gaussian_noise(clean)

    print("SSIM behavior diagnostic")
    print(f"noise_sigma: {NOISE_SIGMA}")
    print(f"random_seed: {RANDOM_SEED}")
    print()

    print_image_check("noisy_image baseline", clean, noisy)
    print()

    denoised_images = {}
    for filter_sigma in FILTER_SIGMAS:
        denoised = filters.gaussian(
            noisy,
            sigma=filter_sigma,
            preserve_range=True,
        )
        denoised_images[filter_sigma] = denoised
        print_image_check(f"gaussian_filter filter_sigma={filter_sigma}", clean, denoised)
        print()

    figure_path = FIGURES_DIR / "check_ssim_filter_sigma_0.5.png"
    save_visual_comparison(
        clean=clean,
        noisy=noisy,
        denoised_05=denoised_images[0.5],
        denoised_10=denoised_images[1.0],
        denoised_20=denoised_images[2.0],
        path=figure_path,
    )

    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
