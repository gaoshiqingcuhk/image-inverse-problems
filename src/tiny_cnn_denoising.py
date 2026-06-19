from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import color, data, filters, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Phase 8 / v0.9: Tiny CNN denoising baseline.
#
# This script intentionally uses a very small CPU-friendly convolutional neural
# network. It is a minimal learning-based baseline, not a modern deep-learning
# denoising benchmark.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

NOISE_SIGMA = 0.10
RANDOM_SEED = 42
PATCH_SIZE = 64
NUM_TRAIN_PATCHES = 512
NUM_VAL_PATCHES = 128
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-3

GAUSSIAN_FILTER_SIGMA = 1.0
TV_WEIGHT = 0.1
NLM_H = 0.10
NLM_PATCH_SIZE = 5
NLM_PATCH_DISTANCE = 6
NLM_FAST_MODE = True


def ensure_output_folders() -> None:
    """Create folders for saved results, figures, and model weights."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def set_random_seeds(seed: int) -> None:
    """Set deterministic seeds for NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Convert a built-in skimage image to grayscale float values in [0, 1]."""
    image = img_as_float(image)
    if image.ndim == 3:
        image = color.rgb2gray(image)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def load_training_images() -> list[np.ndarray]:
    """Load training images that are separate from the camera test image."""
    return [
        to_grayscale_float(data.coins()),
        to_grayscale_float(data.moon()),
        to_grayscale_float(data.page()),
    ]


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise and clip to [0, 1]."""
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


class PatchDenoisingDataset(Dataset):
    """Random clean/noisy patch pairs generated from fixed training images."""

    def __init__(
        self,
        images: list[np.ndarray],
        num_patches: int,
        patch_size: int,
        noise_sigma: float,
        seed: int,
    ) -> None:
        self.images = images
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.noise_sigma = noise_sigma
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.patch_specs = []
        for _ in range(num_patches):
            image_index = int(rng.integers(0, len(images)))
            image = images[image_index]
            max_row = image.shape[0] - patch_size
            max_col = image.shape[1] - patch_size
            row = int(rng.integers(0, max_row + 1))
            col = int(rng.integers(0, max_col + 1))
            noise_seed = int(rng.integers(0, 2**31 - 1))
            self.patch_specs.append((image_index, row, col, noise_seed))

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_index, row, col, noise_seed = self.patch_specs[index]
        clean = self.images[image_index][
            row : row + self.patch_size,
            col : col + self.patch_size,
        ]
        rng = np.random.default_rng(noise_seed)
        noisy = add_gaussian_noise(clean, self.noise_sigma, rng)

        noisy_tensor = torch.from_numpy(noisy[None, :, :]).float()
        clean_tensor = torch.from_numpy(clean[None, :, :]).float()
        return noisy_tensor, clean_tensor


class TinyDenoisingCNN(nn.Module):
    """A deliberately tiny direct-prediction CNN."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        prediction = model(noisy)
        loss = criterion(prediction, clean)
        loss.backward()
        optimizer.step()

        batch_size = noisy.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate mean validation loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            prediction = model(noisy)
            loss = criterion(prediction, clean)
            batch_size = noisy.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train_model(device: torch.device) -> tuple[nn.Module, pd.DataFrame, float]:
    """Train the tiny CNN and return model, loss history, and training time."""
    train_images = load_training_images()
    train_dataset = PatchDenoisingDataset(
        images=train_images,
        num_patches=NUM_TRAIN_PATCHES,
        patch_size=PATCH_SIZE,
        noise_sigma=NOISE_SIGMA,
        seed=RANDOM_SEED,
    )
    val_dataset = PatchDenoisingDataset(
        images=train_images,
        num_patches=NUM_VAL_PATCHES,
        patch_size=PATCH_SIZE,
        noise_sigma=NOISE_SIGMA,
        seed=RANDOM_SEED + 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = TinyDenoisingCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history_rows = []
    start_time = perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    training_time = perf_counter() - start_time
    return model, pd.DataFrame(history_rows), training_time


def compute_metrics(clean: np.ndarray, restored: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM using the clean image as reference."""
    psnr = peak_signal_noise_ratio(clean, restored, data_range=1.0)
    ssim = structural_similarity(clean, restored, data_range=1.0)
    return psnr, ssim


def append_result_row(
    rows: list[dict[str, float | str]],
    clean: np.ndarray,
    restored: np.ndarray,
    method: str,
    parameter: str,
    runtime_seconds: float,
) -> None:
    """Compute metrics and append one comparison row."""
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


def run_cnn_inference(
    model: nn.Module,
    noisy: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Run full-image CNN inference and measure inference time only."""
    model.eval()
    tensor = torch.from_numpy(noisy[None, None, :, :]).float().to(device)

    start_time = perf_counter()
    with torch.no_grad():
        prediction = model(tensor)
    runtime_seconds = perf_counter() - start_time

    restored = prediction.squeeze().detach().cpu().numpy()
    return np.clip(restored, 0.0, 1.0).astype(np.float32), runtime_seconds


def run_test_comparison(
    model: nn.Module,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compare noisy, classical baselines, NLM, and Tiny CNN on camera."""
    rng = np.random.default_rng(RANDOM_SEED)
    clean = to_grayscale_float(data.camera())
    noisy = add_gaussian_noise(clean, NOISE_SIGMA, rng)

    rows: list[dict[str, float | str]] = []
    images = {
        "clean": clean,
        "noisy_image": noisy,
    }

    append_result_row(rows, clean, noisy, "noisy_image", "-", 0.0)

    start_time = perf_counter()
    gaussian_result = filters.gaussian(
        noisy,
        sigma=GAUSSIAN_FILTER_SIGMA,
        preserve_range=True,
    )
    gaussian_runtime = perf_counter() - start_time
    gaussian_result = np.clip(gaussian_result, 0.0, 1.0).astype(np.float32)
    images["gaussian_filter"] = gaussian_result
    append_result_row(
        rows,
        clean,
        gaussian_result,
        "gaussian_filter",
        f"filter_sigma={GAUSSIAN_FILTER_SIGMA}",
        gaussian_runtime,
    )

    start_time = perf_counter()
    tv_result = denoise_tv_chambolle(noisy, weight=TV_WEIGHT, channel_axis=None)
    tv_runtime = perf_counter() - start_time
    tv_result = np.clip(tv_result, 0.0, 1.0).astype(np.float32)
    images["tv_chambolle"] = tv_result
    append_result_row(
        rows,
        clean,
        tv_result,
        "tv_chambolle",
        f"weight={TV_WEIGHT}",
        tv_runtime,
    )

    start_time = perf_counter()
    nlm_result = denoise_nl_means(
        noisy,
        h=NLM_H,
        patch_size=NLM_PATCH_SIZE,
        patch_distance=NLM_PATCH_DISTANCE,
        fast_mode=NLM_FAST_MODE,
        channel_axis=None,
    )
    nlm_runtime = perf_counter() - start_time
    nlm_result = np.clip(nlm_result, 0.0, 1.0).astype(np.float32)
    images["nlm_denoising"] = nlm_result
    append_result_row(
        rows,
        clean,
        nlm_result,
        "nlm_denoising",
        f"h={NLM_H}",
        nlm_runtime,
    )

    cnn_result, cnn_runtime = run_cnn_inference(model, noisy, device)
    images["tiny_cnn"] = cnn_result
    append_result_row(
        rows,
        clean,
        cnn_result,
        "tiny_cnn",
        "trained_model",
        cnn_runtime,
    )

    return pd.DataFrame(rows), images


def save_training_loss(history: pd.DataFrame, path: Path) -> None:
    """Save train/validation loss curves."""
    fig, axis = plt.subplots(figsize=(6.5, 4.2), facecolor="white")
    axis.plot(history["epoch"], history["train_loss"], marker="o", label="Train loss")
    axis.plot(history["epoch"], history["val_loss"], marker="o", label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title("Tiny CNN denoising training loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_metric_bar(results: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    """Save a bar chart for PSNR, SSIM, or runtime."""
    fig, axis = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    labels = results["method"].to_list()
    axis.bar(labels, results[metric])
    axis.set_ylabel(ylabel)
    axis.set_title(f"Tiny CNN denoising comparison: {ylabel}")
    axis.grid(True, axis="y", alpha=0.25)
    axis.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def format_title(title: str, psnr: float | None = None, ssim: float | None = None) -> str:
    """Create compact subplot titles."""
    if psnr is None or ssim is None:
        return title
    return f"{title}\nPSNR={psnr:.2f}, SSIM={ssim:.3f}"


def row_for(results: pd.DataFrame, method: str) -> pd.Series:
    """Select one result row by method name."""
    return results[results["method"] == method].iloc[0]


def save_visual_grid(results: pd.DataFrame, images: dict[str, np.ndarray], path: Path) -> None:
    """Save clean/noisy/classical/CNN visual comparison."""
    items = [
        ("clean", "Clean camera", None),
        ("noisy_image", "Noisy camera", "noisy_image"),
        ("gaussian_filter", "Gaussian", "gaussian_filter"),
        ("tv_chambolle", "TV", "tv_chambolle"),
        ("nlm_denoising", "NLM h=0.10", "nlm_denoising"),
        ("tiny_cnn", "Tiny CNN", "tiny_cnn"),
    ]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 7), facecolor="white")
    for axis, (image_key, title, method) in zip(axes.ravel(), items):
        axis.imshow(images[image_key], cmap="gray", vmin=0.0, vmax=1.0)
        if method is None:
            axis.set_title(title)
        else:
            row = row_for(results, method)
            axis.set_title(format_title(title, row["PSNR"], row["SSIM"]))
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_error_maps(results: pd.DataFrame, images: dict[str, np.ndarray], path: Path) -> None:
    """Save absolute error maps for noisy, baselines, NLM, and Tiny CNN."""
    clean = images["clean"]
    items = [
        ("Noisy", "noisy_image"),
        ("Gaussian", "gaussian_filter"),
        ("TV", "tv_chambolle"),
        ("NLM h=0.10", "nlm_denoising"),
        ("Tiny CNN", "tiny_cnn"),
    ]
    errors = [np.abs(clean - images[key]) for _, key in items]
    vmax = max(float(error.max()) for error in errors)

    fig, axes = plt.subplots(nrows=1, ncols=len(items), figsize=(3.1 * len(items), 3.5), facecolor="white")
    for axis, (title, _), error in zip(axes, items, errors):
        image = axis.imshow(error, cmap="inferno", vmin=0.0, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")

    fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_metadata(path: Path, training_time_seconds: float, device: torch.device) -> None:
    """Write experiment metadata to a text file."""
    content = "\n".join(
        [
            "architecture: Conv2d(1,16,3,padding=1) -> ReLU -> Conv2d(16,16,3,padding=1) -> ReLU -> Conv2d(16,1,3,padding=1) -> Sigmoid",
            f"noise_sigma: {NOISE_SIGMA}",
            f"patch_size: {PATCH_SIZE}",
            f"num_train_patches: {NUM_TRAIN_PATCHES}",
            f"num_val_patches: {NUM_VAL_PATCHES}",
            f"batch_size: {BATCH_SIZE}",
            f"epochs: {EPOCHS}",
            f"learning_rate: {LEARNING_RATE}",
            f"training_time_seconds: {training_time_seconds:.6f}",
            f"device: {device}",
            f"torch_version: {torch.__version__}",
            "train_images: coins, moon, page",
            "test_image: camera",
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def save_outputs(
    model: nn.Module,
    results: pd.DataFrame,
    history: pd.DataFrame,
    images: dict[str, np.ndarray],
    training_time_seconds: float,
    device: torch.device,
) -> None:
    """Save all required Phase 8 outputs."""
    results.to_csv(RESULTS_DIR / "16_tiny_cnn_denoising_results.csv", index=False)
    history.to_csv(RESULTS_DIR / "16_tiny_cnn_training_history.csv", index=False)
    save_metadata(RESULTS_DIR / "16_tiny_cnn_metadata.txt", training_time_seconds, device)
    torch.save(model.state_dict(), MODELS_DIR / "16_tiny_cnn_denoising.pt")

    save_training_loss(history, FIGURES_DIR / "16_tiny_cnn_training_loss.png")
    save_metric_bar(results, "PSNR", "PSNR", FIGURES_DIR / "16_tiny_cnn_denoising_psnr.png")
    save_metric_bar(results, "SSIM", "SSIM", FIGURES_DIR / "16_tiny_cnn_denoising_ssim.png")
    save_metric_bar(
        results,
        "runtime_seconds",
        "Runtime seconds",
        FIGURES_DIR / "16_tiny_cnn_denoising_runtime.png",
    )
    save_visual_grid(results, images, FIGURES_DIR / "16_tiny_cnn_denoising_visual_grid.png")
    save_error_maps(results, images, FIGURES_DIR / "16_tiny_cnn_denoising_error_maps.png")


def main() -> None:
    ensure_output_folders()
    set_random_seeds(RANDOM_SEED)

    device = torch.device("cpu")
    model, history, training_time_seconds = train_model(device)
    results, images = run_test_comparison(model, device)
    save_outputs(model, results, history, images, training_time_seconds, device)

    print("Tiny CNN denoising experiment completed successfully.")
    print(f"Training time seconds: {training_time_seconds:.6f}")
    print("\nTraining history:")
    print(history.to_string(index=False))
    print("\nComparison results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
