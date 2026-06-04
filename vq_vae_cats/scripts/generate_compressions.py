import argparse
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import random_split
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
VQVAE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = VQVAE_ROOT.parent
SRC_DIR = VQVAE_ROOT / "src"
DATASETS_DIR = REPO_ROOT / "datasets"
MODEL_SEARCH_DIRS = [
    VQVAE_ROOT / "models" / "vq_vae_models",
    VQVAE_ROOT / "models" / "vqvae_models",
]

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DATASETS_DIR))

from cat_datasets import CatDataset
from models.vq_vae import VQ_VAE

BETAS = [0.5, 1.0, 1.5]
CODEBOOK_SIZES = [128, 256, 512]
VAL_SIZE = 500
TEST_SIZE = 500
SPLIT_SEED = 42
IMAGE_SIZE = 128
GRID_ROWS = 3
COLS_PER_ROW = 6
N_TEST_IMAGES = GRID_ROWS * (COLS_PER_ROW // 2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(
        description="VQ-VAE reconstructions on test images for all beta x K configs."
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--n-images", type=int, default=N_TEST_IMAGES)
    p.add_argument("--seed", type=int, default=0, help="Seed for picking test indices")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=VQVAE_ROOT / "outputs" / "vqvae_compressions",
    )
    p.add_argument("--padding", type=int, default=2)
    return p.parse_args()


def resolve_data_dir(data_dir):
    if data_dir is not None:
        path = Path(data_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for candidate in (
        Path.home() / "data" / "cats",
        REPO_ROOT / "data" / "cats",
        VQVAE_ROOT / "data" / "cats",
    ):
        candidate = candidate.expanduser().resolve()
        if candidate.is_dir() and any(candidate.rglob("*.jpg")):
            return candidate
    raise FileNotFoundError("Pass --data-dir (folder with cat .jpg files)")


def load_test_images(data_dir, n_images, pick_seed):
    transform = transforms.Compose(
        [
            transforms.Resize(int(IMAGE_SIZE * 1.15)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
    dataset = CatDataset(str(data_dir), transform=transform)
    total = len(dataset)
    train_size = total - VAL_SIZE - TEST_SIZE
    if train_size <= 0:
        raise ValueError(f"Dataset too small: {total} images")

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    _, _, test_set = random_split(
        dataset, [train_size, VAL_SIZE, TEST_SIZE], generator=generator
    )

    n = min(n_images, len(test_set))
    g = torch.Generator().manual_seed(pick_seed)
    indices = torch.randperm(len(test_set), generator=g)[:n].tolist()

    images = torch.stack([test_set[i] for i in indices])
    return images, indices


def find_checkpoint(beta, k):
    for base in MODEL_SEARCH_DIRS:
        path = base / f"checkpoints_{beta}_{k}" / "vqvae_best.pt"
        if path.exists():
            return path
    return MODEL_SEARCH_DIRS[0] / f"checkpoints_{beta}_{k}" / "vqvae_best.pt"


def load_vqvae(beta, k, ckpt_path):
    model = VQ_VAE(
        in_channels=3,
        hidden_dim=128,
        embedding_dim=256,
        num_embedding=k,
        n_residual=2,
        beta=beta,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def reconstruct(model, images):
    x = images.to(DEVICE)
    x_recon, _, _ = model(x)
    return x_recon.cpu()


def save_comparison_grid(originals, reconstructions, out_path, title_suffix, padding):
    n = originals.shape[0]
    expected = N_TEST_IMAGES
    if n != expected:
        raise ValueError(f"Need {expected} test images for {GRID_ROWS}x{COLS_PER_ROW} grid, got {n}")

    tiles = []
    for i in range(n):
        tiles.append(originals[i])
        tiles.append(reconstructions[i])

    grid = vutils.make_grid(
        tiles,
        nrow=COLS_PER_ROW,
        padding=padding,
        pad_value=1.0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, out_path)
    print(f"  saved {out_path}  ({title_suffix}, {GRID_ROWS} rows x {COLS_PER_ROW} cols)")


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Data: {data_dir}")
    print(f"Output: {out_dir}")

    originals, test_indices = load_test_images(data_dir, args.n_images, args.seed)
    print(f"Test images: {originals.shape[0]}  (subset indices {test_indices})")

    shared_path = out_dir / "test_originals.png"
    vutils.save_image(
        vutils.make_grid(originals, nrow=GRID_ROWS, padding=args.padding),
        shared_path,
    )
    print(f"  saved {shared_path}  (originals only, {GRID_ROWS} rows x 3 cols)")

    for beta in BETAS:
        for k in CODEBOOK_SIZES:
            ckpt = find_checkpoint(beta, k)
            tag = f"beta{beta}_K{k}"
            print(f"\n{tag}")

            if not ckpt.exists():
                print(f"  SKIP missing checkpoint: {ckpt}")
                continue

            model = load_vqvae(beta, k, ckpt)
            recons = reconstruct(model, originals)
            mse = ((originals - recons) ** 2).mean().item()
            print(f"  test MSE: {mse:.6f}")

            save_comparison_grid(
                originals,
                recons,
                out_dir / f"recon_{tag}.png",
                tag,
                args.padding,
            )

    summary = out_dir / "README.txt"
    summary.write_text(
        "VQ-VAE test reconstructions (encode -> quantize -> decode).\n"
        "9 files recon_beta*_K*.png (one per VQ-VAE config).\n"
        f"Each grid: {GRID_ROWS} rows x {COLS_PER_ROW} images "
        f"({COLS_PER_ROW // 2} pairs per row: original | reconstruction).\n"
        f"Same {N_TEST_IMAGES} test images (split seed {SPLIT_SEED}) for all configs.\n",
        encoding="utf-8",
    )
    print(f"\nDone. Grids in {out_dir}")


if __name__ == "__main__":
    main()
