"""
Compute FID for each StyleGAN2-ADA checkpoint x truncation_psi combination.
Also saves 10x10 grid of generated images for each combination.
Results saved to metrics_output/stylegan_fid_results.csv
"""

import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
import torch_fidelity
import pickle

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

sys.path.append(str(REPO_ROOT / "datasets"))
sys.path.append(str(REPO_ROOT / "stylegan2-ada-pytorch"))

from cat_datasets import get_cat_dataloaders

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────

RUN_DIR         = Path("/home2/faculty/kbokhan/DL-3-Generative-Models/stylegan_runs_cats/00000-cats128-mirror-auto1-kimg1000-noaug")

N_GEN           = 500
N_REAL          = 500
REAL_DIR        = Path("outputs/fid_real")
LOG_PATH        = Path("outputs_cats/stylegan_fid_results.csv")
GRIDS_DIR       = Path("outputs_cats/stylegan_grids")
DATA_DIR        = Path("/home2/faculty/kbokhan/data/cats")

TRUNCATIONS     = [0.1, 0.8, 0.85, 0.9, 0.95, 1.0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────

def get_checkpoints():
    return sorted(RUN_DIR.glob("network-snapshot-*.pkl"))


def ensure_real_images():
    existing = list(REAL_DIR.glob("*.png"))
    if len(existing) >= N_REAL:
        print(f"  Real images already cached ({len(existing)}), skipping.")
        return
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    _, _, test_loader = get_cat_dataloaders(
        DATA_DIR, batch_size=16, image_size=128,
        num_workers=1, model_type="vqvae",
        val_size=500, test_size=N_REAL, seed=42,
    )
    saved = 0
    for batch in test_loader:
        for img in batch:
            vutils.save_image(img, REAL_DIR / f"{saved:05d}.png")
            saved += 1
            if saved >= N_REAL:
                break
        if saved >= N_REAL:
            break
    print(f"  Saved {saved} real test images to {REAL_DIR}")


def generate_images(G, out_dir, n_images, truncation_psi):
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.png"))
    if len(existing) >= n_images:
        print(f"    Already generated ({len(existing)}), skipping.")
        return

    generated = 0
    batch_size = 16

    while generated < n_images:
        torch.manual_seed(42 + generated)
        n_this = min(batch_size, n_images - generated)
        z = torch.randn(n_this, G.z_dim, device=DEVICE)
        c = torch.zeros(n_this, G.c_dim, device=DEVICE) if G.c_dim > 0 else None

        with torch.no_grad():
            imgs = G(z, c, truncation_psi=truncation_psi)
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)

        for i, img in enumerate(imgs):
            vutils.save_image(img, out_dir / f"{generated + i:05d}.png")

        generated += n_this
        print(f"      {generated}/{n_images}", end="\r")

    print()


def save_grid(out_dir, grid_path, n=100):
    """Save 10x10 grid from first 100 images in out_dir."""
    from PIL import Image
    import torchvision.transforms.functional as TF

    files = sorted(out_dir.glob("*.png"))[:n]
    imgs  = [TF.to_tensor(Image.open(f).convert("RGB")) for f in files]
    grid  = vutils.make_grid(torch.stack(imgs), nrow=10, padding=2)
    vutils.save_image(grid, grid_path)
    print(f"    Grid saved to {grid_path}")


# ────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)

    if not LOG_PATH.exists():
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w") as f:
            f.write("checkpoint,kimg,truncation_psi,FID\n")

    print("\nPreparing real test images...")
    ensure_real_images()

    checkpoints = get_checkpoints()
    print(f"\nFound {len(checkpoints)} checkpoints, {len(TRUNCATIONS)} truncations")
    print(f"Total runs: {len(checkpoints) * len(TRUNCATIONS)}\n")

    results = []

    for pkl_path in checkpoints:
        kimg = int(pkl_path.stem.split("-")[-1])

        print(f"\n{'='*55}")
        print(f"  Loading {pkl_path.name}  (kimg={kimg})")
        print(f"{'='*55}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        G = data["G_ema"].to(DEVICE)
        G.eval()

        for psi in TRUNCATIONS:
            psi_str = str(psi).replace(".", "")
            tag     = f"kimg{kimg:06d}_psi{psi_str}"

            print(f"\n  truncation_psi={psi}")

            gen_dir   = Path(f"outputs/stylegan_generated_{tag}")
            grid_path = GRIDS_DIR / f"grid_{tag}.png"

            # generate images
            print(f"    Generating {N_GEN} images...")
            generate_images(G, gen_dir, N_GEN, psi)

            # 10x10 grid
            save_grid(gen_dir, grid_path)

            # FID
            print(f"    Computing FID...")
            metrics = torch_fidelity.calculate_metrics(
                input1=str(gen_dir),
                input2=str(REAL_DIR),
                fid=True, isc=False, kid=False, verbose=False,
            )
            fid = metrics["frechet_inception_distance"]
            print(f"    FID = {fid:.2f}")

            results.append((pkl_path.name, kimg, psi, fid))
            with open(LOG_PATH, "a") as f:
                f.write(f"{pkl_path.name},{kimg},{psi},{fid:.4f}\n")

        del G
        torch.cuda.empty_cache()

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  RESULTS")
    print(f"{'='*55}")
    print(f"  {'checkpoint':<35} {'psi':>6}  {'FID':>8}")
    print(f"  {'-'*35} {'-'*6}  {'-'*8}")
    for name, kimg, psi, fid in results:
        print(f"  kimg={kimg:5d}  psi={psi:.2f}  FID={fid:.2f}")

    best = min(results, key=lambda x: x[3])
    print(f"\nBest: kimg={best[1]}  psi={best[2]}  FID={best[3]:.2f}")
    print(f"Results saved to {LOG_PATH}")
    print(f"Grids saved to   {GRIDS_DIR}/")


if __name__ == "__main__":
    main()