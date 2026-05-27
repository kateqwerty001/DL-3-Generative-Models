import sys
sys.path.append("src")
from cat_datasets import get_cat_dataloaders
import torchvision.utils as vutils
from pathlib import Path

out_dir = Path("data_stylegan_raw")
out_dir.mkdir(exist_ok=True)

train_loader, _, _ = get_cat_dataloaders(
    "data", batch_size=64, image_size=128,
    num_workers=1, model_type="vqvae", seed=42
)

saved = 0
for batch in train_loader:
    for img in batch:
        vutils.save_image(img, out_dir / f"{saved:05d}.png")
        saved += 1
print(f"Saved {saved} images")