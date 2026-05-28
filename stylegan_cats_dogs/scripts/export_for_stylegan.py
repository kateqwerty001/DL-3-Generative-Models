import sys
from pathlib import Path
import torchvision.utils as vutils

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

sys.path.append(str(REPO_ROOT / "datasets")) 
from cats_dogs_datasets import get_cats_dogs_dataloaders

out_dir = REPO_ROOT.parent / "data_stylegan_raw"
out_dir.mkdir(parents=True, exist_ok=True)

DATA_DIR = REPO_ROOT.parent / "data" / "cats_dogs" # select folder with cats and dogs 

print("Loading data from: {}".format(DATA_DIR))
print("Saving images to: {}".format(out_dir))

train_loader, _, _ = get_cats_dogs_dataloaders(
    root_dir=str(DATA_DIR), 
    batch_size=64, 
    image_size=128,
    num_workers=1, 
    seed=42
)

saved = 0
for batch in train_loader:
    for img in batch:
        save_path = str(out_dir) + "/{:05d}.png".format(saved)
        vutils.save_image(img, save_path)
        saved += 1
        
print("Successfully saved {} images!".format(saved))