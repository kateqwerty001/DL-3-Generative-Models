from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CatDataset(Dataset):
    """Custom dataset for loading cat images."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_dir.rglob('*.jpg'))
        
        if len(self.image_paths) == 0:
            print(f"Warning: No .jpg files found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def get_cat_dataloader(root_dir, batch_size=64, image_size=128, num_workers=4, model_type="gan"):
    """Creates a DataLoader with transforms. model_type can be 'gan' or 'vae'."""
    
    transform_list = [
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    
    if model_type == "gan":
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
    transform = transforms.Compose(transform_list)

    dataset = CatDataset(root_dir=root_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

def visualize_batch(dataloader, model_type="gan", n=8, save_path=None):
    """Show a batch of images from the dataloader."""
    import matplotlib.pyplot as plt

    batch = next(iter(dataloader))
    imgs = batch[:n]

    if model_type == "gan":
        imgs = imgs * 0.5 + 0.5  # [-1, 1] -> [0, 1]

    imgs = imgs.permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    for ax, img in zip(axes, imgs):
        ax.imshow(img.clip(0, 1))
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    loader = get_cat_dataloader("data/cats", model_type="gan", num_workers=0)
    visualize_batch(loader, model_type="gan")