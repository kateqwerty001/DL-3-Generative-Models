from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class CatsDogsDataset(Dataset):
    """Custom dataset for loading cat and dog images."""
    
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

def get_cats_dogs_dataloaders(root_dir, batch_size=64, image_size=128, num_workers=4, model_type="vqvae", val_size=500, test_size=500, seed=42):
    """Creates Train, Val, and Test DataLoaders."""
    
    transform_list = [
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    
    if model_type == "gan":
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
    transform = transforms.Compose(transform_list)
    dataset = CatsDogsDataset(root_dir=root_dir, transform=transform)
    
    total_size = len(dataset)
    train_size = total_size - val_size - test_size
    
    if train_size <= 0:
        raise ValueError(f"Dataset is too small ({total_size} images) for the specified val ({val_size}) and test ({test_size}) sizes.")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )
    
    def make_loader(subset, shuffle, drop_last):
        return DataLoader(
            subset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last
        )
    
    train_loader = make_loader(train_set, shuffle=True, drop_last=True)
    val_loader = make_loader(val_set, shuffle=False, drop_last=False)
    test_loader = make_loader(test_set, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader