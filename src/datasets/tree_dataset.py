from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TreePointDataset(Dataset):
    """
    Dataset loader for preprocessed tree point clouds (.npy files).
    Output shape per sample: (2048, 3) float32 tensor.
    """
    def __init__(self, data_dir: str = "data/processed", transform=None):
        self.data_dir = Path(data_dir)
        self.file_paths = sorted(list(self.data_dir.glob("*.npy")))
        self.transform = transform

        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .npy tree point cloud files found in '{self.data_dir}'.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path = self.file_paths[idx]
        pts = np.load(file_path).astype(np.float32)  # (2048, 3)
        
        tensor_pts = torch.from_numpy(pts)
        if self.transform:
            tensor_pts = self.transform(tensor_pts)
            
        return tensor_pts


def get_tree_dataloader(data_dir: str = "data/processed", batch_size: int = 4, shuffle: bool = True) -> DataLoader:
    """Helper function creating a DataLoader optimized for M3 Mac Unified Memory."""
    dataset = TreePointDataset(data_dir=data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for MPS stability inside Jupyter
        pin_memory=False
    )
