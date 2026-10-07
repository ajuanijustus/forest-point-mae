import torch
import torch.nn as nn


def pure_pytorch_fps(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest Point Sampling (FPS) in pure PyTorch (MPS compatible)."""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def pure_pytorch_knn(xyz: torch.Tensor, centroids_idx: torch.Tensor, k: int):
    """Extracts k-NN patches around centers and converts to center-relative coordinates."""
    B, N, _ = xyz.shape
    G = centroids_idx.shape[1]

    # Gather 3D center points using integer indices
    batch_idx = torch.arange(B, device=xyz.device).unsqueeze(-1).expand(-1, G)
    center_pts = xyz[batch_idx, centroids_idx]  # (B, G, 3)

    # Compute pairwise Euclidean distances
    dists = torch.cdist(center_pts, xyz)  # (B, G, N)

    # Pick k nearest neighbors
    knn_idx = torch.topk(dists, k=k, largest=False, dim=-1)[1]  # (B, G, k)

    # Gather neighbor coordinates
    knn_idx_flat = knn_idx.view(B, -1)
    batch_idx_flat = torch.arange(B, device=xyz.device).unsqueeze(-1).expand(-1, G * k)
    grouped_pts = xyz[batch_idx_flat, knn_idx_flat].view(B, G, k, 3)

    # Normalize to center-relative coordinates
    grouped_pts_norm = grouped_pts - center_pts.unsqueeze(2)

    return grouped_pts_norm, center_pts


class Point3DPatchEmbedder(nn.Module):
    """
    3D Patch Embedding Layer for Forest Point-MAE.
    Maps (B, 2048, 3) point cloud -> (B, num_patches, embed_dim) patch tokens.
    """
    def __init__(self, num_patches: int = 64, patch_size: int = 32, embed_dim: int = 384):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, xyz: torch.Tensor):
        # 1. Sample patch center points via FPS
        center_indices = pure_pytorch_fps(xyz, self.num_patches)
        
        # 2. Extract k-NN patches & normalize
        grouped_pts_norm, center_pts = pure_pytorch_knn(xyz, center_indices, self.patch_size)  # (B, G, k, 3)
        
        # 3. Mini PointNet MLP + Max Pooling across patch points
        patch_features = self.patch_encoder(grouped_pts_norm)  # (B, G, k, embed_dim)
        patch_tokens = torch.max(patch_features, dim=2)[0]    # (B, G, embed_dim)

        return patch_tokens, center_pts, grouped_pts_norm