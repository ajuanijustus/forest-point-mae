import torch
from src.datasets.tree_dataset import get_tree_dataloader
from src.models.point_mae import PointMAE

# 1. Hardware setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚡ Testing Full Point-MAE Architecture on Device: {device}")

# 2. Load DataLoader batch
loader = get_tree_dataloader(batch_size=4, shuffle=True)
tree_batch = next(iter(loader)).to(device)  # (4, 2048, 3)

# 3. Instantiate PointMAE Model
model = PointMAE(
    num_patches=64,
    patch_size=32,
    embed_dim=384,
    decoder_embed_dim=192,
    encoder_depth=4,     # Compact for fast local execution
    decoder_depth=2,
    mask_ratio=0.70      # 70% masked patches
).to(device)

# 4. Forward Pass Test
model.train()
loss, pred_patches, gt_patches = model(tree_batch)

print(f"\n--- Point-MAE Forward Pass Verification ---")
print(f"Input Tree Cloud Shape:        {tree_batch.shape}")
print(f"Predicted Masked Patch Shape:  {pred_patches.shape} -> (Batch, Num_Masked, Patch_Size, 3)")
print(f"Ground Truth Masked Shape:     {gt_patches.shape}   -> (Batch, Num_Masked, Patch_Size, 3)")
print(f"Reconstruction Chamfer Loss:   {loss.item():.6f}")
print("✅ Phase 2 Point-MAE Autoencoder architecture fully operational on Apple Silicon!")
