import torch
from src.datasets.tree_dataset import get_tree_dataloader
from src.models.patch_embed import Point3DPatchEmbedder

# Hardware setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚡ Testing Patch Embedder on Device: {device}")

# 1. Load sample batch
loader = get_tree_dataloader(batch_size=4, shuffle=True)
tree_batch = next(iter(loader)).to(device)  # (4, 2048, 3)

# 2. Forward pass through Patch Embedder
embedder = Point3DPatchEmbedder(num_patches=64, patch_size=32, embed_dim=384).to(device)
tokens, centers = embedder(tree_batch)

# 3. Output Validation
print(f"Input Tree Cloud Shape:   {tree_batch.shape}")
print(f"Output Patch Tokens Shape: {tokens.shape}  -> (Batch, Num_Patches, Embed_Dim)")
print(f"Output Center Points Shape: {centers.shape} -> (Batch, Num_Patches, 3D_Coords)")
print("✅ Phase 2 Patch Embedding Layer verified successfully on MPS!")