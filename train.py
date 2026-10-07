import time
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.datasets.tree_dataset import get_tree_dataloader
from src.models.point_mae import PointMAE

# =========================================================================
# Hyperparameters & Configuration
# =========================================================================
DATA_DIR = "data/processed"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Training Settings (Optimized for fanless M3 MacBook Air)
BATCH_SIZE = 8          # Small batch size keeps memory swap & thermal throttling low
EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
MASK_RATIO = 0.70

# Point-MAE Architecture Settings
NUM_PATCHES = 64
PATCH_SIZE = 32
EMBED_DIM = 384
DECODER_EMBED_DIM = 192
ENCODER_DEPTH = 6
DECODER_DEPTH = 4
NUM_HEADS = 6


def train_point_mae():
    # 1. Hardware Routing (MPS for Apple Silicon)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Initializing Point-MAE Pre-Training on Device: {device}")

    # 2. Dataset & DataLoader
    train_loader = get_tree_dataloader(
        data_dir=DATA_DIR, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    print(f"📊 Loaded dataset from '{DATA_DIR}' ({len(train_loader.dataset)} trees, {len(train_loader)} batches/epoch)")

    # 3. Model Setup
    model = PointMAE(
        num_patches=NUM_PATCHES,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        decoder_embed_dim=DECODER_EMBED_DIM,
        encoder_depth=ENCODER_DEPTH,
        decoder_depth=DECODER_DEPTH,
        num_heads=NUM_HEADS,
        mask_ratio=MASK_RATIO
    ).to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 5. Training Loop State
    best_loss = float("inf")
    start_total_time = time.time()

    print("\n" + "=" * 65)
    print(f"{'Epoch':^8} | {'Chamfer Loss':^14} | {'LR':^10} | {'Time/Epoch':^12}")
    print("=" * 65)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_epoch_loss = 0.0
        start_epoch_time = time.time()

        for batch_idx, tree_batch in enumerate(train_loader):
            tree_batch = tree_batch.to(device)

            optimizer.zero_grad()
            
            # Forward pass & Chamfer distance loss calculation
            loss, _, _ = model(tree_batch)
            
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        # Step scheduler
        scheduler.step()
        avg_loss = total_epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        epoch_duration = time.time() - start_epoch_time

        # Print progress
        print(f"{epoch:^8d} | {avg_loss:^14.6f} | {current_lr:^10.2e} | {epoch_duration:^10.2f}s")

        # Save Best Model Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = CHECKPOINT_DIR / "point_mae_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, checkpoint_path)

    total_duration = (time.time() - start_total_time) / 60.0
    print("=" * 65)
    print(f"🎉 Pre-Training Complete in {total_duration:.2f} minutes!")
    print(f"🏆 Best Chamfer Reconstruction Loss: {best_loss:.6f}")
    print(f"💾 Best checkpoint saved to: {CHECKPOINT_DIR / 'point_mae_best.pt'}")


if __name__ == "__main__":
    train_point_mae()