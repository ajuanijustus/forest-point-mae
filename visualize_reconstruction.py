from pathlib import Path
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.datasets.tree_dataset import get_tree_dataloader
from src.models.point_mae import PointMAE

# Hardware routing
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def visualize_point_mae_reconstruction(checkpoint_path: str = "checkpoints/point_mae_best.pt"):
    print(f"🎨 Loading checkpoint '{checkpoint_path}' on {device}...")
    
    # 1. Load Model and Weights
    model = PointMAE(
        num_patches=64,
        patch_size=32,
        embed_dim=384,
        decoder_embed_dim=192,
        encoder_depth=6,
        decoder_depth=4,
        num_heads=6,
        mask_ratio=0.70
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Grab a test tree cloud
    loader = get_tree_dataloader(batch_size=1, shuffle=True)
    tree_batch = next(iter(loader)).to(device)  # (1, 2048, 3)

    with torch.no_grad():
        # Step A: Run patch embedder & masking internally
        tokens, center_pts, grouped_pts_norm = model.patch_embedder(tree_batch)
        vis_tokens, vis_centers, mask_centers, ids_restore, ids_mask = model.random_masking(tokens, center_pts)
        
        # Step B: Get predictions from full forward pass
        loss, pred_rel_coords, gt_masked_patches = model(tree_batch)

    # 3. Convert relative patch coordinates to absolute 3D space
    # Visible Patches
    B, G, K, C = grouped_pts_norm.shape
    batch_idx = torch.arange(B, device=device).unsqueeze(-1)
    
    ids_vis = ids_restore[:, :model.num_vis]
    vis_patches_norm = grouped_pts_norm[batch_idx, ids_vis]  # (1, num_vis, 32, 3)
    vis_patches_abs = (vis_patches_norm + vis_centers.unsqueeze(2)).reshape(-1, 3).cpu().numpy()

    # Ground Truth Masked Patches
    gt_masked_abs = (gt_masked_patches + mask_centers.unsqueeze(2)).reshape(-1, 3).cpu().numpy()

    # Predicted Reconstructed Masked Patches
    pred_masked_abs = (pred_rel_coords + mask_centers.unsqueeze(2)).reshape(-1, 3).cpu().numpy()

    print("📊 Generating 3D Interactive Comparison Plot...")

    # 4. Create Side-by-Side 3D Plotly Figure
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Visible (Green) + Ground Truth Masked (Gray)", "Visible (Green) + Reconstructed (Red)")
    )

    # Subplot 1: Ground Truth Input Cloud
    fig.add_trace(
        go.Scatter3d(
            x=vis_patches_abs[:, 0], y=vis_patches_abs[:, 1], z=vis_patches_abs[:, 2],
            mode='markers', marker=dict(size=2, color='forestgreen', opacity=0.8),
            name='Visible Patches (30%)'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(
            x=gt_masked_abs[:, 0], y=gt_masked_abs[:, 1], z=gt_masked_abs[:, 2],
            mode='markers', marker=dict(size=2, color='lightgray', opacity=0.5),
            name='Masked Ground Truth (70%)'
        ),
        row=1, col=1
    )

    # Subplot 2: Point-MAE Reconstruction
    fig.add_trace(
        go.Scatter3d(
            x=vis_patches_abs[:, 0], y=vis_patches_abs[:, 1], z=vis_patches_abs[:, 2],
            mode='markers', marker=dict(size=2, color='forestgreen', opacity=0.8),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter3d(
            x=pred_masked_abs[:, 0], y=pred_masked_abs[:, 1], z=pred_masked_abs[:, 2],
            mode='markers', marker=dict(size=2, color='crimson', opacity=0.7),
            name='Point-MAE Reconstructed'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=f"Point-MAE 3D Tree Reconstruction (Chamfer Loss: {loss.item():.5f})",
        width=1200,
        height=600,
        margin=dict(r=10, l=10, b=10, t=50)
    )

    output_path = "reconstruction_vis.html"
    fig.write_html(output_path)
    print(f"✨ Visualization saved to: {Path(output_path).resolve()}")
    print("👉 Open 'reconstruction_vis.html' in your browser to inspect the 3D trees!")

if __name__ == "__main__":
    visualize_point_mae_reconstruction()