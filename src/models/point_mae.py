import torch
import torch.nn as nn
from src.models.patch_embed import Point3DPatchEmbedder


def pure_pytorch_chamfer_loss(pred_patches: torch.Tensor, target_patches: torch.Tensor) -> torch.Tensor:
    """Computes bidirectional Chamfer Distance Loss between predicted and target 3D patches."""
    B, M, K, C = pred_patches.shape
    P = pred_patches.view(-1, K, C)
    T = target_patches.view(-1, K, C)

    dist_mat = torch.cdist(P, T, p=2) ** 2

    min_dist_P_to_T = torch.min(dist_mat, dim=2)[0]
    min_dist_T_to_P = torch.min(dist_mat, dim=1)[0]

    chamfer_loss = torch.mean(min_dist_P_to_T) + torch.mean(min_dist_T_to_P)
    return chamfer_loss


class PointMAE(nn.Module):
    """Full 3D Point-MAE Architecture for Forest Point Clouds."""

    def __init__(
        self,
        num_patches: int = 64,
        patch_size: int = 32,
        embed_dim: int = 384,
        decoder_embed_dim: int = 192,
        encoder_depth: int = 6,
        decoder_depth: int = 4,
        num_heads: int = 6,
        mask_ratio: float = 0.70
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_vis = int(num_patches * (1.0 - mask_ratio))

        # 1. Patch Embedding & Positional Encoders
        self.patch_embedder = Point3DPatchEmbedder(
            num_patches=num_patches, 
            patch_size=patch_size, 
            embed_dim=embed_dim
        )
        
        self.pos_embed_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        self.pos_embed_decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, decoder_embed_dim)
        )

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        # 3. Encoder -> Decoder Projection & Mask Token
        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # 4. Transformer Decoder (Uses even head count num_heads=6 for MPS fast-path)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, 
            nhead=num_heads, 
            dim_feedforward=decoder_embed_dim * 4,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # 5. Reconstruction Head
        self.reconstruction_head = nn.Linear(decoder_embed_dim, patch_size * 3)

    def random_masking(self, tokens: torch.Tensor, center_pts: torch.Tensor):
        """Randomly selects visible vs. masked patch indices per sample."""
        B, G, C = tokens.shape
        num_vis = self.num_vis

        noise = torch.rand(B, G, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_vis = ids_shuffle[:, :num_vis]
        ids_mask = ids_shuffle[:, num_vis:]

        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(-1)
        vis_tokens = tokens[batch_idx, ids_vis]
        vis_centers = center_pts[batch_idx, ids_vis]
        mask_centers = center_pts[batch_idx, ids_mask]

        return vis_tokens, vis_centers, mask_centers, ids_restore, ids_mask

    def forward(self, xyz: torch.Tensor):
        B, N, _ = xyz.shape

        # Step A: Patch Embedding
        tokens, center_pts, grouped_pts_norm = self.patch_embedder(xyz)

        # Step B: Masking
        vis_tokens, vis_centers, mask_centers, ids_restore, ids_mask = self.random_masking(tokens, center_pts)

        # Step C: Encoder Forward Pass
        vis_pos = self.pos_embed_encoder(vis_centers)
        encoder_input = vis_tokens + vis_pos
        encoded_vis_tokens = self.encoder(encoder_input)

        # Step D: Decoder Assembly
        vis_decoder_tokens = self.decoder_proj(encoded_vis_tokens) + self.pos_embed_decoder(vis_centers)
        
        num_masked = self.num_patches - self.num_vis
        mask_tokens_expanded = self.mask_token.expand(B, num_masked, -1) + self.pos_embed_decoder(mask_centers)

        full_decoder_tokens = torch.cat([vis_decoder_tokens, mask_tokens_expanded], dim=1)
        
        batch_idx = torch.arange(B, device=xyz.device).unsqueeze(-1)
        full_decoder_tokens = full_decoder_tokens[batch_idx, ids_restore]

        # Step E: Decoder Forward Pass
        decoded_tokens = self.decoder(full_decoder_tokens)

        # Step F: Predict 3D Coordinates for Masked Patches
        masked_decoded_tokens = decoded_tokens[batch_idx, ids_mask]
        pred_rel_coords = self.reconstruction_head(masked_decoded_tokens)
        pred_rel_coords = pred_rel_coords.view(B, num_masked, self.patch_size, 3)

        # Step G: Extract Ground Truth Masked Patches from Step A
        gt_masked_patches = grouped_pts_norm[batch_idx, ids_mask]

        # Step H: Chamfer Loss
        loss = pure_pytorch_chamfer_loss(pred_rel_coords, gt_masked_patches)

        return loss, pred_rel_coords, gt_masked_patches