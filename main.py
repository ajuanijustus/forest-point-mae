import os
import urllib.request
from pathlib import Path
import laspy
import numpy as np
import open3d as o3d
import pandas as pd

# =========================================================================
# Configuration & Path Setup
# =========================================================================
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_POINTS = 2048
ZENODO_RECORD_ID = "13255198"
METADATA_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/tree_metadata_dev.csv?download=1"


# =========================================================================
# Helper Functions
# =========================================================================
def fetch_metadata() -> pd.DataFrame:
    """Downloads FOR-species20K development metadata CSV if not present."""
    metadata_path = RAW_DIR / "tree_metadata_dev.csv"
    if not metadata_path.exists():
        print(f"📥 Fetching dataset metadata from Zenodo ({METADATA_URL})...")
        urllib.request.urlretrieve(METADATA_URL, metadata_path)
        print("✅ Metadata downloaded successfully.")
    else:
        print("📁 Local metadata CSV found.")
    return pd.read_csv(metadata_path)


def generate_synthetic_tree(num_raw_points: int = 10000) -> np.ndarray:
    """
    Generates a realistic 3D tree point cloud (Trunk + Canopy)
    for quick local pipeline testing before full Zenodo dev.zip download.
    """
    # Trunk: Vertical cylinder with slight noise
    trunk_height = np.random.uniform(3.0, 6.0)
    trunk_pts_count = int(num_raw_points * 0.2)
    z_trunk = np.random.uniform(0, trunk_height, trunk_pts_count)
    radius_trunk = np.random.uniform(0.1, 0.3, trunk_pts_count)
    angle_trunk = np.random.uniform(0, 2 * np.pi, trunk_pts_count)
    x_trunk = radius_trunk * np.cos(angle_trunk) + np.random.normal(0, 0.02, trunk_pts_count)
    y_trunk = radius_trunk * np.sin(angle_trunk) + np.random.normal(0, 0.02, trunk_pts_count)
    trunk = np.column_stack((x_trunk, y_trunk, z_trunk))

    # Crown: Paraboloid/ellipsoid canopy
    canopy_pts_count = num_raw_points - trunk_pts_count
    u = np.random.uniform(0, 1, canopy_pts_count)
    v = np.random.uniform(0, 1, canopy_pts_count)
    theta = u * 2.0 * np.pi
    phi = np.arccos(2.0 * v - 1.0)
    
    rx, ry, rz = np.random.uniform(2.5, 4.0), np.random.uniform(2.5, 4.0), np.random.uniform(4.0, 7.0)
    x_canopy = rx * np.sin(phi) * np.cos(theta) + np.random.normal(0, 0.1, canopy_pts_count)
    y_canopy = ry * np.sin(phi) * np.sin(theta) + np.random.normal(0, 0.1, canopy_pts_count)
    z_canopy = rz * np.cos(phi) + trunk_height + (rz / 2.0)
    canopy = np.column_stack((x_canopy, y_canopy, z_canopy))

    return np.vstack((trunk, canopy)).astype(np.float32)


def read_las_point_cloud(filepath: Path) -> np.ndarray:
    """Reads a .las or .laz point cloud file into an (N, 3) float32 numpy array."""
    las = laspy.read(filepath)
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    return np.column_stack((x, y, z))


def preprocess_tree_point_cloud(points: np.ndarray, num_samples: int = 2048) -> np.ndarray:
    """
    1. Centers XY coordinates and aligns base (min Z) to origin (0, 0, 0).
    2. Performs Farthest Point Sampling (FPS) using Open3D down to `num_samples`.
    3. Normalizes coordinate magnitudes to unit sphere [-1, 1].
    """
    if len(points) == 0:
        raise ValueError("Cannot process empty point cloud array.")

    # 1. Ground Alignment & Centering
    centroid_xy = np.mean(points[:, :2], axis=0)
    min_z = np.min(points[:, 2])
    points[:, :2] -= centroid_xy
    points[:, 2] -= min_z

    # 2. Open3D Farthest Point Sampling (FPS)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if len(points) >= num_samples:
        sampled_pcd = pcd.farthest_point_down_sample(num_samples)
        sampled_points = np.asarray(sampled_pcd.points, dtype=np.float32)
    else:
        # Oversample if point count is less than target
        indices = np.random.choice(len(points), num_samples, replace=True)
        sampled_points = points[indices]

    # 3. Unit Sphere Normalization
    distances = np.linalg.norm(sampled_points, axis=1)
    max_radius = np.max(distances)
    if max_radius > 0:
        sampled_points /= max_radius

    return sampled_points


# =========================================================================
# Main Execution Pipeline
# =========================================================================
def main():
    print("🌲 Initializing FOR-species20K Preprocessing Pipeline...")
    
    # Fetch metadata table
    try:
        metadata = fetch_metadata()
        print(f"📊 Dataset contains {len(metadata)} tree metadata records across {metadata['species'].nunique()} species.")
    except Exception as e:
        print(f"⚠️ Metadata fetch warning: {e}. Proceeding with sample generation.")
        metadata = None

    # Check for raw LAS/LAZ files in data/raw
    raw_files = list(RAW_DIR.glob("*.las")) + list(RAW_DIR.glob("*.laz"))
    
    if len(raw_files) == 0:
        print(f"\n💡 No raw .las/.laz files found in `{RAW_DIR}`.")
        print(f"🛠️ Generating 10 sample tree point clouds for pipeline verification...")
        
        for i in range(1, 11):
            tree_id = f"sample_tree_{i:03d}"
            raw_pts = generate_synthetic_tree(num_raw_points=np.random.randint(8000, 15000))
            processed_pts = preprocess_tree_point_cloud(raw_pts, num_samples=TARGET_POINTS)
            
            # Save as NumPy binary array and 3D PLY file
            npy_path = PROCESSED_DIR / f"{tree_id}.npy"
            ply_path = PROCESSED_DIR / f"{tree_id}.ply"
            
            np.save(npy_path, processed_pts)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(processed_pts)
            o3d.io.write_point_cloud(str(ply_path), pcd)

        print(f"✅ Generated and processed 10 sample trees in `{PROCESSED_DIR}`.")

    else:
        print(f"\n📂 Processing {len(raw_files)} raw LAS/LAZ files from `{RAW_DIR}`...")
        for filepath in raw_files:
            tree_id = filepath.stem
            try:
                raw_pts = read_las_point_cloud(filepath)
                processed_pts = preprocess_tree_point_cloud(raw_pts, num_samples=TARGET_POINTS)
                
                npy_path = PROCESSED_DIR / f"{tree_id}.npy"
                ply_path = PROCESSED_DIR / f"{tree_id}.ply"
                
                np.save(npy_path, processed_pts)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(processed_pts)
                o3d.io.write_point_cloud(str(ply_path), pcd)
                
                print(f"  ✓ Processed: {tree_id} -> Shape: {processed_pts.shape}")
            except Exception as err:
                print(f"  ❌ Error processing {filepath.name}: {err}")

    # Summary Check
    processed_files = list(PROCESSED_DIR.glob("*.npy"))
    print(f"\n🎉 Done! {len(processed_files)} tree point clouds (N={TARGET_POINTS}) ready in `{PROCESSED_DIR}`.")


if __name__ == "__main__":
    main()