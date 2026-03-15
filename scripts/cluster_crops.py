"""
scripts/cluster_crops.py

Clusters cat crop images into N groups using CLIP ViT-B/32 embeddings
and K-Means, then organises them into output folders for manual verification
before fine-tuning.

Model choice: CLIP ViT-B/32
  - ~600MB RAM, fast on CPU with 12 threads
  - Handles visible and IR frames well due to diverse pretraining
  - Better inter-individual separation than CNN-based models for this task

Usage:
    python scripts/cluster_crops.py --crops data/raw --n-clusters 2
    python scripts/cluster_crops.py --crops data/raw --n-clusters 2 --output data/clustered
    python scripts/cluster_crops.py --crops data/raw --n-clusters 2 --uncertainty-threshold 0.15
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_clip_model():
    """Load CLIP ViT-B/32 on CPU."""
    import clip
    import torch
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    return model, preprocess, device


def extract_embeddings(
    image_paths: list[Path],
    model,
    preprocess,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract CLIP image embeddings for a list of image paths.

    Processes in batches to avoid memory spikes.
    Returns an (N, 512) float32 array of L2-normalised embeddings.
    """
    import torch
    from PIL import Image

    all_embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"  [WARN] Skipping {p.name}: {e}")

        if not images:
            continue

        batch_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)  # L2 normalise

        all_embeddings.append(features.cpu().numpy().astype(np.float32))
        print(f"  Embedded {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")

    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run K-Means on embeddings.

    Returns
    -------
    labels : (N,) int array of cluster assignments
    distances : (N,) float array of distance to assigned cluster centre,
                NOT normalised — raw Euclidean distances in embedding space.
                Use with --uncertainty-percentile rather than a fixed threshold.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(embeddings)

    centres = km.cluster_centers_
    assigned_centres = centres[labels]
    distances = np.linalg.norm(embeddings - assigned_centres, axis=1)

    return labels, distances


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_clusters(
    image_paths: list[Path],
    labels: np.ndarray,
    distances: np.ndarray,
    output_dir: Path,
    n_clusters: int,
    uncertainty_threshold: float,
) -> dict:
    """
    Copy images into cluster_N/ or uncertain/ subdirectories.
    uncertainty_threshold is a raw distance value — images above it go to uncertain/.
    Returns a summary dict with counts per folder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create cluster folders
    for i in range(n_clusters):
        (output_dir / f"cluster_{i}").mkdir(exist_ok=True)
    (output_dir / "uncertain").mkdir(exist_ok=True)

    counts = {f"cluster_{i}": 0 for i in range(n_clusters)}
    counts["uncertain"] = 0

    for path, label, dist in zip(image_paths, labels, distances):
        if dist >= uncertainty_threshold:
            dest_folder = output_dir / "uncertain"
            counts["uncertain"] += 1
        else:
            dest_folder = output_dir / f"cluster_{label}"
            counts[f"cluster_{label}"] += 1

        shutil.copy2(path, dest_folder / path.name)

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_crop_paths(crops_dir: Path) -> list[Path]:
    """Collect all JPEG crop files from the given directory."""
    paths = sorted(
        p for p in crops_dir.glob("*_crop_*.jpg")
    )
    if not paths:
        # Fallback: accept any jpg if no crops found
        paths = sorted(crops_dir.glob("*.jpg"))
    return paths


def parse_args():
    p = argparse.ArgumentParser(description="Cluster cat crops using CLIP + K-Means")
    p.add_argument("--crops", default="data/raw",
                   help="Directory containing crop images (default: data/raw)")
    p.add_argument("--output", default="data/clustered",
                   help="Output directory for clustered images (default: data/clustered)")
    p.add_argument("--n-clusters", type=int, default=2,
                   help="Number of clusters / cats (default: 2)")
    p.add_argument("--uncertainty-percentile", type=float, default=20.0,
                   help="Top N%% of most uncertain crops go to uncertain/ folder (default: 20)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Embedding batch size (default: 64, reduce if RAM is tight)")
    p.add_argument("--random-state", type=int, default=42,
                   help="K-Means random seed for reproducibility")
    p.add_argument("--show-distances", action="store_true",
                   help="Print distance distribution stats to help tune --uncertainty-threshold")
    return p.parse_args()


def main():
    args = parse_args()

    crops_dir = Path(args.crops)
    output_dir = Path(args.output)

    if not crops_dir.exists():
        raise FileNotFoundError(f"Crops directory not found: {crops_dir}")

    # 1. Collect images
    print(f"\nScanning {crops_dir} for crop images...")
    image_paths = collect_crop_paths(crops_dir)
    if not image_paths:
        raise FileNotFoundError(f"No crop images found in {crops_dir}")
    print(f"Found {len(image_paths)} crop images")

    if len(image_paths) < args.n_clusters:
        raise ValueError(
            f"Not enough images ({len(image_paths)}) to form "
            f"{args.n_clusters} clusters"
        )

    # 2. Load CLIP
    print("\nLoading CLIP ViT-B/32 (downloading on first run ~600MB)...")
    model, preprocess, device = load_clip_model()
    print("Model loaded")

    # 3. Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(
        image_paths, model, preprocess, device, batch_size=args.batch_size
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # 4. Cluster
    print(f"\nClustering into {args.n_clusters} groups...")
    labels, distances = cluster_embeddings(
        embeddings, args.n_clusters, random_state=args.random_state
    )

    # 5. Distance distribution diagnostics
    percentile_values = [10, 25, 50, 75, 90, 95, 100]
    print(f"\n--- Distance Distribution (raw Euclidean) ---")
    for pv in percentile_values:
        print(f"  p{pv:3d}: {np.percentile(distances, pv):.4f}")
    print(f"  mean: {distances.mean():.4f}  std: {distances.std():.4f}")

    # Compute raw threshold from percentile
    uncertainty_threshold = float(np.percentile(distances, 100 - args.uncertainty_percentile))
    print(f"\n  Uncertainty percentile : top {args.uncertainty_percentile:.0f}%")
    print(f"  Raw distance threshold : {uncertainty_threshold:.4f}")
    print(f"  --> Confident assigns  : {(distances <= uncertainty_threshold).sum()}/{len(distances)}")
    print(f"  --> Uncertain          : {(distances > uncertainty_threshold).sum()}/{len(distances)}")

    # 6. Save
    print(f"\nSaving clusters to {output_dir}...")
    counts = save_clusters(
        image_paths, labels, distances, output_dir,
        n_clusters=args.n_clusters,
        uncertainty_threshold=uncertainty_threshold,
    )

    # 7. Summary
    print("\n--- Clustering Summary ---")
    for folder, count in counts.items():
        print(f"  {folder:12s}: {count} images")
    uncertain_pct = counts["uncertain"] / len(image_paths) * 100
    print(f"\n  Uncertain  : {uncertain_pct:.1f}% of images")
    print(f"  Output dir : {output_dir}")
    print(
        "\nNext step: open each cluster folder and verify assignments. "
        "Rename cluster_0/ and cluster_1/ to your cats' names once verified."
    )


if __name__ == "__main__":
    main()
