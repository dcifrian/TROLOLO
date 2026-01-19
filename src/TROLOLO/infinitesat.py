# Copyright 2025 Diego Cifrian Bueno
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import zipfile
import json
import argparse
from datetime import datetime

import torchvision
from copernicusapi import QueryConstructor
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import rasterio
import numpy as np
from rasterio.plot import show
from skimage import exposure
from PIL import Image
import os
import torch
import shutil
import random
from collections import defaultdict

from torch.utils.data import DataLoader

from TROLOLO_pyramid import TROLOLO

def load_or_create_index(index_path="data/infinitesat/index.json"):
    """Load existing index or create new one"""
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            return json.load(f)
    else:
        return {
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "total_patches": 0,
                "total_tiles": 0
            },
            "tiles": []
        }


def save_index(index_data, index_path="data/infinitesat/index.json"):
    """Save index to JSON file"""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)


def extract_tile_info_from_safe_folder(path=None):
    """Extract tile information from .SAFE folder name"""
    safe_folders = glob.glob('data/infinitesat/*.SAFE')
    if path is not None:
        safe_folders = glob.glob(path)
    if not safe_folders:
        return None

    safe_folder = os.path.basename(safe_folders[0])  # Get just the folder name

    # Parse: S2B_MSIL2A_20250612T102559_N0511_R108_T32UPA_20250612T131505.SAFE
    parts = safe_folder.replace('.SAFE', '').split('_')

    return {
        "tile_id": safe_folder.replace('.SAFE', ''),
        "satellite": parts[0],  # S2B
        "date": datetime.strptime(parts[2][:8], '%Y%m%d').strftime('%Y-%m-%d'),
        "utm_tile": parts[5]  # T32UPA
    }


def is_tile_already_processed(index_data, tile_id):
    """Check if tile already exists in index"""
    for tile in index_data["tiles"]:
        if tile["tile_id"] == tile_id:
            return True
    return False


def querySentinel(aoi_polygon, time1, time2):
    """Query Sentinel with given AOI polygon"""
    # Clear previous filters
    query_constructor = QueryConstructor()

    # define the collection
    query_constructor.add_collection_filter('sentinel-2')

    # define product type
    query_constructor.add_product_type_filter('l2a')

    # define AOI via a shapely geometry Point or Polygon in WGS84
    query_constructor.add_aoi_filter(aoi_polygon)

    # define the timeframe based on sensing start date
    query_constructor.add_sensing_start_date_filter(time1, time2)

    # search for products with a cloud cover <= 1%
    query_constructor.add_cloud_cover_filter(0.01)

    # Check and send query
    n_products = query_constructor.check_query()
    print(f"Found {n_products} products")
    result = None
    products = None
    if n_products > 0:
        products, result = query_constructor.send_query()
    return products, result


def downloadProduct(products):
    """Download and extract Sentinel-2 product"""
    # Your client credentials
    client_id = "sh-ec0a9a2e-6352-4ebd-b368-3431440bd00d"
    client_secret = "LWvDdCEeO3JZjgpjRQUw8Kv67qrzApUK"
    file_name = products["file_name"].iloc[0]
    if len(glob.glob('**/' + file_name, recursive=True)) == 0:

        # Create a session
        client = BackendApplicationClient(client_id=client_id)
        oauth = OAuth2Session(client=client)

        # Get token for the session
        token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                                  client_secret=client_secret, include_client_id=True)

        if isinstance(products, tuple):
            products = products[0]

        resp = oauth.get(products["download_url"][0])

        # Save it as binary
        with open("data/infinitesat/sentinel_data.zip", "wb") as f:
            f.write(resp.content)

        print(f"Saved {len(resp.content)} bytes")

        # Extract
        with zipfile.ZipFile("data/infinitesat/sentinel_data.zip", 'r') as zip_ref:
            zip_ref.extractall("data/infinitesat/")
            print("Extracted files")

        # Clean up zip file
        os.remove("data/infinitesat/sentinel_data.zip")
    else:
        print("Was already downloaded")


def extract_patches_to_jpg(folder="data/infinitesat/", SAFE=None, patch_size=64):
    """Extract patches and return statistics for indexing"""

    # Get tile info from SAFE folder
    path = None
    if SAFE is not None:
        path = glob.glob(folder + '**' + SAFE, recursive=True)[0]
        print(path)

    tile_info = extract_tile_info_from_safe_folder(path=path)
    if not tile_info:
        print("No .SAFE folder found!")
        return None

    # Load index (no need to check here since we check before download)
    index_data = load_or_create_index()

    output_dir = folder + "unbalanced/images"
    os.makedirs(output_dir, exist_ok=True)

    # Open bands directly
    r_path = glob.glob(path + '/**/*B04_10m.jp2', recursive=True)[0]  # Red
    g_path = glob.glob(path + '/**/*B03_10m.jp2', recursive=True)[0]  # Green
    b_path = glob.glob(path + '/**/*B02_10m.jp2', recursive=True)[0]  # Blue

    with rasterio.open(r_path) as r, rasterio.open(g_path) as g, rasterio.open(b_path) as b:
        # Read all bands at once
        red = r.read(1)
        green = g.read(1)
        blue = b.read(1)

        # Stack into RGB
        rgb = np.stack([red, green, blue], axis=-1)

        # CRITICAL: Only compute statistics on VALID pixels (non-zero)
        valid_mask = np.sum(rgb, axis=2) > 3
        valid_pixels = rgb[valid_mask]
        valid_area_percent = (np.sum(valid_mask) / valid_mask.size) * 100

        # Compute percentiles only on valid data
        p2, p98 = np.percentile(valid_pixels, (2, 98))

        # Apply normalization (keep in float)
        rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0, 1)

        # Apply gamma correction (still in 0-1 range)
        gamma = 1.2
        rgb_norm = np.power(rgb_norm, 1 / gamma)

        # Reduce contrast to match EuroSAT
        target_contrast = 0.7  # Reduces std from ~58 to ~40
        rgb_norm = (rgb_norm - 0.5) * target_contrast + 0.5

        # Convert to uint8 only at the very end
        rgb_norm = (rgb_norm * 255).astype(np.uint8)

        # Extract patches and save
        h, w = rgb_norm.shape[:2]
        patch_count = 0
        current_tile_count = index_data["metadata"]["total_tiles"]
        patch_prefix = f"tile_{current_tile_count}"

        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                # Check if ALL pixels in this patch are valid
                patch_mask = valid_mask[y:y + patch_size, x:x + patch_size]

                # Skip if ANY pixel in the patch is invalid
                if np.all(patch_mask):
                    patch = rgb_norm[y:y + patch_size, x:x + patch_size]

                    # Save as JPEG with tile prefix
                    patch_img = Image.fromarray(patch)
                    filename = f"{patch_prefix}_{patch_count}_{x}_{y}.jpg"
                    patch_img.save(os.path.join(output_dir, filename), "JPEG", quality=75, subsampling=2)
                    patch_count += 1

        print(f"Extracted {patch_count} patches to {output_dir}")

        # Calculate statistics for the extracted patches
        if patch_count > 0:
            # Get stats from the actual saved patches (only valid areas)
            valid_rgb = rgb_norm[valid_mask.any(axis=-1) if len(valid_mask.shape) == 3 else valid_mask]
            stats = {
                "mean": float(rgb_norm[valid_mask].mean()),
                "std": float(rgb_norm[valid_mask].std()),
                "valid_area_percent": float(valid_area_percent)
            }
        else:
            stats = {"mean": 0, "std": 0, "valid_area_percent": 0}

        return {
            "tile_info": tile_info,
            "patch_count": patch_count,
            "patch_prefix": patch_prefix,
            "stats": stats
        }


def create_balanced_dataset(dataloader, model,
                            output_dir="data/infinitesat/balanced/",
                            device="cuda"):
    """
    Create a balanced dataset by classifying patches from a DataLoader and sampling equally from each class
    Uses the minimum class count as the target for all classes
    """
    # EuroSAT class names (adjust order to match your model)
    class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                   'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
                   'River', 'SeaLake']

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    # Set model to eval mode
    model.eval()
    model = model.to(device)

    # Store image indices grouped by predicted class
    class_indices = defaultdict(list)

    print("Classifying images in batches...")
    total_processed = 0
    current_idx = 0

    with (torch.compiler.set_stance("force_eager"), torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16),torch.inference_mode()):
        for batch_idx, (images, _) in enumerate(dataloader):  # Ignore labels (all zeros)
            images = images.to(device)

            # Get predictions for the batch
            outputs = model(images)
            predicted_classes = torch.argmax(outputs, dim=1)

            # Group image indices by predicted class
            for i, predicted_class in enumerate(predicted_classes):
                class_idx = predicted_class.item()
                global_idx = current_idx + i
                class_indices[class_idx].append(global_idx)

            current_idx += len(images)
            total_processed += len(images)

            if batch_idx % 10 == 0:
                print(f"Processed {total_processed} images in {batch_idx + 1} batches")

    # Print class distribution
    print("\nClass distribution:")
    class_counts = []
    for i, class_name in enumerate(class_names):
        count = len(class_indices[i])
        class_counts.append(count)
        print(f"{class_name}: {count} images")

    # Find minimum class count (excluding classes with 0 images)
    non_zero_counts = [count for count in class_counts if count > 0]
    if not non_zero_counts:
        print("Error: No images classified into any class!")
        return 0

    samples_per_class = min(non_zero_counts)
    samples_per_class = 1000
    print(f"\nUsing minimum class count as target: {samples_per_class} samples per class")

    # Sample balanced dataset
    total_copied = 0

    for class_idx, class_name in enumerate(class_names):
        available_indices = class_indices[class_idx]

        if len(available_indices) == 0:
            print(f"Warning: No images found for {class_name}")
            continue

        if len(available_indices) >= samples_per_class:
            # Randomly sample
            selected_indices = random.sample(available_indices, samples_per_class)
        else:
            # Use all available images (shouldn't happen since we use min count)
            selected_indices = available_indices

        # Copy selected images using dataset paths
        for idx in selected_indices:
            # Get the file path from the dataset
            img_path, _ = dataloader.dataset.samples[idx]  # ImageFolder format
            img_filename = os.path.basename(img_path)

            dst_path = os.path.join(output_dir, class_name, img_filename)
            shutil.copy2(img_path, dst_path)
            total_copied += 1

    print(f"Created balanced dataset with {total_copied} total images in {output_dir}")
    print(f"Each available class has {samples_per_class} samples")
    return total_copied


def update_index_with_tile(extraction_result, aoi_polygon):
    """Update index with new tile information"""
    if not extraction_result:
        return

    index_data = load_or_create_index()

    # Create new tile entry
    new_tile = {
        **extraction_result["tile_info"],
        "polygon": {
            "type": "Polygon",
            "coordinates": [list(aoi_polygon.exterior.coords)]
        },
        "download_date": datetime.now().isoformat(),
        "processing_status": "completed",
        "patch_count": extraction_result["patch_count"],
        "patch_prefix": extraction_result["patch_prefix"],
        "stats": extraction_result["stats"]
    }

    # Add to index
    index_data["tiles"].append(new_tile)

    # Update metadata
    index_data["metadata"]["total_patches"] += extraction_result["patch_count"]
    index_data["metadata"]["total_tiles"] += 1
    index_data["metadata"]["last_updated"] = datetime.now().isoformat()

    # Save updated index
    save_index(index_data)
    print(f"Updated index: {extraction_result['patch_count']} patches from {extraction_result['tile_info']['tile_id']}")


def process_region(aoi_polygon, time1, time2):
    """Process a complete region: query, download, extract, and index"""
    print(f"Processing region with {len(aoi_polygon.exterior.coords)} vertices")

    # Query for products
    products, result = querySentinel(aoi_polygon, time1, time2)

    if products is None or len(products) == 0:
        print("No products found for this region")
        return

    # Check if we already have this tile before downloading
    index_data = load_or_create_index()

    # Use group_tile_id or file_name to check if already processed
    if "file_name" in products.columns and len(products) > 0:
        tile_id = products["file_name"].iloc[0].split('.')[0]  # Remove .SAFE extension
    else:
        print("Cannot determine tile ID from products")
        return
    print(tile_id)
    print(products["file_name"].iloc[0].split('.')[0])
    if is_tile_already_processed(index_data, tile_id):
        print(f"Tile {tile_id} already processed, skipping download...")
    else:
        # Download first product
        downloadProduct(products)
        # Extract patches and get results
        extraction_result = extract_patches_to_jpg(SAFE=products["file_name"].iloc[0])
        # Update index
        update_index_with_tile(extraction_result, aoi_polygon)


def reprocess_all_tiles():
    """Reprocess all downloaded .SAFE folders"""
    print("Reprocessing all downloaded tiles...")
    safe_folders = glob.glob('data/infinitesat/*.SAFE')

    if not safe_folders:
        print("No .SAFE folders found to reprocess")
        return

    for safe_folder in safe_folders:
        safe_name = os.path.basename(safe_folder)
        print(f"Reprocessing {safe_name}")
        extraction_result = extract_patches_to_jpg(SAFE=safe_name)
        # Note: Not updating index here since we're reprocessing
        if extraction_result:
            print(f"Reprocessed {extraction_result['patch_count']} patches from {safe_name}")


def datasetStats(path="data/eurosat/", name="Eurosat"):
    """Calculate and display dataset statistics"""
    eurosat_images = []
    eurosat_paths = glob.glob(path + '**/*.jpg', recursive=True)

    if not eurosat_paths:
        print(f"No images found in {path}")
        return

    for im in eurosat_paths:
        img = np.array(Image.open(im))
        eurosat_images.append(img)

    eurosat_stack = np.stack(eurosat_images)

    print(f"{name} pixel value stats:")
    print(f"Min: {eurosat_stack.min()}")
    print(f"Max: {eurosat_stack.max()}")
    print(f"Mean: {eurosat_stack.mean():.2f}")
    print(f"Std: {eurosat_stack.std():.2f}")
    print(f"Percentiles: {np.percentile(eurosat_stack, [1, 5, 25, 50, 75, 95, 99])}")

    for ch, name in enumerate(['Red', 'Green', 'Blue']):
        channel = eurosat_stack[:, :, :, ch]
        print(f"{name} - Mean: {channel.mean():.1f}, Std: {channel.std():.1f}")


def main():
    parser = argparse.ArgumentParser(description='InfiniteSat: Satellite Image Processing Pipeline')
    parser.add_argument('--mode', choices=['all', 'download', 'reprocess', 'rebalance', 'stats'],
                        default='rebalance', help='Processing mode')
    parser.add_argument('--device', default='cuda', help='Device for model inference')

    args = parser.parse_args()

    # Define regions and dates
    dates = [(datetime(2025, 6, 5), datetime(2025, 8, 25))]
    dates.append((datetime(2025, 1, 5), datetime(2025, 3, 25)))
    dates.append((datetime(2024, 9, 5), datetime(2024, 12, 25)))
    dates = []

    central_germany = Polygon([(9.43111261, 50.16247502),
                               (10.83966524, 50.16247502),
                               (10.83966524, 48.43995062),
                               (9.43111261, 48.439950626),
                               (9.43111261, 50.16247502)])
    southern_france = Polygon([(1.0, 43.2), (2.0, 43.2), (2.0, 44.0), (1.0, 44.0), (1.0, 43.2)])
    provenze = Polygon([(4.5, 43.4), (5.5, 43.4), (5.5, 44.2), (4.5, 44.2), (4.5, 43.4)])
    northern_italy = Polygon([(10.5, 44.8), (11.5, 44.8), (11.5, 45.6), (10.5, 45.6), (10.5, 44.8)])

    regions = [central_germany, southern_france, northern_italy]
    trololo = TROLOLO(image_size=64,
                      img_channels=3,
                      patch_size=4,
                      kernel_size=6,
                      num_layers=6,
                      num_heads=48,
                      embed_dim=192,
                      mlp_dim=192,
                      n_class_tokens=2,
                      num_classes=10,
                      mlp_rank=0.05,
                      qkv_rank=0.05,
                      attnproj_rank=0.05,
                      sequence_pyramid=[(2, 4)],
                      attn_rank_pyramid=[(0, 32), (1, 32)],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=1.0,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits= True
                      )
    trololo.load_state_dict(torch.load("trololo.weight",weights_only=False))

    if args.mode == 'all':
        # Download and process all regions
        for d in dates:
            for region in regions:
                process_region(region, d[0], d[1])

        # Create balanced dataset
        print("\nCreating balanced dataset...")
        model = trololo

        if model is not None:
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            dataset =   torchvision.datasets.ImageFolder("/home/pickman/Escritorio/TROLOLO/data/infinitesat/images", transform=transform)
            dataloader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
            create_balanced_dataset(dataloader, model, device=args.device)
        else:
            print("Model not provided - skipping rebalancing")

        # Show statistics
        datasetStats()
        print("")
        datasetStats(path="data/infinitesat/unbalanced/", name="Infinitesat Unbalanced")
        if os.path.exists("data/infinitesat/balanced/"):
            datasetStats(path="data/infinitesat/balanced/", name="Infinitesat Balanced")

    elif args.mode == 'download':
        # Only download and extract
        for d in dates:
            for region in regions:
                process_region(region, d[0], d[1])

    elif args.mode == 'reprocess':
        # Reprocess existing tiles
        reprocess_all_tiles()

    elif args.mode == 'rebalance':
        # Only create balanced dataset
        print("Creating balanced dataset...")
        model = trololo
        if model is not None:
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            dataset =   torchvision.datasets.ImageFolder("/home/pickman/Escritorio/TROLOLO/data/infinitesat/images", transform=transform)
            dataloader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
            create_balanced_dataset(dataloader, model, device=args.device)
        else:
            print("Model not provided - cannot rebalance")

    elif args.mode == 'stats':
        # Only show statistics
        datasetStats()
        print("")
        datasetStats(path="data/infinitesat/unbalanced/", name="Infinitesat Unbalanced")
        if os.path.exists("data/infinitesat/balanced/"):
            datasetStats(path="data/infinitesat/balanced/", name="Infinitesat Balanced")


if __name__ == "__main__":
    main()