#!/usr/bin/env python3
"""
Vegetation Classification Pipeline

For the detection of different vegetation types (i.e. high vs. low and sparse vs. completely filled) in tiled UAV aerial
imagery of Freetown, Sierra Leone.

Combined script for:
1. Labeling GUI (Context Aware)
2. Model Training (YOLOv8 Classify)
3. Map Stitching
4. Spatial Cross-Validation (with Cluster Visualization)
5. Inference & Grid Generation
6. Sample Inspection
"""
import argparse
import sys
import glob
import time
import random
import shutil
import csv
import re
import pathlib
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# For error handling of loading large stitched maps
Image.MAX_IMAGE_PIXELS = None

try:
    from sklearn.cluster import KMeans
    from sklearn.model_selection import StratifiedKFold
    SKLEARN_AVAIL = True
except ImportError:
    SKLEARN_AVAIL = False
    print("Warning: scikit-learn not found. Spatial CV will fallback to simple grid splitting.")

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    sys.exit(1)

# Configuartions for the vegetation detection pipeline

PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()
DATASETS_ROOT = PROJECT_ROOT / "datasets"
ALL_TILES_DIR = DATASETS_ROOT / "all_tiles"

# Dataset structures for classification
VEG_DATASET_DIR = DATASETS_ROOT / "vegetation_cls"
QUAL_DATASET_DIR = DATASETS_ROOT / "quality_cls"
VEG_POOL_DIR = DATASETS_ROOT / "cls_pool" / "vegetation"
QUAL_POOL_DIR = DATASETS_ROOT / "cls_pool" / "quality"

# Labels
VEG_LABELS = [
    'no_veg', 
    'low_veg_partially_filled', 
    'high_veg_partially_filled', 
    'low_veg_completely_filled', 
    'high_veg_completely_filled'
]

QUAL_LABELS = [
    'good', 
    'bad'
]

# Output paths
MODELS_DIR = PROJECT_ROOT / "models"
VEG_MODEL_PATH = MODELS_DIR / "veg_model.pt"
QUAL_MODEL_PATH = MODELS_DIR / "qual_model.pt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"




# Utilitys

def setup_logger():
    return print

log = setup_logger()

def ensure_folders():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def get_image_files(directory, recursive=False):
    extensions = ["*.tif", "*.jpg", "*.jpeg", "*.png"]
    files = []
    for ext in extensions:
        if recursive:
            files.extend(directory.rglob(ext))
        else:
            files.extend(directory.glob(ext))
    return sorted(list(set(files)))


# 1. GUI LABELING (context-aware)

def run_gui_labeling(args):
    if not ALL_TILES_DIR.exists():
        log(f"Tiles directory not found: {ALL_TILES_DIR}")
        return

    TRACKER_CSV = DATASETS_ROOT / "labeled_tracker.csv"

    if args.reset:
        log("Resetting labeling data...")
        if TRACKER_CSV.exists():
            TRACKER_CSV.unlink()
        if (DATASETS_ROOT / "cls_pool").exists():
            shutil.rmtree(DATASETS_ROOT / "cls_pool")
        
        # Clear the old train/val split dirs if they exist
        if VEG_DATASET_DIR.exists(): shutil.rmtree(VEG_DATASET_DIR)
        if QUAL_DATASET_DIR.exists(): shutil.rmtree(QUAL_DATASET_DIR)
        
        log("Reset complete.")


    TRACKER_CSV = DATASETS_ROOT / "labeled_tracker.csv"
    
    def load_tracker():
        if not TRACKER_CSV.exists():
            return set()
        with open(TRACKER_CSV, "r") as f:
            return set(line.strip() for line in f if line.strip())

    def update_tracker(filename):
        with open(TRACKER_CSV, "a") as f:
            f.write(f"{filename}\n")

    already_labeled = load_tracker()
    all_images = get_image_files(ALL_TILES_DIR)
    
    # Index all images by row, col
    tile_map = {}
    valid_queue = []
    
    for p in all_images:
        match = re.search(r"_r(\d+)_c(\d+)", p.name)
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            tile_map[(r, c)] = p
            # Only add to queue if not labeled
            if p.name not in already_labeled:
                valid_queue.append(p)
    
    random.shuffle(valid_queue)
    queue = valid_queue

    log(f"Total tiles: {len(all_images)}")
    log(f"Already labeled: {len(already_labeled)}")
    log(f"Remaining: {len(queue)}")
    
    if not queue:
        log("All tiles labeled!")
        return

    # Instructions
    print("="*60)
    print("KEYS:")
    print("  [Space]  : Skip Image")
    print("  [Enter]  : Save & Next (Requires both Veg & Qual selected)")
    print("  [Esc]    : Quit")
    print("-" * 20)
    print("Vegetation:")
    for i, lab in enumerate(VEG_LABELS):
        print(f"  [{i+1}] : {lab}")
    print("-" * 20)
    print("Quality:")
    qual_keys = ['q', 'w']
    for k, lab in zip(qual_keys, QUAL_LABELS):
        print(f"  [{k}] : {lab}")
    print("="*60)

    win_name = "Labeling (Context View)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 900, 900)

    idx = 0
    
    # Cache for image size to avoid reading repeatedly (assuming all same size)
    first_img = cv2.imread(str(queue[0])) if queue else None
    base_h, base_w = first_img.shape[:2] if first_img is not None else (100, 100)
    
    while idx < len(queue):
        target_path = queue[idx]
        fname = target_path.name
        
        # Parse target coords
        match = re.search(r"_r(\d+)_c(\d+)", fname)
        if not match:
            idx += 1
            continue
        tr, tc = int(match.group(1)), int(match.group(2))
        
        canvas = np.zeros((base_h * 3, base_w * 3, 3), dtype=np.uint8)
        
        for i in range(-1, 2): 
            for j in range(-1, 2):
                real_r = tr + i 
                real_c = tc + j 
                
                n_path = tile_map.get((real_r, real_c))
                
                y_start = (i + 1) * base_h
                x_start = (j + 1) * base_w
                
                if n_path:
                    tile_img = cv2.imread(str(n_path))
                    if tile_img is not None:
                        # Resize if dimensions slightly mismatch (robustness)
                        if tile_img.shape[:2] != (base_h, base_w):
                             tile_img = cv2.resize(tile_img, (base_w, base_h))

                        mh, mw = base_h, base_w
                        txt = f"r{real_r} c{real_c}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(tile_img, txt, (5, mh-5), font, 0.4, (0,0,0), 3)
                        cv2.putText(tile_img, txt, (5, mh-5), font, 0.4, (255,255,255), 1)
                        
                        canvas[y_start:y_start+base_h, x_start:x_start+base_w] = tile_img
                else:
                    pass

        # Red bounding box around center tile --> file which will labeled
        cv2.rectangle(canvas, 
                      (base_w, base_h), 
                      (base_w*2, base_h*2), 
                      (0, 0, 255), 3) # Red, 3px thickness
        
        img = canvas
            
        selected_veg = None
        selected_qual = None
        
        DISP_W, DISP_H = 900, 900
        
        while True:
            h, w = img.shape[:2]
            scale = min(DISP_W / w, DISP_H / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            status_txt = f"CTX File: {fname} ({idx+1}/{len(queue)})"
            cv2.putText(disp, status_txt, (10, 30), font, 0.7, (0, 255, 0), 2)
            
            v_color = (0, 255, 0) if selected_veg else (0, 0, 255)
            q_color = (0, 255, 0) if selected_qual else (0, 0, 255)
            
            v_txt = f"Veg: {selected_veg if selected_veg else 'Select 1-5'}"
            q_txt = f"Qual: {selected_qual if selected_qual else 'Select q-w'}"
            
            cv2.putText(disp, v_txt, (10, 60), font, 0.7, v_color, 2)
            cv2.putText(disp, q_txt, (10, 90), font, 0.7, q_color, 2)
            
            cv2.imshow(win_name, disp)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27: # Esc
                print("Quitting...")
                cv2.destroyAllWindows()
                return
            
            elif key == 32: # Space -> skip
                print("Skipped.")
                idx += 1
                break
                
            elif key == 13: # Enter -> save
                if selected_veg and selected_qual:
                    # Save to POOL (no split yet)
                    dest_v = VEG_POOL_DIR / selected_veg / fname
                    dest_q = QUAL_POOL_DIR / selected_qual / fname
                    
                    dest_v.parent.mkdir(parents=True, exist_ok=True)
                    dest_q.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        shutil.copy2(target_path, dest_v)
                        shutil.copy2(target_path, dest_q)
                        update_tracker(fname)
                        print(f"Saved to Pool: {selected_veg}, {selected_qual}")
                        idx += 1
                        break
                    except Exception as e:
                        print(f"Save failed: {e}")
                else:
                    print("Selection incomplete!")
            
            # Vegetation keys 1-5
            elif ord('1') <= key <= ord('5'):
                idx_v = key - ord('1')
                if idx_v < len(VEG_LABELS):
                    selected_veg = VEG_LABELS[idx_v]
            
            # Quality keys q, w
            elif key == ord('q'): selected_qual = QUAL_LABELS[0]
            elif key == ord('w'): selected_qual = QUAL_LABELS[1]
            
    cv2.destroyAllWindows()




# 2. TRAINING

def prepare_dataset_from_pool(pool_dir, target_dir, split_ratios=(0.8, 0.2)):
    """
    Splits data from pool_dir into target_dir (train/val) for final training.
    """
    if not pool_dir.exists():
        return False
        
    log(f"Preparing dataset from pool: {pool_dir} -> {target_dir}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
        
    images = get_image_files(pool_dir, recursive=True)
    if not images:
        return False
        
    random.shuffle(images)
    split_idx = int(len(images) * split_ratios[0])
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    for img in train_imgs:
        cls_name = img.parent.name
        dest = target_dir / "train" / cls_name / img.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest)
        
    for img in val_imgs:
        cls_name = img.parent.name
        dest = target_dir / "val" / cls_name / img.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest)
        
    return True

def train_one_model(task_name, dataset_dir, pool_dir, output_model_path, epochs, device):
    log(f"\\n>>> Starting Training for {task_name}...")
    
    # Check if dataset exists, if not try to create from pool
    if not dataset_dir.exists() or not get_image_files(dataset_dir, recursive=True):
        log(f"    Dataset not found at {dataset_dir}, attempting to create from pool...")
        success = prepare_dataset_from_pool(pool_dir, dataset_dir)
        if not success:
            log(f"Error: Pool directory {pool_dir} does not exist or is empty. Run labels GUI first.")
            return

    log(f"    Dataset: {dataset_dir}")
    
    if not dataset_dir.exists():
        log(f"Error: Dataset directory {dataset_dir} does not exist. Run labels GUI first.")
        return

    if not get_image_files(dataset_dir, recursive=True):
        log(f"Error: No images found in {dataset_dir}. Run labels GUI first.")
        return
        
    try:
        # Load pretrained YOLOv8 classification model form ultralytics
        model = YOLO('yolov8n-cls.pt') 
        
        # Train
        results = model.train(
            data=str(dataset_dir), 
            epochs=epochs, 
            imgsz=224, 
            project=str(PROJECT_ROOT / "runs" / "classify"),
            name=task_name,
            exist_ok=True,
            device=device
        )
        
        # Save best modelh
        best_path = pathlib.Path(model.trainer.best)
        log(f"Best model saved at: {best_path}")
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, output_model_path)
        log(f"Copied best model to: {output_model_path}")
        
    except Exception as e:
        log(f"Training failed: {e}")

def train_models(args):
    device = args.device if args.device else None
    
    if device == 'cuda' and not torch.cuda.is_available():
        log("Warning: CUDA requested but not available. Falling back to auto-selection.")
        device = None 
    
    if args.task in ["all", "veg"]:
        train_one_model("vegetation", VEG_DATASET_DIR, VEG_POOL_DIR, VEG_MODEL_PATH, args.epochs, device)
        
    if args.task in ["all", "qual"]:
        train_one_model("quality", QUAL_DATASET_DIR, QUAL_POOL_DIR, QUAL_MODEL_PATH, args.epochs, device)



# 3. STITCHING

def stitch_label_map(full_w, full_h, max_row, min_row):
    veg_grid_path = OUTPUTS_DIR / "veg_grid.npy"
    if not veg_grid_path.exists():
        log("No veg_grid.npy found. Skipping label stitch.")
        return

    try:
        veg_grid = np.load(veg_grid_path)
        grid_img = Image.fromarray(veg_grid.astype(np.uint8))
        grid_flipped = grid_img.transpose(Image.FLIP_TOP_BOTTOM)
        
        label_map = grid_flipped.resize((full_w, full_h), resample=Image.Resampling.NEAREST)
        
        out_path = OUTPUTS_DIR / "stitched_labels.tif"
        label_map.save(out_path, bigtiff=True)
        log(f"Saved stitched label TIFF to {out_path}")
        
    except Exception as e:
        log(f"Error stitching label map: {e}")

def stitch_tiles(args):
    if not ALL_TILES_DIR.exists():
        log(f"Tiles directory not found: {ALL_TILES_DIR}")
        return

    tiles = []
    tile_files = get_image_files(ALL_TILES_DIR)
    if not tile_files:
        log("No tiles.")
        return
        
    for p in tile_files:
        match = re.search(r"_r(\d+)_c(\d+)", p.stem)
        if match:
            tiles.append({
                "path": p,
                "row": int(match.group(1)),
                "col": int(match.group(2))
            })
            
    if not tiles:
        return
        
    df = pd.DataFrame(tiles)
    min_row, max_row = df['row'].min(), df['row'].max()
    min_col, max_col = df['col'].min(), df['col'].max()
    
    with Image.open(tiles[0]['path']) as img:
        orig_w, orig_h = img.size
        
    scale = args.downscale
    tile_w = orig_w // scale
    tile_h = orig_h // scale
    
    n_rows = max_row - min_row + 1
    n_cols = max_col - min_col + 1
    full_w = n_cols * tile_w
    full_h = n_rows * tile_h
    
    # Stitch labels
    stitch_label_map(full_w, full_h, max_row, min_row)
    
    log(f"Stitching {n_rows}x{n_cols} grid into {full_w}x{full_h} image...")
    
    # Increase PIL limit for large mosaics
    Image.MAX_IMAGE_PIXELS = None
    
    canvas = Image.new("RGB", (full_w, full_h), (0, 0, 0))
    
    for _, item in tqdm(df.iterrows(), total=len(df), desc="Stitching"):
        r = int(item['row'])
        c = int(item['col'])
        
        y_idx = max_row - r
        x_idx = c - min_col
        x_pos = x_idx * tile_w
        y_pos = y_idx * tile_h
        
        try:
            with Image.open(item['path']) as img:
                if scale > 1:
                    img = img.resize((tile_w, tile_h), Image.Resampling.BILINEAR)
                canvas.paste(img, (x_pos, y_pos))
        except Exception:
            pass
            
    out_path = OUTPUTS_DIR / f"stitched_map_downscale{scale}.tif"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    canvas.save(out_path, compression="tiff_lzw", bigtiff=True)
    log(f"Saved stitched map to {out_path}")


# 4. SPATIAL CROSS VALIDATION

def run_spatial_cv_workflow(args):
    """ 5-fold spataial cross validation. Worflow:
    # 1. Gather all data from POOL
    # 2. Assign folds (spatially)
    # 3. Train 5 models forvVegetation classifcataion and 5 for quality classification
    # 4. Print results
    # 5. Visualize Clusters
    """
    
    log(f"Starting Spatial CV (5-Fold)...")
    
    # 1. Vegetation CV
    log("\\n--- Vegetation CV ---")
    run_cv_for_task("vegetation", VEG_POOL_DIR, args)
    
    # 2. Quality CV
    log("\\n--- Quality CV ---")
    run_cv_for_task("quality", QUAL_POOL_DIR, args)

    # 3. Visualize Clusters
    log("\\n--- Visualizing CV Clusters ---")
    visualize_cv_clusters(args)

def run_cv_for_task(task, pool_dir, args):
    if not pool_dir.exists():
        log(f"Pool dir {pool_dir} empty.")
        return
        
    images = get_image_files(pool_dir, recursive=True)
    if not images:
        log("No images in pool.")
        return
        
    data = []
    for p in images:
        cls_name = p.parent.name
        # Parse r/c
        match = re.search(r"_r(\d+)_c(\d+)", p.name)
        if match:
             r, c = int(match.group(1)), int(match.group(2))
             data.append({"path": p, "r": r, "c": c, "class": cls_name})
             
    if not data:
        log("No parseable images.")
        return
        
    df = pd.DataFrame(data)
    
    # Assign Folds
    if SKLEARN_AVAIL:
        # Spatial k-means clustering
        coords = df[['r', 'c']].values
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df['fold'] = kmeans.fit_predict(coords)
    else:
        log("Sklearn missing, using random split (NOT SPATIAL).")
        indices = np.arange(len(df))
        np.random.shuffle(indices)
        df['fold'] = indices % 5
        
    metrics = []
    
    for fold_i in range(5):
        log(f"\\n> Fold {fold_i+1}/5")
        
        fold_dir = DATASETS_ROOT / f"cv_{task}_fold{fold_i}"
        if fold_dir.exists(): shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True)
        
        # Train / val splits
        train_df = df[df['fold'] != fold_i]
        val_df = df[df['fold'] == fold_i]
        
        for _, row in train_df.iterrows():
            dest = fold_dir / "train" / row['class'] / row['path'].name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row['path'], dest)
            
        for _, row in val_df.iterrows():
            dest = fold_dir / "val" / row['class'] / row['path'].name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row['path'], dest)
            
        # Train
        run_name = f"{task}_cv_fold{fold_i}"
        
        try:
            model = YOLO('yolov8n-cls.pt')
            device = args.device if args.device else None
            if device == 'cuda' and not torch.cuda.is_available(): device=None
            
            model.train(
                data=str(fold_dir), 
                epochs=args.epochs, 
                imgsz=224, 
                project=str(PROJECT_ROOT / "runs" / "cv"),
                name=run_name,
                exist_ok=True,
                device=device
            )
            
            # Metrics
            best = YOLO(str(model.trainer.best))
            res_val = best.val(split='val') 
            acc = res_val.results_dict['metrics/accuracy_top1']
            log(f"  Fold {fold_i+1} Accuracy: {acc:.4f}")
            metrics.append(acc)
            
        except Exception as e:
            log(f"Fold {fold_i} failed: {e}")
            
    # Report
    if metrics:
        avg = sum(metrics) / len(metrics)
        log(f"\\n=== {task.upper()} CV RESULTS ===")
        log(f"Accuracies: {[round(x, 4) for x in metrics]}")
        log(f"Mean Accuracy: {avg:.4f}")

def visualize_cv_clusters(args):
    """
    Visualizes the spatial distribution of labeled training data and the resulting
    Spatial Cross-Validation folds (clusters). 
    """
    log("Starting CV Cluster Visualization...")
    
    # Gather labeled data
    labeled_files = set()
    
    if VEG_POOL_DIR.exists():
        labeled_files.update(VEG_POOL_DIR.rglob("*.jpg"))
    if QUAL_POOL_DIR.exists():
        labeled_files.update(QUAL_POOL_DIR.rglob("*.jpg"))
        
    if not labeled_files:
        log("No labeled data found in datasets/cls_pool.")
        return

    log(f"Found {len(labeled_files)} labeled tile instances (counting duplicates across tasks).")
    
    data = []
    seen_coords = set()
    
    for p in labeled_files:
        match = re.search(r"_r(\d+)_c(\d+)", p.name)
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            if (r, c) not in seen_coords:
                data.append({"r": r, "c": c})
                seen_coords.add((r, c))
                
    if not data:
        log("Could not parse coordinates from filenames.")
        return
        
    df = pd.DataFrame(data)
    log(f"Unique labeled locations: {len(df)}")

    all_tiles = list(ALL_TILES_DIR.glob("*.jpg"))
    if not all_tiles:
        log("No tiles found in datasets/all_tiles")
        return
        
    all_rows = []
    all_cols = []
    for p in all_tiles:
        match = re.search(r"_r(\d+)_c(\d+)", p.name)
        if match:
            all_rows.append(int(match.group(1)))
            all_cols.append(int(match.group(2)))
            
    min_row, max_row = min(all_rows), max(all_rows)
    min_col, max_col = min(all_cols), max(all_cols)
    
    n_rows = max_row - min_row + 1
    n_cols = max_col - min_col + 1
    
    log(f"Grid Dimensions: {n_rows} rows x {n_cols} cols")
    log(f"Bounds: R[{min_row}-{max_row}], C[{min_col}-{max_col}]")

    if SKLEARN_AVAIL:
        coords = df[['r', 'c']].values
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df['fold'] = kmeans.fit_predict(coords)
    else:
        indices = np.arange(len(df))
        np.random.shuffle(indices)
        df['fold'] = indices % 5


    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Background: stitched map
    # Attempt to find the stitched map corresponding to args.downscale
    stitched_map_path = OUTPUTS_DIR / f"stitched_map_downscale{args.downscale}.tif"
    
    if not stitched_map_path.exists():
         fallback = OUTPUTS_DIR / "stitched_map_downscale8.tif"
         if fallback.exists():
             stitched_map_path = fallback

    use_map = False
    if stitched_map_path.exists():
        try:
            log(f"Loading stitched map: {stitched_map_path}")
            
            img = Image.open(stitched_map_path)
            img_w, img_h = img.size
            
            est_tile_w = img_w / n_cols
            est_tile_h = img_h / n_rows
            
            # Display background image - north up
            ax.imshow(img, extent=[0, img_w, 0, img_h], origin='lower') 
            
            plot_x = (df['c'] - min_col) * est_tile_w + (est_tile_w / 2)
            plot_y = (max_row - df['r']) * est_tile_h + (est_tile_h / 2)
            
            use_map = True
        except Exception as e:
            log(f"Failed to load map: {e}")
            use_map = False
            
    if not use_map:
        log("Using scatter plot background.")
        ax.scatter(all_cols, all_rows, c='lightgrey', s=5, marker='s', label='Unlabeled')
        ax.invert_yaxis()
        plot_x = df['c']
        plot_y = df['r']

    scatter = ax.scatter(plot_x, plot_y, c=df['fold'], cmap='tab10', s=30, edgecolors='white', linewidth=0.5, zorder=10)
    
    handles, _ = scatter.legend_elements()
    legend_labels = [f"Fold {i}" for i in range(5)]
    ax.legend(handles, legend_labels, title="CV Folds", loc='upper right')
    
    title = f"Spatial CV Clusters (k=5)\n{len(df)} Labeled Tiles"
    if use_map:
        title += " (Overlaid on Stitched Map)"
    ax.set_title(title)
    
    if not use_map:
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
    else:
        ax.axis('off')

    out_path = OUTPUTS_DIR / "cv_clusters_viz.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    log(f"Saved visualization to: {out_path}")
    plt.close()


# 3. INFERENCE

def run_inference(args):
    ensure_folders()
    
    if not VEG_MODEL_PATH.exists() and not QUAL_MODEL_PATH.exists():
        log("No models found. Please train first.")
        return

    if not ALL_TILES_DIR.exists():
        log("No tiles to infer.")
        return
        
    tile_files = get_image_files(ALL_TILES_DIR)
    if not tile_files:
        log("No image files found in tiles dir.")
        return

    log(f"Found {len(tile_files)} tiles. Loading models...")

    veg_model = None
    qual_model = None
    
    if VEG_MODEL_PATH.exists():
        veg_model = YOLO(str(VEG_MODEL_PATH))
    else:
        log("Warning: Vegetation model missing, skipping veg predictions.")
        
    if QUAL_MODEL_PATH.exists():
        qual_model = YOLO(str(QUAL_MODEL_PATH))
    else:
        log("Warning: Quality model missing, skipping quality predictions.")

    veg_preds_map = {}
    qual_preds_map = {}
    
    # 1. Vegetation
    if veg_model:
        log("Inferring Vegetation...")
        for r in tqdm(veg_model.predict(source=str(ALL_TILES_DIR), stream=True, verbose=False), total=len(tile_files)):
            path = pathlib.Path(r.path)
            top1_idx = r.probs.top1
            class_name = r.names[top1_idx]
            veg_preds_map[path.name] = class_name
            
    # 2. Quality
    if qual_model:
        log("Inferring Quality...")
        for r in tqdm(qual_model.predict(source=str(ALL_TILES_DIR), stream=True, verbose=False), total=len(tile_files)):
            path = pathlib.Path(r.path)
            top1_idx = r.probs.top1
            class_name = r.names[top1_idx]
            qual_preds_map[path.name] = class_name
            
    # Combine
    data = []
    for p in tile_files:
        fname = p.name
        
        match = re.search(r"_r(\d+)_c(\d+)", fname)
        if not match:
            continue
        row = int(match.group(1))
        col = int(match.group(2))
        
        v_pred = veg_preds_map.get(fname, "N/A")
        q_pred = qual_preds_map.get(fname, "N/A")
        
        data.append({
            "row": row,
            "col": col,
            "veg_pred": v_pred,
            "qual_pred": q_pred
        })
        
    df_res = pd.DataFrame(data)
    if df_res.empty:
        log("No results parsed.")
        return
        
    save_outputs(df_res)

def save_outputs(df):
    min_row, max_row = df['row'].min(), df['row'].max()
    min_col, max_col = df['col'].min(), df['col'].max()
    n_rows = max_row - min_row + 1
    n_cols = max_col - min_col + 1
    
    v_map = {name: i for i, name in enumerate(VEG_LABELS)}
    q_map = {name: i for i, name in enumerate(QUAL_LABELS)}
    
    veg_grid = np.full((n_rows, n_cols), -1, dtype=int)
    qual_grid = np.full((n_rows, n_cols), -1, dtype=int)
    
    for _, item in df.iterrows():
        r = item['row'] - min_row
        c = item['col'] - min_col
        
        v_lbl = item['veg_pred']
        q_lbl = item['qual_pred']
        
        if v_lbl in v_map:
            veg_grid[r, c] = v_map[v_lbl]
        if q_lbl in q_map:
            qual_grid[r, c] = q_map[q_lbl]
            
    np.save(OUTPUTS_DIR / "veg_grid.npy", veg_grid)
    np.save(OUTPUTS_DIR / "qual_grid.npy", qual_grid)
    
    try:
        Image.fromarray(veg_grid.astype(np.uint8)).save(OUTPUTS_DIR / "veg_grid.tif")
        Image.fromarray(qual_grid.astype(np.uint8)).save(OUTPUTS_DIR / "qual_grid.tif")
    except Exception as e:
        log(f"Error saving grid TIFFs: {e}")

    plot_grid(veg_grid, VEG_LABELS, "Vegetation Map", OUTPUTS_DIR / "veg_grid_map.png")
    plot_grid(qual_grid, QUAL_LABELS, "Quality Map", OUTPUTS_DIR / "qual_grid_map.png")
    
    log(f"Saved outputs to {OUTPUTS_DIR}")

def plot_grid(grid, class_names, title, save_path):
    masked_grid = np.ma.masked_where(grid == -1, grid)
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab10", len(class_names))
    img = plt.imshow(masked_grid, cmap=cmap, origin="lower", interpolation="nearest", 
                     vmin=-0.5, vmax=len(class_names)-0.5)
    plt.title(title)
    cbar = plt.colorbar(img, ticks=range(len(class_names)))
    cbar.ax.set_yticklabels(class_names)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




# 5. INSPECTION

def plot_classified_samples(args):
    """
    Generates a plot showing N random samples for each class based on the
    inference grids (veg_grid.npy, qual_grid.npy).
    """
    ensure_folders()
    
    veg_grid_path = OUTPUTS_DIR / "veg_grid.npy"
    qual_grid_path = OUTPUTS_DIR / "qual_grid.npy"
    
    if not veg_grid_path.exists() and not qual_grid_path.exists():
        log("No inference grids found. Run --stage infer first.")
        return

    if not ALL_TILES_DIR.exists():
        log("Tiles directory not found.")
        return

    tile_files = get_image_files(ALL_TILES_DIR)
    if not tile_files:
        log("No tiles found.")
        return
        
    tiles_map = {} 
    rows = []
    cols = []
    
    for p in tile_files:
        match = re.search(r"_r(\d+)_c(\d+)", p.name)
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            tiles_map[(r, c)] = p
            rows.append(r)
            cols.append(c)
            
    if not rows:
        return
        
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    def get_tile_path(grid_r, grid_c):
        real_r = grid_r + min_row
        real_c = grid_c + min_col
        return tiles_map.get((real_r, real_c))

    samples_per_class = args.samples
    tasks_to_plot = []
    
    if veg_grid_path.exists():
        veg_grid = np.load(veg_grid_path)
        tasks_to_plot.append({
            "name": "Vegetation",
            "grid": veg_grid,
            "labels": VEG_LABELS
        })
        
    if qual_grid_path.exists():
        qual_grid = np.load(qual_grid_path)
        tasks_to_plot.append({
            "name": "Quality",
            "grid": qual_grid,
            "labels": QUAL_LABELS
        })
        
    for task in tasks_to_plot:
        task_name = task['name']
        grid = task['grid']
        labels = task['labels']
        
        log(f"Generating samples for {task_name}...")
        
        n_rows = len(labels)
        n_cols = samples_per_class
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.5))
        if n_rows == 1 and n_cols == 1:
             axes = np.array([[axes]])
        elif n_rows == 1:
             axes = np.expand_dims(axes, 0)
        elif n_cols == 1:
             axes = np.expand_dims(axes, 1)
             
        # fig.suptitle(f"{task_name} Samples - {samples_per_class} per class", fontsize=16)
        
        for i, label_name in enumerate(labels):
            matches = np.argwhere(grid == i) 
            
            selected = []
            
            # Helper for balanced selection
            def balanced_sample_selection(candidates, limit_black=True):
                if len(candidates) == 0: return []
                
                if not limit_black:
                    if len(candidates) > samples_per_class:
                        idx = np.random.choice(len(candidates), samples_per_class, replace=False)
                        return candidates[idx]
                    return candidates

                # Logic: min 1 all black tile (if exists), max 1 all black tile (due to overrepresentation in the bad and no_veg class)
                np.random.shuffle(candidates)
                non_black_samples = []
                black_samples = []
                
                check_limit = 1000 
                checks = 0
                
                for m in candidates:
                    if len(black_samples) >= 1 and len(non_black_samples) >= (samples_per_class - 1):
                         break
                    if checks >= check_limit:
                        break
                        
                    gr, gc = m
                    t_path = get_tile_path(gr, gc)
                    if not t_path: continue
                    
                    try:
                        checks += 1
                        with Image.open(t_path) as img:
                            if np.sum(np.array(img)) == 0:
                                if len(black_samples) < 1:
                                    black_samples.append(m)
                            else:
                                if len(non_black_samples) < samples_per_class:
                                    non_black_samples.append(m)
                    except Exception:
                        pass
                
                final_sel = []
                if len(black_samples) > 0:
                    final_sel.append(black_samples[0])
                
                slots_left = samples_per_class - len(final_sel)
                if slots_left > 0:
                     final_sel.extend(non_black_samples[:slots_left])
                     
                return np.array(final_sel)

            if task_name == "Vegetation":
                selected = balanced_sample_selection(matches, limit_black=True)
            elif task_name == "Quality" and label_name == "bad":
                selected = balanced_sample_selection(matches, limit_black=True)
            else:
                selected = balanced_sample_selection(matches, limit_black=False)
                
            for j in range(n_cols):
                ax = axes[i, j]
                ax.axis('off')
                
                if j == 0:
                    ax.set_title(label_name, loc='left', fontsize=24, fontweight='bold', rotation=0, x=-0.1, y=0.5, ha='right', va='center')
                
                if j < len(selected):
                    gr, gc = selected[j]
                    t_path = get_tile_path(gr, gc)
                    if t_path:
                        try:
                            img = Image.open(t_path)
                            ax.imshow(img)
                        except:
                            ax.text(0.5, 0.5, "Err", ha='center')
                    else:
                        ax.text(0.5, 0.5, "Miss", ha='center')
                else:
                    if j == 0 and len(selected) == 0:
                         ax.text(0.5, 0.5, "No Preds", ha='center', fontsize=8)
                         
        plt.tight_layout(rect=[0.2, 0.03, 1, 0.95]) 
        
        out_file = OUTPUTS_DIR / f"samples_{task_name.lower()}.png"
        plt.savefig(out_file, dpi=150)
        plt.close()
        log(f"Saved {task_name} samples to {out_file}")

    # Filtered plot: vegetation (good quality only)
    if veg_grid_path.exists() and qual_grid_path.exists():
        veg_grid = np.load(veg_grid_path)
        qual_grid = np.load(qual_grid_path)
        
        try:
            good_idx = QUAL_LABELS.index('good')
        except ValueError:
            log("Label 'good' not found in QUAL_LABELS. Skipping filtered plot.")
            return

        log("Generating samples for vegetation (good quality only)...")
        
        n_rows = len(VEG_LABELS)
        n_cols = samples_per_class
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.5))
        if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
        elif n_rows == 1: axes = np.expand_dims(axes, 0)
        elif n_cols == 1: axes = np.expand_dims(axes, 1)
             
        # fig.suptitle(f"Vegetation Samples (Good Quality Only) - {samples_per_class} per class", fontsize=16)
        
        for i, label_name in enumerate(VEG_LABELS):
            matches = np.argwhere((veg_grid == i) & (qual_grid == good_idx))
            
            selected = []
            if len(matches) > 0:
                 
                 if len(matches) > samples_per_class:
                    indices = np.random.choice(len(matches), samples_per_class, replace=False)
                    selected = matches[indices]
                 else:
                    selected = matches
            
            for j in range(n_cols):
                ax = axes[i, j]
                ax.axis('off')
                
                if j == 0:
                    ax.set_title(label_name, loc='left', fontsize=24, fontweight='bold', rotation=0, x=-0.1, y=0.5, ha='right', va='center')
                
                if j < len(selected):
                    gr, gc = selected[j]
                    t_path = get_tile_path(gr, gc)
                    if t_path:
                        try:
                            img = Image.open(t_path)
                            ax.imshow(img)
                        except:
                            ax.text(0.5, 0.5, "Err", ha='center')
                    else:
                        ax.text(0.5, 0.5, "Miss", ha='center')
                else:
                    if j == 0 and len(selected) == 0:
                         ax.text(0.5, 0.5, "No Matches", ha='center', fontsize=8)
                         
        plt.tight_layout(rect=[0.2, 0.03, 1, 0.95])
        
        out_file = OUTPUTS_DIR / "samples_vegetation_good_only.png"
        plt.savefig(out_file, dpi=150)
        plt.close()
        log(f"Saved Filtered Vegetation samples to {out_file}")










# MAIN

def main():
    parser = argparse.ArgumentParser(description="Vegetation Classification Pipeline")
    parser.add_argument("--stage", type=str, required=True, 
                        choices=["gui", "train", "infer", "stitch", "cv", "inspect", "all"],
                        help="Stage to run")
    parser.add_argument("--task", type=str, default="all", choices=["all", "veg", "qual"],
                        help="Which task to train (default: all)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (not used in all modes)")
    parser.add_argument("--device", type=str, default="", help="cuda/cpu/mps (empty for auto)")
    parser.add_argument("--downscale", type=int, default=4, help="Stitch downscale")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per class for inspect stage")
    parser.add_argument("--reset", action="store_true", help="Reset all labeling data")
    
    args = parser.parse_args()
    
    if args.stage == "gui":
        run_gui_labeling(args)
    elif args.stage == "train":
        train_models(args)
    elif args.stage == "infer":
        run_inference(args)
    elif args.stage == "cv":
        run_spatial_cv_workflow(args)
    elif args.stage == "stitch":
        stitch_tiles(args)
    elif args.stage == "inspect":
        plot_classified_samples(args)
    elif args.stage == "all":
        train_models(args)
        run_inference(args)
        stitch_tiles(args)

if __name__ == "__main__":
    main()
