# Vegetation Type Classification Pipeline

This repository contains the `vegetation_classification.py` pipeline for classifying vegetation and quality from tiled UAV aerial imagery of Freetown, Sierra Leone, using pretrained YOLOv8 model architectur from Ultralytics. Based on the results of this vegetation classification subsequently some patch metrics as well as a connectivity analysis can be determined/performed. 

![Workflow of the vegetation type classification pipeline ](workflow.jpg)


## Overview

The **vegetation classification pipeline** handles follwing entire workflow:
1.  **Labeling**: Interactive GUI for manual context-aware labeling.
2.  **Training**: Training YOLOv8 classification models.
3.  **Stitching**: Recombining tiles into a large mosaic map (including a label map).
4.  **Cross-Validation**: Spatial K-Fold cross-validation to assess model performance and geographic bias.
5.  **Inference**: Running predictions on full datasets.
6.  **Inspection**: Visualizing sample predictions for quality control.


The **patch metrics and connectivity analysis** module handles:
1.  **Raster preprocessing**: Reprojection and I/O utilities.
2.  **Patch-based landscape metrics**: Quantitative assessment of fragmentation.
3.  **Local connectivity analysis**: Join-count class connectivity measurements.
4.  **Main execution**: Integration and output generation for landscape structure.



## Requirements

- Python 3.8+
- `ultralytics` (YOLOv8)
- `opencv-python`
- `pillow`
- `numpy`, `scipy`, `pandas`, `matplotlib`, `pathlib`
- `scikit-learn`
- `rasterio`

Install dependencies:
```bash
pip install ultralytics opencv-python pillow numpy pandas matplotlib scikit-learn tqdm rasterio scipy
```


## Usage – Vegetation Classification

The vegetation pipeline script is controlled via terminal commands using the `--stage` argument.

### 1. Labeling (GUI)
Open the interactive labeling window. You will see a 3x3 context grid around the center tile, which is to be labeled.
```bash
python vegetation_classification.py --stage gui
```
**Controls:**
- `1-5`: Select Vegetation Class
- `q/w`: Select Quality Class (`good`/`bad`)
- `Enter`: Save and Next
- `Space`: Skip
- `Esc`: Quit

To **reset** all labels (start fresh):
```bash
python vegetation_classification.py --stage gui --reset
```

### 2. Stitching
Stilches the original tiles and the predicted labels into large mosaic TIFFs.
```bash
python vegetation_classification.py --stage stitch --downscale 4
```
*Note:* 
- For adding the predicted labels the inference fisrt has to be run.
- The output maps are saved as BigTIFF to support large file sizes.

### 3. Spatial Cross-Validation (CV)
Run 5-Fold Spatial Cross-Validation. This clusters tiles geographically to prevent data leakage between train/val sets.
It creates a visualization of the clusters at the end.
```bash
python vegetation_classification.py --stage cv --epochs 10
```

### 3. Training
Train the final models (`veg_model.pt` and `qual_model.pt`) on all available labeled data.
```bash
python vegetation_classification.py --stage train --epochs 20 --task all
```
*Options:* `--task veg` or `--task qual` to train only one model.

### 4. Inference
Run inference on all tiles in the `datasets/all_tiles` directory using the trained models. Generates grid files (`.npy`) and maps (`.tif`).
```bash
python vegetation_classification.py --stage infer
```

### 6. Inspection
Generate summary plots of random samples for each class to visually verify model performance.
```bash
python vegetation_classification.py --stage inspect --samples 10
```

### Run All
Execute Training -> Inference -> Stitching in sequence.
```bash
python vegetation_classification.py --stage all
```



## Usage – Patch Metrics and Connectivity Analysis
This module extends the pipeline by computing landscape metrics from the stitched classification maps. The scripts are located in the patch_metrics_and_connectivity_analysis/ subfolder.

### 1. Raster Preprocessing and I/O (`io_utils.py`)
This module provides basic raster handling utilities used throughout the analysis:
- Reprojection of rasters to a metric CRS to ensure valid area and distance calculations
- Loading raster data including array, metadata, NoData value, and CRS
- Writing output GeoTIFFs with consistent spatial metadata
All subsequent patch and connectivity calculations rely on these utilities to ensure spatial consistency and reproducibility.

### 2. Patch Metrics (`patch_metrics.py`)
Patch metrics describe the **spatial configuration and fragmentation** of selected land-cover classes using connected-component labeling (8-neighbourhood).
Results are printed to the console for quick inspection.
Computed metrics include:
- Number of Patches (NP)
- Patch Density (PD)
- Mean Patch Size (MPS)
- Largest Patch Index (LPI)
- Edge Density (ED)
- Clumpiness Index, measuring aggregation versus dispersion (range −1 to +1)
Metrics can be computed for individual classes, combined vegetation classes, or selected vegetation subsets.

### 3. Connectivity Analysis (`join_count_class_connectivity.py`)
Local connectivity is quantified using a **join-count approach** within a moving window. For each valid pixel, adjacent same-class joins (horizontal and vertical) are counted.
Implemented variants include:
- Class-specific connectivity (e.g. no-vegetation class)
- Vegetation connectivity across multiple vegetation classes
- Full vegetation connectivity for selected core vegetation classes
Pixels outside the target classes, NoData values, and NaNs are excluded from the calculations.
Connectivity results are written to GeoTIFF files for further spatial analysis and visualization.

### 4. Main Execution (`fragmentation.py`)
The main script integrates all components and executes the full analysis workflow:
1. Loads and reprojects the stitched vegetation raster
2. Computes connectivity maps for all defined class groups
3. Writes connectivity outputs as GeoTIFFs
4. Computes class-level area statistics and global patch metrics
5. Calculates clumpiness indices for selected vegetation configurations
The resulting outputs provide a **quantitative characterization of landscape structure, fragmentation, and spatial connectivity**, complementing the vegetation type classification pipeline.



## File Structure
- `vegetation_classification.py`: Main entry point for classification.
- `patch_metrics_and_connectivity_analysis/`:
    - `io_utils.py`: Raster I/O utilities.
    - `patch_metrics.py`: Calculation of landscape metrics.
    - `join_count_class_connectivity.py`: Connectivity algorithms.
    - `fragmentation.py`: Analysis integration script.
- `datasets/all_tiles`: Input image tiles (must be named `..._r{row}_c{col}.ext`)
- `datasets/cls_pool`: Labeled data sorted by class.
- `models/`: Saved model weights.
- `outputs/`: Inference grids, maps, stitched mosaics, and inspection plots.



