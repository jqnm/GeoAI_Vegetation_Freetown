#Main script for the execution of patch metrics and connectivity maps

from pathlib import Path
import numpy as np
import rasterio

#utils-import
from io_utils import reproject_to_metric, load_raster, write_tif

#connectivty map functions
from join_count_class_connectivity import (
    class_specific_connectivity,
    vegetation_connectivity,
    vegetation_full_connectivity
)
#metrics-import
from patch_metrics import (
    calculate_patch_statistics,
    class_patch_statistics,
    clumpiness_index
)

# Parameter
INPUT_TIF = r"C:...GeoAIsem\code\veg_grid_ref_true.tif" # Input
WORK_TIF = "vegetation_metric.tif"

WINDOW_SIZE = 10
CLASSES = [0, 1, 2, 3, 4]
VEG_CLASSES = [1, 2, 3, 4]
VEG_FULL = [3, 4]
CLASS_1 = [1]

BASE_DIR = Path(__file__).parent
TIF_DIR = BASE_DIR / "tif"
TIF_DIR.mkdir(exist_ok=True)

# Reproject
with rasterio.open(INPUT_TIF) as src:
    if src.crs is None:
        raster_path = INPUT_TIF
    elif not src.crs.is_projected:
        reproject_to_metric(INPUT_TIF, WORK_TIF)
        raster_path = WORK_TIF
    else:
        raster_path = INPUT_TIF

veg, profile, nodata, _ = load_raster(raster_path)

if nodata is None:
    nodata = 255

veg = veg.astype("float32")
veg[veg == nodata] = np.nan

profile = profile.copy()
profile.update({
    "height": veg.shape[0],
    "width": veg.shape[1],
    "dtype": "float32",
    "nodata": nodata
})

# connectivity functions
class_conn = class_specific_connectivity(veg, nodata, WINDOW_SIZE, CLASSES)

veg_full_conn = vegetation_full_connectivity(veg, nodata, WINDOW_SIZE, VEG_FULL)

veg_conn = vegetation_connectivity(veg, nodata, WINDOW_SIZE, VEG_CLASSES)

# NaN to NoData 
for cls in class_conn:
    class_conn[cls][np.isnan(class_conn[cls])] = nodata

veg_full_conn[np.isnan(veg_full_conn)] = nodata
veg_conn[np.isnan(veg_conn)] = nodata

# Writing Tiffs
for cls, data in class_conn.items():
    write_tif(TIF_DIR / f"connectivity_class_{cls}.tif", data, profile)

write_tif(TIF_DIR / "connectivity_full_vegetation.tif", veg_full_conn, profile)
write_tif(TIF_DIR / "connectivity_vegetation.tif", veg_conn, profile)


# Overview patches
pixel_size = profile["transform"][0]
print("\n Class patches overview")

class_stats, total_np = class_patch_statistics(
    veg_array=veg,
    classes=CLASSES,
    pixel_size_m=pixel_size,
    nodata=nodata
)

for cls, s in class_stats.items():
    print(f"\nKlasse {cls}:")
    print(f" - NP_k:     {s['NP_k']}")
    print(f" - Area_k:   {s['Area_k']:.2f} km²")
    print(f" - Area_%_k: {s['Area_%_k']:.2f} %")


print(f"\n Summierte Patch-Anzahl (all classes): {total_np}")

# Patch-metrics

stats_no = calculate_patch_statistics(veg, CLASS_1, pixel_size)
stats_veg = calculate_patch_statistics(veg, VEG_CLASSES, pixel_size)
stats_full_veg = calculate_patch_statistics(veg, VEG_FULL, pixel_size)

def print_stats(name, s):
    print(f"\n{name}:")
    '''print(f" - NP:   {s['NP']}")'''
    print(f" - PD:   {s['PD']:.6f} n/km²")
    print(f" - MPS:  {s['MPS']:.2f} km²")
    print(f" - LPI:  {s['LPI']:.2f} %")
    print(f" - ED:   {s['ED']:.4f} m/m²")

print_stats("No Vegetation", stats_no)
print_stats("Combined Vegetation", stats_veg)
print_stats("Complete Vegetation", stats_full_veg)

#  Clumpines_metric
clump_no_veg = clumpiness_index(veg, CLASS_1, nodata)
clump_veg = clumpiness_index(veg, VEG_CLASSES, nodata)
clump_forest = clumpiness_index(veg, VEG_FULL, nodata)

print("\n Clumpiness Index")

print(f"Clumpiness (No Vegetation):          {clump_no_veg:.3f}")
print(f"Clumpiness (Combined Vegetation):        {clump_veg:.3f}")
print(f"Clumpiness (Complete Vegetation):                {clump_forest:.3f}")
