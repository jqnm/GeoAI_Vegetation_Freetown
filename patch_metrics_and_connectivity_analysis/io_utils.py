#Basic outsourced functions

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

def reproject_to_metric(src_path, dst_path, dst_crs="EPSG:32628"):
    '''Rreprojects raster if needed'''
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        meta = src.meta.copy()
        meta.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(dst_path, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )

def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile, src.nodata, src.crs

def write_tif(out_path, data, profile):
    meta = profile.copy()
    meta.update(dtype="float32", count=1)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)
