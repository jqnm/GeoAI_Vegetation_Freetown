#Calculation of the different patch metrics

import numpy as np
from scipy.ndimage import label


def calculate_patch_statistics(veg_array, vegetation_classes, pixel_size_m=1):
    """ Berechnet globale Patch-Metriken (Np, PD, MPS, LDI, ED) für Klassen"""
    mask = np.isin(veg_array, vegetation_classes)

    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_patches = label(mask, structure=structure)

    if num_patches == 0:
        return {
            "NP": 0, "PD": 0, "MPS": 0, "LPI": 0, "ED": 0
        }

    total_area_pixels = np.sum(~np.isnan(veg_array))
    pixel_area = pixel_size_m ** 2

    patch_ids, patch_sizes = np.unique(
        labeled_array[labeled_array > 0],
        return_counts=True
    )

    np_val = num_patches
    pd_val = np_val / ((total_area_pixels * pixel_area)/1000)
    mps_val = np.mean(patch_sizes) * pixel_area
    mps_val_km = mps_val/1000
    lpi_val = (np.max(patch_sizes) / total_area_pixels) * 100

    edges = 0
    edges += np.sum(mask[:-1, :] != mask[1:, :])
    edges += np.sum(mask[:, :-1] != mask[:, 1:])

    ed_val = (edges * pixel_size_m) / (total_area_pixels * pixel_area)

    return {
        "NP": np_val,
        "PD": pd_val,
        "MPS": mps_val_km,
        "LPI": lpi_val,
        "ED": ed_val
    }


def class_patch_statistics(veg_array, classes, pixel_size_m, nodata=255):
    """Berechnet globale Klassenstatistiken: NP_k, Area_k (km²), Area_%_k (ohne no_data)"""

    valid_mask = (~np.isnan(veg_array)) & (veg_array != nodata)
    total_area_pixels = np.sum(valid_mask)
    total_area_m2 = total_area_pixels * pixel_size_m ** 2

    structure = np.ones((3, 3), dtype=int)

    results = {}
    total_patches_all_classes = 0

    for cls in classes:
        class_mask = veg_array == cls

        labeled, num_patches = label(class_mask, structure=structure)
        total_patches_all_classes += num_patches

        area_pixels = np.sum(class_mask)
        area_m2 = area_pixels * pixel_size_m ** 2
        area_km2 = area_m2 / 1000

        area_percent = (
            (area_m2 / total_area_m2) * 100
            if total_area_m2 > 0 else 0.0
        )

        results[cls] = {
            "NP_k": int(num_patches),
            "Area_k": float(area_km2),
            "Area_%_k": float(area_percent)
        }

    return results, total_patches_all_classes

import numpy as np


def clumpiness_index(veg_array, target_classes, nodata=255):
    """Clumpiness Index: Messung von Aggregation vs. Dispersion der Zielklasse;
    Wertebereich:
    -1 = maximal dispergiert
     0 = zufällig
    +1 = maximal geklumpt
    """

    valid = (~np.isnan(veg_array)) & (veg_array != nodata)

    mask = np.isin(veg_array, target_classes) & valid

    if np.sum(mask) == 0:
        return np.nan

    same_adj = 0
    total_adj = 0

    #Neighbourhoods (horizontal, vertical, diagonal)
    shifts = [
        (0, 1), (1, 0),
        (1, 1), (1, -1)
    ]

    for dx, dy in shifts:
        m1 = mask[:-dx or None, :-dy or None]
        m2 = mask[dx:, dy:]

        v1 = valid[:-dx or None, :-dy or None]
        v2 = valid[dx:, dy:]

        valid_pair = v1 & v2
        total_adj += np.sum(valid_pair)
        same_adj += np.sum(m1 & m2)

    if total_adj == 0:
        return np.nan

    #Proportion of identical neighbourhoods
    g_obs = same_adj / total_adj

    #Proportion of areas per class
    p = np.sum(mask) / np.sum(valid)

    #expected proportion in random distribution
    g_exp = p * p

    #Clumpiness (normalized)
    if g_obs >= g_exp:
        return (g_obs - g_exp) / (1 - g_exp)
    else:
        return (g_obs - g_exp) / g_exp
