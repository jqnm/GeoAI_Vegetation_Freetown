#Calculation of class connectivity

import numpy as np
from scipy.ndimage import generic_filter

def class_specific_connectivity(veg, nodata, window_size, classes):
    '''berechnet die Connectifty für class=0'''
    MAX_JOINS = 2 * window_size * (window_size - 1)
    results = {}

    for cls in classes:

        def _func(window, target=cls):
            center = window[len(window) // 2]

            #real and explicit no_data
            if np.isnan(center):
                return np.nan
            if center == nodata:
                return np.nan

            #Centre is not target class
            if center != target:
                return 0.0

            w = window.reshape((window_size, window_size))
            count = 0

            for i in range(window_size):
                for j in range(window_size):
                    if np.isnan(w[i, j]):
                        continue
                    if w[i, j] != target:
                        continue

                    if i + 1 < window_size and w[i + 1, j] == target:
                        count += 1
                    if j + 1 < window_size and w[i, j + 1] == target:
                        count += 1

            return count / MAX_JOINS

        results[cls] = generic_filter(
            veg,
            _func,
            size=window_size,
            mode="nearest"
        )

    return results

def vegetation_full_connectivity(veg, nodata, window_size, vegetation_classes):
    '''berechnet connectifty für classes 3, 4'''
    veg_classes = set(vegetation_classes)

    def _func(window):
        center = window[len(window) // 2]

        if np.isnan(center):
            return np.nan

        if center == nodata:
            return np.nan

        if center not in veg_classes:
            return 0.0

        w = window.reshape((window_size, window_size))
        count = 0

        for i in range(window_size):
            for j in range(window_size):
                if np.isnan(w[i, j]):
                    continue
                if w[i, j] not in veg_classes:
                    continue

                if i + 1 < window_size and w[i + 1, j] in veg_classes:
                    count += 1
                if j + 1 < window_size and w[i, j + 1] in veg_classes:
                    count += 1

        return float(count)

    return generic_filter(
        veg,
        _func,
        size=window_size,
        mode="nearest"
    )



def vegetation_connectivity(veg, nodata, window_size, vegetation_classes):
    '''berechnet connectivity für classes=1, 2, 3, 4'''
    veg_classes = set(vegetation_classes)

    def _func(window):
        center = window[len(window) // 2]

        if np.isnan(center):
            return np.nan

        if center == nodata:
            return np.nan

        if center not in veg_classes:
            return 0.0

        w = window.reshape((window_size, window_size))
        count = 0
        for i in range(window_size):
            for j in range(window_size):
                if np.isnan(w[i, j]):
                    continue
                if w[i, j] not in veg_classes:
                    continue
                if i + 1 < window_size and w[i + 1, j] in veg_classes:
                    count += 1
                if j + 1 < window_size and w[i, j + 1] in veg_classes:
                    count += 1

        return float(count)

    return generic_filter(
        veg,
        _func,
        size=window_size,
        mode="nearest"
    )



