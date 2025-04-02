"""
Utility functions for the monkey_genericutils package.
"""

import os
import datetime
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d, gaussian_filter1d

def convert_from_ms(milliseconds):
    """Convert milliseconds to days, hours, minutes, seconds."""
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    seconds = seconds + milliseconds/1000
    return int(days), int(hours), int(minutes), seconds

def convert_hhmmss_to_seconds(time_str):
    """Convert time string in format 'hh:mm:ss.sss' to seconds."""
    hours, minutes, seconds = time_str.split(':')
    total_seconds = (int(hours)*3600) + (int(minutes)*60) + float(seconds)
    return round(total_seconds, 3)

def convert_datetime_format_to_seconds(datetimeobject):
    """Convert datetime object to seconds."""
    seconds = datetime.timedelta(
        hours=datetimeobject.hour,
        minutes=datetimeobject.minute,
        seconds=datetimeobject.second,
        milliseconds=datetimeobject.microsecond/1000
    ).total_seconds()
    return seconds

def pad_img_to_square(img):
    """Pad an image to make it square."""
    h, w, _ = img.shape
    maxside = max(h, w)
    imgsq = np.zeros((maxside, maxside, 3), 'uint8')
    imgsq[:h, :w, :] = img
    return imgsq

def rolling_window(array, window_size, freq, axis=0):
    """Create a rolling window view of an array."""
    shape = array.shape[:axis] + (array.shape[axis] - window_size + 1, window_size) + array.shape[axis+1:]
    strides = array.strides[:axis] + (array.strides[axis],) + array.strides[axis:]
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return np.take(rolled, np.arange(0, shape[axis], freq), axis=axis)

def rolling_median_by_column(array, window_size, output_equal_length=True):
    """Calculate rolling median by column for a 2D array."""
    if len(array.shape) != 2:
        raise AssertionError('array must be a bidimensional array')
    arr_windowed = rolling_window(array, window_size, freq=1, axis=0)
    arr_med = np.median(arr_windowed, axis=1)
    if output_equal_length:
        delta = array.shape[0] - arr_med.shape[0]
        arr_med = np.repeat(arr_med, [1]*(arr_med.shape[0]-1) + [1+delta], axis=0)
    return arr_med

def get_XY_coords_from_roi(rois):
    """Get coordinates from ROI array."""
    if not isinstance(rois, np.ndarray):
        raise TypeError('the roi needs to be a numpy array')

    x1, x2, y1, y2 = ([] for i in range(4))
    for i in range(rois.shape[0]):
        x1.append(rois[i,0]-(rois[i,2]/2))
        y1.append(rois[i,1]-(rois[i,3]/2))
        x2.append(rois[i,0]+(rois[i,2]/2))
        y2.append(rois[i,1]+(rois[i,3]/2))    
    x1 = np.min(x1)
    y1 = np.min(y1)
    x2 = np.max(x2)
    y2 = np.max(y2)
    center = np.mean(rois, axis=0)[0:2]
    return x1, y1, x2, y2, center 