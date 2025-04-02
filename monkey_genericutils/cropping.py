"""
Functions for cropping videos based on detected macaques.
"""

import os
import cv2
import numpy as np
import progressbar
from vidgear.gears.stabilizer import Stabilizer
from vidgear.gears import VideoGear

from .utils import pad_img_to_square, convert_from_ms
from .detection import get_crop_coord_list, fix_detected_filenames_stride

def crop_frame(frame, coordsvec):
    """
    Crop a frame based on coordinates.
    
    THIS FUNCTION ONLY APPLIES TO TALL FRAMES SUCH AS THE VERTICAL ONES RECORDED IN THE WIRELESS PROJECT 2022
    
    Parameters
    ----------
    frame : numpy.ndarray
        Input frame
    coordsvec : numpy.ndarray
        x1,y1,x2,y2,center[0](X),center[1](Y) of the single metabox containing detected boxes, in relative coords [0,1]

    Returns
    -------
    numpy.ndarray
        Cropped frame
    """
    pad = 0
    H = frame.shape[0]
    W = frame.shape[1]

    # Coordinates of the center (now in real pixels)
    fx1 = int(np.round(coordsvec[0]*W))
    fx2 = int(np.round(coordsvec[2]*W))
    fy1 = int(np.round(coordsvec[1]*H))
    fy2 = int(np.round(coordsvec[3]*H))    
    fy = int(np.round(coordsvec[5]*H))  # CENTER y
    
    # Check the limits in order to use just the center as a reference
    if fy >= W/2 and fy <= H-(W/2) and (fy2-fy1) <= W:
        x1 = 0
        x2 = W
        y1 = fy-int(W/2)
        y2 = fy+int(W/2)        
    elif (fy2-fy1) > W:
        pad = 1
        x1 = 0
        x2 = W
        y1 = fy1
        y2 = fy2              
    elif fy < H/2:
        x1 = 0
        y1 = 0
        x2 = W
        y2 = W    
    elif fy > H-(W/2):
        x1 = 0
        x2 = W
        y1 = H-W
        y2 = H
    else:
        raise ValueError(f'error with coordinate vector: {coordsvec}')  
    
    frame = frame[y1:y2, x1:x2, :]
    if pad == 1:
        frame = pad_img_to_square(frame) 
        frame = cv2.resize(frame, (W, W))
    assert frame.shape[0] == W and frame.shape[1] == W    
    return frame

def crop_transformed_video(basedir, source_video, stride, unsharp=False, timestamp=True,
                         stabilize=False, suffix='_CROP', filter_type='gaussian',
                         filter_windoworsigma=20, overwrite=False):
    """
    Crop a video based on detected macaques.
    
    Parameters
    ----------
    basedir : str
        Base directory for output
    source_video : str
        Path to source video
    stride : int
        Stride for detection
    unsharp : bool, optional
        Whether to apply unsharp masking
    timestamp : bool, optional
        Whether to add timestamp to frames
    stabilize : bool, optional
        Whether to stabilize the video
    suffix : str, optional
        Suffix for output filename
    filter_type : str, optional
        Type of filter to apply ('gaussian', 'median', or None)
    filter_windoworsigma : int, optional
        Window size or sigma for filter
    overwrite : bool, optional
        Whether to overwrite existing files

    Returns
    -------
    str
        Path to output video
    """
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d
    
    # Create output video name
    if unsharp:
        outvideofile = os.path.join(basedir, os.path.splitext(os.path.basename(source_video))[0]+suffix+'_unsharp.mp4')
    else:
        outvideofile = os.path.join(basedir, os.path.splitext(os.path.basename(source_video))[0]+suffix+'.mp4')
    
    if os.path.exists(outvideofile) and not overwrite:
        return outvideofile
    
    # Get the directory where the yolo labels were saved
    exp = [a for a in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, a))]
    exp.sort()
    exp = exp[-1]
    txtdirpath = os.path.join(basedir, exp, 'labels')

    if stride == 1:
        txtdirpathfixed = txtdirpath
    else:
        txtdirpathfixed = txtdirpath + '_fixed'
        if not os.path.exists(txtdirpathfixed):
            txtdirpathfixed = fix_detected_filenames_stride(txtdirpath, stride, source_video)
    
    # Load the video
    cap = cv2.VideoCapture(source_video)
    numberofframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Get coordinates for cropping
    coords = get_crop_coord_list(txtdirpathfixed, source_video, numberofframes)
    
    if filter_type == 'median':
        coords = rolling_median_by_column(coords, window_size=filter_windoworsigma, output_equal_length=True)
    elif filter_type == 'gaussian':
        coords = gaussian_filter1d(coords, sigma=filter_windoworsigma, axis=0)
    elif filter_type == 'median':
        coords = uniform_filter1d(coords, size=filter_windoworsigma, axis=0)
    
    # Initialize video writer
    out = cv2.VideoWriter(outvideofile, cv2.VideoWriter_fourcc(*'mp4v'), 60, (width, width))
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    
    # Initialize stabilizer if needed
    if stabilize:
        stab = Stabilizer(smoothing_radius=40, border_type='reflect_101')

    # Parameters for timestamp
    font = cv2.FONT_HERSHEY_COMPLEX
    org = (10, width-10)
    color = (0, 0, 255)
    fontScale = 1
    thickness = 2

    frame_idx = 0
    print('\n = = = cropping the video = = =')
    bar = progressbar.ProgressBar(max_value=numberofframes, redirect_stdout=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        
        if stabilize:
            stabilized_frame = stab.stabilize(frame)
            if stabilized_frame is None:
                continue
            framecrop = crop_frame(stabilized_frame, coords[frame_idx,:])
        else:
            framecrop = crop_frame(frame, coords[frame_idx,:])
        
        if unsharp:
            framecrop = cv2.filter2D(framecrop, -1, kernel)
            
        if timestamp:
            _, hh, mm, ss = convert_from_ms(cap.get(cv2.CAP_PROP_POS_MSEC))
            hhmmss = f'{hh:02d}:{mm:02d}:{ss:05.3f}'
            _ = cv2.putText(framecrop, hhmmss, org, font, fontScale, color, thickness,
                          cv2.LINE_AA, bottomLeftOrigin=False)
            
        out.write(framecrop)
        bar.update(frame_idx)
        frame_idx += 1
        
    cap.release()
    out.release()
    return outvideofile

def crop_transformed_video_stabilize(basedir, source_video, stride, suffix='_CROP',
                                  filter_type='gaussian', filter_windoworsigma=20,
                                  overwrite=False):
    """
    Crop and stabilize a video based on detected macaques.
    
    Parameters
    ----------
    basedir : str
        Base directory for output
    source_video : str
        Path to source video
    stride : int
        Stride for detection
    suffix : str, optional
        Suffix for output filename
    filter_type : str, optional
        Type of filter to apply ('gaussian', 'median', or None)
    filter_windoworsigma : int, optional
        Window size or sigma for filter
    overwrite : bool, optional
        Whether to overwrite existing files

    Returns
    -------
    str
        Path to output video
    """
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d
    
    outvideofile = os.path.join(basedir, os.path.splitext(os.path.basename(source_video))[0]+suffix+'.mp4')
    if os.path.exists(outvideofile) and not overwrite:
        return outvideofile
    
    # Get the directory where the yolo labels were saved
    exp = [a for a in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, a))]
    exp.sort()
    exp = exp[-1]
    txtdirpath = os.path.join(basedir, exp, 'labels')

    if stride == 1:
        txtdirpathfixed = txtdirpath
    else:
        txtdirpathfixed = txtdirpath + '_fixed'
        if not os.path.exists(txtdirpathfixed):
            txtdirpathfixed = fix_detected_filenames_stride(txtdirpath, stride, source_video)
    
    # Load the video with VideoGear for stabilization
    options = {'SMOOTHING_RADIUS': 40, 'BORDER_TYPE': 'black'}
    stream_stab = VideoGear(source=source_video, stabilize=True, **options).start()
    cap = cv2.VideoCapture(source_video)
    
    numberofframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Get coordinates for cropping
    coords = get_crop_coord_list(txtdirpathfixed, source_video, numberofframes)
    
    if filter_type == 'median':
        coords = rolling_median_by_column(coords, window_size=filter_windoworsigma, output_equal_length=True)
    elif filter_type == 'gaussian':
        coords = gaussian_filter1d(coords, sigma=filter_windoworsigma, axis=0)
    elif filter_type == 'median':
        coords = uniform_filter1d(coords, size=filter_windoworsigma, axis=0)
    
    # Initialize video writer
    out = cv2.VideoWriter(outvideofile, cv2.VideoWriter_fourcc(*'mp4v'), 60, (width, width))

    frame_idx = 0
    print('\n = = = cropping the video = = =')
    bar = progressbar.ProgressBar(max_value=numberofframes, redirect_stdout=True)
    
    while True:
        frame = stream_stab.read()
        ret, frameCV2 = cap.read()
        if frame is None:
            break
            
        framecrop = crop_frame(frame, coords[frame_idx,:])
        out.write(framecrop)
        bar.update(frame_idx)
        frame_idx += 1
        
    cap.release()
    out.release()
    stream_stab.stop()
    return outvideofile 