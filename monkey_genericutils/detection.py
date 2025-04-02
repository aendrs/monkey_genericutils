"""
Functions for macaque detection in videos.
"""

import os
import shutil
import subprocess
import progressbar
import numpy as np

def get_detected_list(txtdirpath):
    """Get list of detected frames from YOLO output files."""
    detectedlist = os.listdir(txtdirpath)
    detectedlist = [int(a.split('_')[-1].split('.')[0]) for a in detectedlist]
    detectedlist.sort()
    return detectedlist

def fix_detected_filenames_stride(txtdirpath, stride, videofile):
    """
    Fix YOLO output file indices according to stride.
    
    Yolo saves txt files with indices corresponding to the frame number, but if a stride was used 
    the indexing is wrong. The correct index is calculated by real_frame_idx=stride*(frame_idx-1)+1,
    remember that it starts counting on 1, not 0.
    """
    txtdirpathfixed = txtdirpath + '_fixed'
    if not os.path.exists(txtdirpathfixed):
        os.makedirs(txtdirpathfixed)
    detectedlist = os.listdir(txtdirpath)
    detectedlist.sort()
    videoname = os.path.splitext(os.path.basename(videofile))[0]
    print('\n Fixing the frame indices with correct stride ===== \n')
    if stride > 1:
        for file in progressbar.progressbar(detectedlist):
            n = int(file.split('_')[-1].split('.')[0])
            real_n = stride*(n-1)+1
            shutil.copy2(
                os.path.join(txtdirpath, file),
                os.path.join(txtdirpathfixed, videoname+'_'+str(real_n)+'.txt')
            )
    return txtdirpathfixed

def get_crop_coord_list(txtdirpath, videofile, numberofframes):
    """
    Get list of crop coordinates for each frame.
    
    Returns an np array with coords: x1,x2,y1,y2,center0,center1
    x1,x2,y1,y2 refer to the megabox corners containing the detected macaques
    the coords are expressed in relative coords
    """
    from .utils import get_XY_coords_from_roi
    
    detectedlist = get_detected_list(txtdirpath)
    videofile = os.path.basename(videofile)
    coords = np.zeros((numberofframes, 6))
    x1, y1, x2, y2 = (0 for i in range(4))
    center = [0, 0]
    print('\n \n Getting the coordinate list of the cropped frames')
    
    for f in progressbar.progressbar(range(1, numberofframes+1)):
        if f in detectedlist:
            rois = np.loadtxt(os.path.join(txtdirpath, os.path.splitext(videofile)[0]+'_'+str(f)+'.txt'))
            if rois.size == 0:
                continue
            if len(rois.shape) == 2:
                rois = np.delete(rois, 0, axis=1)
            else:
                rois = np.delete(rois, 0)
                rois = rois.reshape(1, rois.shape[0])
            x1, y1, x2, y2, center = get_XY_coords_from_roi(rois)
            coords[f-1,:] = x1, y1, x2, y2, center[0], center[1]
        else:
            coords[f-1,:] = x1, y1, x2, y2, center[0], center[1]
    return coords

def macaquedetector3000(detect_script, weights, stride, basedir, source_video):
    """Run YOLOv5 detection on video."""
    detectstring = f'python {detect_script} --weights {weights} --source {source_video} --device=0 --line-thickness=3 \
        --iou-thres=0.5 --max-det=2 --save-txt --project {basedir} --vid-stride={stride}'
    subprocess.run(detectstring, shell=True) 