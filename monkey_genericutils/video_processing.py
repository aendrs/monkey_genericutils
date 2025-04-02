"""
Functions for video processing and transformation.
"""

import os
import subprocess
import cv2
import numpy as np
import progressbar
import pandas as pd
import datetime
import shutil

def ffmpeg_video_transformation(video_original, rootdir='/data/socialeyes/EthoProcessedVideos', overwrite=False):
    """
    Transform a video using ffmpeg.
    
    Parameters
    ----------
    video_original : str
        Path to original video
    rootdir : str, optional
        Root directory for output
    overwrite : bool, optional
        Whether to overwrite existing files

    Returns
    -------
    tuple
        (basedir, videosmall) - Paths to output directory and transformed video
    """
    if 'Camera0' in video_original:
        camera = 'Camera0'
    elif 'Camera1' in video_original:
        camera = 'Camera1'
    else:    
        camera = 'other'
        
    basedir = os.path.join(rootdir, os.path.basename(os.path.dirname(video_original)), camera)
    if not os.path.exists(basedir):
        os.makedirs(basedir) 
        
    videosmall = os.path.join(basedir, os.path.splitext(os.path.basename(video_original))[0]+'_small.mp4')
    if os.path.exists(videosmall) and not overwrite:
        return basedir, videosmall
        
    ffmpegstring = f'ffmpeg -i "{video_original}" -vf "fps=60, scale=iw/2:ih/2, eq=saturation=1.1,unsharp" -c:v mpeg4 -qscale:v 2 -vsync 1 "{videosmall}"'
    subprocess.call(ffmpegstring, shell=True)
    return basedir, videosmall

def cut_video_clip(video_in, video_out, time_start, time_end, overwrite=False):
    """
    Cut a video clip between specified times.
    
    Parameters
    ----------
    video_in : str
        Path to input video
    video_out : str
        Path to output video
    time_start : str
        Start time in format 'hh:mm:ss.sss'
    time_end : str
        End time in format 'hh:mm:ss.sss'
    overwrite : bool, optional
        Whether to overwrite existing files
    """
    assert os.path.exists(video_in), 'Input Video file not found'
    if os.path.exists(video_out) and not overwrite:
        return
        
    ffmpegstring = f'ffmpeg -i "{video_in}" -ss {time_start} -to {time_end} -c:v copy -c:a copy {video_out}'
    subprocess.call(ffmpegstring, shell=True)

def cut_video_clip_crop(video_in, video_out, time_start, time_end, crop_ffmpeg_string, overwrite=False):
    """
    Cut and crop a video clip.
    
    Parameters
    ----------
    video_in : str
        Path to input video
    video_out : str
        Path to output video
    time_start : str
        Start time in format 'hh:mm:ss.sss'
    time_end : str
        End time in format 'hh:mm:ss.sss'
    crop_ffmpeg_string : str
        FFmpeg crop string in format 'width:height:x:y'
    overwrite : bool, optional
        Whether to overwrite existing files
    """
    assert os.path.exists(video_in), f'Input Video file not found, {video_in}'
    if os.path.exists(video_out) and not overwrite:
        return
        
    ffmpegstring = f'ffmpeg -i "{video_in}" -ss {time_start} -to {time_end} -filter:v "crop={crop_ffmpeg_string}" {video_out}'
    subprocess.call(ffmpegstring, shell=True)

def pad_video(basedir, source_video, suffix='_CROP', overwrite=False):
    """
    Pad a video to make it square.
    
    Parameters
    ----------
    basedir : str
        Base directory for output
    source_video : str
        Path to source video
    suffix : str, optional
        Suffix for output filename
    overwrite : bool, optional
        Whether to overwrite existing files

    Returns
    -------
    str
        Path to output video
    """
    from .utils import pad_img_to_square
    
    outvideofile = os.path.join(basedir, os.path.splitext(os.path.basename(source_video))[0]+suffix+'.mp4')
    if os.path.exists(outvideofile) and not overwrite:
        return outvideofile
    
    cap = cv2.VideoCapture(source_video)
    numberofframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    maxside = max(width, height)
    out = cv2.VideoWriter(outvideofile, cv2.VideoWriter_fourcc(*'mp4v'), 60, (maxside, maxside))

    frame_idx = 0
    print('\n = = = padding video = = =')
    while True:
        ret, frameCV2 = cap.read()
        if not ret:
            break              
        framepad = pad_img_to_square(frameCV2)
        out.write(framepad)
        frame_idx += 1
        
    cap.release()
    out.release()
    return outvideofile

def generate_video_dataset_file(rootdir, output_file, classdir={'chilling':0, 'foraging':1, 'grooming':2, 'moving':3}):
    """
    Generate video annotation TXT file for MMACTION2 library.
    
    Parameters
    ----------
    rootdir : str
        Root directory containing video files
    output_file : str
        Path to output annotation file
    classdir : dict, optional
        Dictionary mapping class names to indices
    """
    filelist = []
    for root, dirs, files in os.walk(rootdir, topdown=False):
        for file in files:
            filelist.append(os.path.join(root, file)+' '+str(classdir[os.path.split(root)[1]]))
            
    with open(output_file, 'w') as fp:
        fp.write('\n'.join(filelist)) 