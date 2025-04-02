"""
Example script demonstrating advanced video processing operations including negative class generation.
"""

import os
import pandas as pd
import numpy as np
from monkey_genericutils import video_processing, utils

def extract_negative_clips(video_path, behavior_dict, output_dir, basetime=10, negative_class_dir='Other'):
    """
    Extract video clips for negative class (non-behavior periods).
    
    Parameters
    ----------
    video_path : str
        Path to input video
    behavior_dict : dict
        Dictionary mapping behaviors to time intervals
    output_dir : str
        Directory to save output clips
    basetime : int, optional
        Base time interval in seconds
    negative_class_dir : str, optional
        Name of directory for negative class clips
    """
    # Get video duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    # Create list of all behavior intervals
    all_intervals = []
    for intervals in behavior_dict.values():
        all_intervals.extend(intervals)
    
    # Sort intervals
    all_intervals.sort()
    
    # Find gaps between behavior intervals
    negative_intervals = []
    current_time = 0
    
    for start, end in all_intervals:
        if start - current_time >= basetime:
            negative_intervals.append((current_time, start))
        current_time = end
    
    # Add final interval if there's time left
    if duration - current_time >= basetime:
        negative_intervals.append((current_time, duration))
    
    # Extract negative clips
    negative_dir = os.path.join(output_dir, negative_class_dir)
    if not os.path.exists(negative_dir):
        os.makedirs(negative_dir)
    
    for i, (start_time, end_time) in enumerate(negative_intervals):
        start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{start_time%60:06.3f}"
        end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{end_time%60:06.3f}"
        
        output_file = os.path.join(
            negative_dir,
            f"{os.path.splitext(os.path.basename(video_path))[0]}_negative_{i}.mp4"
        )
        
        video_processing.cut_video_clip(
            video_path,
            output_file,
            start_str,
            end_str,
            overwrite=True
        )

def process_annotation_file(annotation_file, behaviors, basetime=10, basepath=None, ffmpegconvert=False):
    """
    Process a generic annotation file to extract video clips.
    
    Parameters
    ----------
    annotation_file : str
        Path to annotation file
    behaviors : list
        List of behaviors to extract
    basetime : int, optional
        Base time interval in seconds
    basepath : str, optional
        Base path for video files
    ffmpegconvert : bool, optional
        Whether to convert videos using ffmpeg
    """
    df = pd.read_csv(annotation_file)
    behavior_dict = {}
    
    for behavior in behaviors:
        behavior_dict[behavior] = []
        for _, row in df.iterrows():
            if row['Behavior'] == behavior:
                tstartsecs = utils.convert_hhmmss_to_seconds(row['Start time'])
                tendsecs = utils.convert_hhmmss_to_seconds(row['Stop time'])
                clip_table = utils.create_clip_cut_table(tstartsecs, tendsecs, basetime)
                behavior_dict[behavior].extend(clip_table)
    
    return behavior_dict

def process_ballesta_annotations(annotation_file, behaviors, basetime=10, basepath=None):
    """
    Process Ballesta format annotation file to extract video clips.
    
    Parameters
    ----------
    annotation_file : str
        Path to Ballesta annotation file
    behaviors : list
        List of behaviors to extract
    basetime : int, optional
        Base time interval in seconds
    basepath : str, optional
        Base path for video files
    """
    df = pd.read_csv(annotation_file)
    behavior_dict = {}
    
    for behavior in behaviors:
        behavior_dict[behavior] = []
        for _, row in df.iterrows():
            if row['Behavior'] == behavior:
                tstartsecs = utils.convert_hhmmss_to_seconds(row['Start time'])
                tendsecs = utils.convert_hhmmss_to_seconds(row['Stop time'])
                clip_table = utils.create_clip_cut_table(tstartsecs, tendsecs, basetime)
                behavior_dict[behavior].extend(clip_table)
    
    return behavior_dict

if __name__ == "__main__":
    # Example usage
    video_path = "path/to/your/video.mp4"
    annotation_file = "path/to/your/annotations.csv"
    output_dir = "path/to/output/directory"
    
    # Define behaviors
    behaviors = ['Feeding', 'Resting', 'Grooming', 'Moving']
    
    # Process annotations
    behavior_dict = process_annotation_file(
        annotation_file,
        behaviors,
        basetime=10,
        basepath=None
    )
    
    # Extract behavior clips
    for behavior, intervals in behavior_dict.items():
        behavior_dir = os.path.join(output_dir, behavior)
        if not os.path.exists(behavior_dir):
            os.makedirs(behavior_dir)
            
        for i, (start_time, end_time) in enumerate(intervals):
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{start_time%60:06.3f}"
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{end_time%60:06.3f}"
            
            output_file = os.path.join(
                behavior_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_{i}.mp4"
            )
            
            video_processing.cut_video_clip(
                video_path,
                output_file,
                start_str,
                end_str,
                overwrite=True
            )
    
    # Extract negative clips
    extract_negative_clips(video_path, behavior_dict, output_dir)
    
    print("Advanced processing complete!") 