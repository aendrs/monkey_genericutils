"""
Example script demonstrating video clipping and dataset generation operations.
"""

import os
import pandas as pd
from monkey_genericutils import video_processing, utils

def create_clip_cut_table(tstartsecs, tendsecs, basetime=10):
    """
    Create a table of time intervals for video clipping.
    
    Parameters
    ----------
    tstartsecs : float
        Start time in seconds
    tendsecs : float
        End time in seconds
    basetime : int, optional
        Base time interval in seconds
        
    Returns
    -------
    list
        List of (start_time, end_time) tuples
    """
    tstartsecs = float(tstartsecs)
    tendsecs = float(tendsecs)
    tstartsecs = tstartsecs - (tstartsecs % basetime)
    tendsecs = tendsecs + (basetime - (tendsecs % basetime))
    tstartsecs = int(tstartsecs)
    tendsecs = int(tendsecs)
    
    clip_table = []
    for t in range(tstartsecs, tendsecs, basetime):
        clip_table.append((t, t + basetime))
    return clip_table

def open_boris_csv(file, behaviors=['Feeding', 'Resting', 'Grooming'], basetime=10):
    """
    Open and process a BORIS CSV file.
    
    Parameters
    ----------
    file : str
        Path to BORIS CSV file
    behaviors : list, optional
        List of behaviors to extract
    basetime : int, optional
        Base time interval in seconds
        
    Returns
    -------
    dict
        Dictionary mapping behaviors to their time intervals
    """
    df = pd.read_csv(file)
    behavior_dict = {}
    
    for behavior in behaviors:
        behavior_dict[behavior] = []
        for _, row in df.iterrows():
            if row['Behavior'] == behavior:
                tstartsecs = utils.convert_hhmmss_to_seconds(row['Start time'])
                tendsecs = utils.convert_hhmmss_to_seconds(row['Stop time'])
                clip_table = create_clip_cut_table(tstartsecs, tendsecs, basetime)
                behavior_dict[behavior].extend(clip_table)
    
    return behavior_dict

def extract_behavior_clips(video_path, behavior_dict, output_dir, basefilesuffix='_small.mp4'):
    """
    Extract video clips for different behaviors.
    
    Parameters
    ----------
    video_path : str
        Path to input video
    behavior_dict : dict
        Dictionary mapping behaviors to time intervals
    output_dir : str
        Directory to save output clips
    basefilesuffix : str, optional
        Suffix for output filenames
    """
    for behavior, intervals in behavior_dict.items():
        behavior_dir = os.path.join(output_dir, behavior)
        if not os.path.exists(behavior_dir):
            os.makedirs(behavior_dir)
            
        for i, (start_time, end_time) in enumerate(intervals):
            # Convert seconds to hh:mm:ss.sss format
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{start_time%60:06.3f}"
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{end_time%60:06.3f}"
            
            output_file = os.path.join(
                behavior_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_{i}{basefilesuffix}"
            )
            
            video_processing.cut_video_clip(
                video_path,
                output_file,
                start_str,
                end_str,
                overwrite=True
            )

def generate_dataset(video_dir, output_file, class_mapping=None):
    """
    Generate a dataset file for video classification.
    
    Parameters
    ----------
    video_dir : str
        Directory containing video clips
    output_file : str
        Path to output annotation file
    class_mapping : dict, optional
        Dictionary mapping class names to indices
    """
    if class_mapping is None:
        class_mapping = {
            'chilling': 0,
            'foraging': 1,
            'grooming': 2,
            'moving': 3
        }
    
    video_processing.generate_video_dataset_file(
        video_dir,
        output_file,
        class_mapping
    )

if __name__ == "__main__":
    # Example usage
    video_path = "path/to/your/video.mp4"
    boris_file = "path/to/your/boris_annotations.csv"
    output_dir = "path/to/output/directory"
    
    # Process BORIS annotations
    behaviors = ['Feeding', 'Resting', 'Grooming']
    behavior_dict = open_boris_csv(boris_file, behaviors, basetime=10)
    
    # Extract clips for each behavior
    extract_behavior_clips(video_path, behavior_dict, output_dir)
    
    # Generate dataset file
    output_file = os.path.join(output_dir, "dataset.txt")
    generate_dataset(output_dir, output_file)
    
    print("Dataset generation complete!") 