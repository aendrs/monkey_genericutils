"""
Example script demonstrating basic video processing operations using monkey_genericutils.
"""

import os
from monkey_genericutils import video_processing, detection, cropping

def process_video(input_video, output_dir, weights_path, detect_script_path, stride=1):
    """
    Process a video through the complete pipeline:
    1. Transform video using ffmpeg
    2. Detect macaques using YOLOv5
    3. Crop and stabilize the video
    """
    # Step 1: Transform video
    print("Step 1: Transforming video...")
    basedir, videosmall = video_processing.ffmpeg_video_transformation(
        input_video,
        rootdir=output_dir,
        overwrite=True
    )
    
    # Step 2: Detect macaques
    print("Step 2: Detecting macaques...")
    #stride = 1  # Process every frame
    detection.macaquedetector3000(
        detect_script=detect_script_path,
        weights=weights_path,
        stride=stride,
        basedir=basedir,
        source_video=videosmall
    )
    
    # Step 3: Crop and stabilize video
    print("Step 3: Cropping and stabilizing video...")
    output_video = cropping.crop_transformed_video_stabilize(
        basedir=basedir,
        source_video=videosmall,
        stride=stride,
        suffix='_CROP',
        filter_type='gaussian',
        filter_windoworsigma=20,
        overwrite=True
    )
    
    return output_video

if __name__ == "__main__":
    # Example usage
    input_video = "path/to/your/input.mp4"
    output_dir = "path/to/output/directory"
    weights_path = "path/to/yolov5/weights.pt"
    detect_script_path = "path/to/yolov5/detect.py"
    stride=3 # this value works fine in practice

    output_video = process_video(input_video, output_dir, weights_path, detect_script_path, stride)
    print(f"Processing complete. Output video saved to: {output_video}") 