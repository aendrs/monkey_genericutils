# Processing Examples

This directory contains example scripts demonstrating how to use the `monkey_genericutils` package for various video processing tasks.

## Example Scripts

### 01_basic_video_processing.py
Demonstrates the basic video processing pipeline:
1. Transform video using ffmpeg
2. Detect macaques using YOLOv5
3. Crop and stabilize the video

Usage:
```python
python 01_basic_video_processing.py
```

### 02_video_clipping.py
Shows how to:
1. Process BORIS annotation files
2. Extract video clips based on behavior annotations
3. Generate dataset files for video classification

Usage:
```python
python 02_video_clipping.py
```

### 03_advanced_processing.py
Demonstrates advanced processing features:
1. Process generic annotation files
2. Process Ballesta format annotations
3. Generate negative class clips
4. Handle multiple behaviors

Usage:
```python
python 03_advanced_processing.py
```

## Prerequisites

Before running these examples, make sure you have:
1. Installed the `monkey_genericutils` package
2. YOLOv5 detection script and weights (for basic processing)
3. Required input files (videos and annotations)
4. Sufficient disk space for output files

## Configuration

Each script contains example paths that need to be updated with your actual file paths:
- Input video paths
- Output directory paths
- YOLOv5 weights and script paths
- Annotation file paths

## Notes

- All scripts include example usage in their `if __name__ == "__main__":` blocks
- Modify the parameters (e.g., `basetime`, `behaviors`) according to your needs
- The scripts assume standard video formats (mp4) and annotation formats (CSV)
- Make sure you have write permissions in the output directories 