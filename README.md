
# YoloCropGUI

YoloCropGUI is a user-friendly graphical interface built with PyQt5 that leverages YOLO models for efficient image cropping based on specified tags. This tool allows users to select models, specify tags to crop or skip, and process images either individually or in bulk, with real-time progress updates and CUDA acceleration support. It supports both `.pt` and TensorRT engine files.

![YoloCropGUI](https://github.com/user-attachments/assets/a8da98bd-beda-4a91-963f-6c0e3f633f05)

## Features

- **Model Selection:** Load and switch between different YOLO models.
- **Tag-Based Cropping:** Specify tags to crop or skip during image processing.
- **Single and Batch Processing:** Process individual images or entire folders of images.
- **CUDA Acceleration:** Utilize GPU acceleration for faster processing if available.
- **Real-Time Progress:** Monitor the progress of image processing with a progress bar.
- **Export Options:** Save cropped images to a specified export folder.
- **Clear Export Folder:** Easily clear the export folder with a single click.
- **Dynamic Parameters:** Automatically adjust font scale and line thickness based on image size.

## Getting Started

### Clone the Repository
```sh
git clone https://github.com/search620/YoloCropGUI.git
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Loading YOLO Models
Place your YOLO models (e.g., `yolov8x-oiv7.pt`) in the root project directory. The GUI reads the list of models from the root directory.

### Run the Application
```sh
python Yolo8_crop_multy_gui.py
```
Note: This version was tested and works on Python 3.10.6.


## Using the GUI

- **Model Path:** Select a YOLO model from the dropdown.
- **Device:** Choose between `cuda:0` (GPU) or `cpu`.
- **Source Folder:** Browse and select the folder containing images to process.
- **Export Folder:** Browse and select the folder to save cropped images.
- **Single Image Path:** Browse and select a single image to process.
- **Tags to Crop:** Enter tags to crop and press Enter to add them to the list.
- **Skip if Included:** Enter tags to skip and press Enter to add them to the list.
- **Mark Box:** Check this option to draw bounding boxes around detected objects and see the detection names recognized by the model.
- **Ignore Tags:** Check this option to ignore tags during processing.
- **Start:** Click to start processing images.
- **Stop:** Click to stop processing images.
- **Clear Export Folder:** Click to clear all files in the export folder.
- **Refresh Models:** Click to refresh the list of available models.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics/tree/v8.2.0)
- PyQt5
- OpenCV
- Torch

## License

This project is licensed under the GNU General Public License v3.0.

---
