# Image Filters and Color Segmentation

Computer vision coursework project that applies:

- custom spatial filters (box, Gaussian, Sobel-style edge detection),
- and HSV color-based object segmentation.

The implementation is in `image_filters.py` and runs over images from a local `resources/` folder.

## What the project does

### 1) Smoothing and edge filtering

For each image in `resources/`, the script:

1. builds a box filter and convolves the image,
2. builds a Gaussian filter and convolves the image,
3. converts to grayscale and applies Sobel-like edge filters.

Results are written under `results/` in these folders:

- `results/box_5/`
- `results/gaussian_5/`
- `results/edge_5/`

### 2) Color-based object extraction (HSV)

For each image in `resources/`, the script also keeps only pixels in selected HSV ranges:

- blue pool -> `results/blue_pool/`
- orange building -> `results/orange_building/`
- red H -> `results/red_h/`

Pixels outside the selected range are set to black.

## Project structure

- `image_filters.py`: all logic (convolution, filter creation, HSV masking, pipeline run).

## Setup

Install dependencies:

```bash
pip install numpy opencv-python pillow
```

Create expected folders (if they do not already exist):

```bash
mkdir -p resources results/box_5 results/gaussian_5 results/edge_5 results/blue_pool results/orange_building results/red_h
```

Put input images (`.jpg`, `.png`, etc.) in `resources/`.

## Run

From this directory:

```bash
python image_filters.py
```

## How to observe results

- Open output images inside the `results/` subfolders listed above.
- Compare each source image with:
  - smoothed outputs (`box_5`, `gaussian_5`),
  - edge output (`edge_5`),
  - color masks (`blue_pool`, `orange_building`, `red_h`).

This project was developed as coursework in computer vision; shared for learning and reproducibility.
