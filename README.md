# Tesseract OCR Preprocessing Pipeline

This repository contains an advanced image preprocessing pipeline designed to optimize scanned documents and noisy images for Tesseract OCR.

## Pipeline Overview

The pipeline executes a highly robust 6-step sequence to eliminate noise, handle irregular shadows, correct skew, and perfectly preserve thin text:

1. **Grayscale Conversion**: Converts the image to a single channel.
2. **NLMeans Denoising**: Uses a gentle `fastNlMeansDenoising` pass (`h=3`) to remove background speckles without destroying thin text details.
3. **Adaptive Thresholding**: Utilizes Gaussian adaptive thresholding (Block Size: 31) to cleanly binarize the image, perfectly handling shadows and uneven lighting without hollowing out thick text.
4. **Median Blur**: Applies a gentle 3x3 median blur immediately after thresholding to cleanly erase 1-pixel salt-and-pepper noise at the source, while leaving dense text untouched.
5. **Deskewing (INTER_CUBIC)**: Uses Canny Edge Detection and `HoughLinesP` to find horizontal text lines, calculates the skew angle, and mathematically rotates the image. It uses `INTER_CUBIC` interpolation followed by a strict binary threshold (`127`) to achieve perfect sub-pixel anti-aliasing without jagged "bitten" edges.
6. **Blob Filtering & Dilation**:
    - *Blob Filtering*: Uses Connected Component Analysis to mathematically eradicate any remaining noise blobs under 25 pixels in area (perfectly preserving small punctuation like the dot on an 'i').
    - *Dilation*: Uses a subtle `3x3` `MORPH_CROSS` kernel to expand the cleanly filtered text, making it delightfully bold without bleeding small letters together.

## Visualization

The sequence of these 6 steps can be visualized perfectly below on an extreme synthetic test image with 4% salt-and-pepper noise, uneven lighting, and a 5-degree rotation:

![Pipeline Visualization](data/processed/visualization_grid.png)

## Usage

You can run the full pipeline and visualize the steps by running:
```bash
python src/visualize_steps.py
```
