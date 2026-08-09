import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# OCR Preprocessing Exploration
In this notebook, we'll experiment with OpenCV to clean a noisy image before passing it to Tesseract.
"""

code = """\
import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np

def show_image(img, title="Image"):
    plt.figure(figsize=(10, 6))
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
    
# Load test image
img = cv2.imread('../data/raw/synthetic_noisy_test.jpg')
show_image(img, 'Raw Image')
"""

code2 = """\
# 1. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_image(gray, 'Grayscale')
"""

code3 = """\
# 2. Noise Removal (Blur)
blurred = cv2.medianBlur(gray, 3)
show_image(blurred, 'Blurred')
"""

code4 = """\
# 3. Thresholding
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
show_image(thresh, 'Thresholded')
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4)
]

with open('../notebooks/01_exploration.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created 01_exploration.ipynb")
