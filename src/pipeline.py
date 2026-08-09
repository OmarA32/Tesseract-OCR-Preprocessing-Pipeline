import cv2
import numpy as np

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    # Use a gentle NLMeans pass to eliminate background noise without destroying thin text
    return cv2.fastNlMeansDenoising(image, None, h=3, templateWindowSize=7, searchWindowSize=21)

def apply_thresholding(image):
    # Standard adaptive thresholding parameters that don't erase thin text
    return cv2.adaptiveThreshold(
        image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

def deskew(image):
    # Deskewing logic
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def morphological_operations(image):
    # 1. Opening: Removes tiny isolated noise specks from the background
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
    
    # 2. Dilation: Expands the white pixels (text) to fill in any hollow gaps
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(opened, kernel_dilate, iterations=1)
    
    # 3. Blob Filtering: Mathematically eradicate remaining tiny pepper noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    cleaned = np.zeros_like(dilated)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 50:
            cleaned[labels == i] = 255
            
    return cleaned

def process_image(image_path, output_path=None):
    """Run the full preprocessing pipeline on an image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    gray = to_grayscale(image)
    no_noise = remove_noise(gray)
    thresh = apply_thresholding(no_noise)
    
    # Invert image to deskew properly (deskew needs white text on black background for minAreaRect)
    inverted = cv2.bitwise_not(thresh)
    deskewed = deskew(inverted)
    
    # Invert back to normal
    final_image = cv2.bitwise_not(deskewed)
    final_image = morphological_operations(final_image)

    if output_path:
        cv2.imwrite(output_path, final_image)
        
    return final_image
