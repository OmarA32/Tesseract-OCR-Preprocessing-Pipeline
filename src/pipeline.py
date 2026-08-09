import cv2
import numpy as np
import math

def to_grayscale(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    # Use a gentle NLMeans pass to eliminate background noise without destroying thin text
    return cv2.fastNlMeansDenoising(image, None, h=3, templateWindowSize=7, searchWindowSize=21)

def apply_thresholding(image):
    # Standard adaptive thresholding parameters that don't erase thin text
    # Increased block size to 31 to prevent thick letters from becoming hollow
    return cv2.adaptiveThreshold(
        image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 31, 2
    )

def get_angle(x1, y1, x2, y2):
    """Get the angle of this line with the horizontal axis."""
    deltaX = x2 - x1
    deltaY = y2 - y1
    return np.arctan2(deltaY , deltaX) * 180 / math.pi

def deskew(image):
    # The image here is inverted (white text on black background)
    edges = cv2.Canny(image, 80, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=20, maxLineGap=10)
    
    if lines is None:
        return image
        
    # Sort lines from widest to shortest (OpenCV may return (N, 4) or (N, 1, 4))
    if len(lines.shape) == 3:
        lines = lines[:, 0, :]
    lines = sorted(lines, key=(lambda l: abs(l[0]-l[2])), reverse=True)
    
    # Use the widest line to determine rotation angle
    angle = 0.0
    for x1, y1, x2, y2 in lines:
        # Check if line spans at least 25% of the image width
        if (abs(x2-x1) / image.shape[1]) > 0.25:
            angle = get_angle(x1, y1, x2, y2)
            break
            
    if abs(angle) < 1.0 or abs(angle) > 45.0:
        return image
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Binarize the cubic-interpolated image to keep edges perfectly smooth without jagged "chunks"
    _, rotated_binary = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
    
    return rotated_binary

def morphological_operations(image):
    # 1. Blob Filtering FIRST: Mathematically eradicate tiny pepper noise BEFORE it gets dilated
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    cleaned = np.zeros_like(image)
    for i in range(1, num_labels):
        # Threshold of 25 perfectly preserves the dot on the 'i' (area ~35) while deleting noise (area 1-20)
        if stats[i, cv2.CC_STAT_AREA] >= 25:
            cleaned[labels == i] = 255
            
    # 2. Dilation: Expands the cleanly filtered white pixels (text) to make them delightfully bold
    # Using a MORPH_CROSS kernel adds subtle, symmetric weight without bleeding small text together
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(cleaned, kernel_dilate, iterations=1)
            
    return dilated

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
    
    # Morphological operations expect white text on black background
    cleaned = morphological_operations(deskewed)
    
    # Invert back to normal (black text on white background)
    final_image = cv2.bitwise_not(cleaned)

    if output_path:
        cv2.imwrite(output_path, final_image)
        
    return final_image
