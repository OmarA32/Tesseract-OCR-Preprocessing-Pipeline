import cv2
import numpy as np

# Create a white image
img = np.ones((300, 600, 3), dtype=np.uint8) * 255

# Add some text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Tesseract OCR', (50, 100), font, 2, (0, 0, 0), 3, cv2.LINE_AA)
cv2.putText(img, 'Preprocessing Pipeline', (50, 180), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(img, 'Testing 123', (50, 250), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

# Add noise (salt and pepper)
noise = np.zeros(img.shape, np.uint8)
cv2.randu(noise, 0, 255)
black = noise < 30
white = noise > 225
img[black] = 0
img[white] = 255

# Add a fake shadow / uneven lighting
shadow = np.zeros(img.shape, np.float32)
for i in range(img.shape[1]):
    shadow[:, i] = 1.0 - (i / img.shape[1]) * 0.5
img = (img * shadow).astype(np.uint8)

# Rotate slightly to test deskewing
M = cv2.getRotationMatrix2D((300, 150), 5, 1) # Rotate by 5 degrees
img = cv2.warpAffine(img, M, (600, 300), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

cv2.imwrite('../data/raw/synthetic_noisy_test.jpg', img)
print("Created synthetic test image at ../data/raw/synthetic_noisy_test.jpg")
