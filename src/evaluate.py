import pytesseract
import cv2
import sys
import matplotlib.pyplot as plt
from pipeline import process_image

def evaluate_ocr(image_path, processed_path):
    print(f"--- Evaluating {image_path} ---")
    
    # 1. OCR on Raw Image
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        print(f"Error: Could not load {image_path}")
        return
        
    print("\n[RAW IMAGE OCR TEXT]")
    try:
        raw_text = pytesseract.image_to_string(raw_image)
        print(raw_text.strip() if raw_text.strip() else "<No text detected>")
    except Exception as e:
        print(f"Error running Tesseract: {e}")

    # 2. Process the image
    print(f"\n[PROCESSING IMAGE...]")
    processed_image = process_image(image_path, processed_path)
    print(f"Saved processed image to {processed_path}")
    
    # 3. OCR on Processed Image
    print("\n[PROCESSED IMAGE OCR TEXT]")
    try:
        processed_text = pytesseract.image_to_string(processed_image)
        print(processed_text.strip() if processed_text.strip() else "<No text detected>")
    except Exception as e:
        print(f"Error running Tesseract: {e}")

    # 4. Show visual comparison
    plt.figure(figsize=(12, 6))
    
    # Raw Image Subplot
    plt.subplot(1, 2, 1)
    # Convert BGR to RGB for matplotlib
    plt.imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Processed Image Subplot
    plt.subplot(1, 2, 2)
    # Processed image might be grayscale or binary (2D array)
    if len(processed_image.shape) == 2:
        plt.imshow(processed_image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Processed Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <path_to_raw_image> <path_to_save_processed_image>")
        sys.exit(1)
        
    raw_path = sys.argv[1]
    proc_path = sys.argv[2]
    evaluate_ocr(raw_path, proc_path)
