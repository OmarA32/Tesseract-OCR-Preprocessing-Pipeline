import cv2
import matplotlib.pyplot as plt
from pipeline import to_grayscale, remove_noise, apply_thresholding, deskew, morphological_operations

def visualize_all_steps(image_path):
    print(f"Loading {image_path}...")
    original = cv2.imread(image_path)
    if original is None:
        print("Error: Could not load image.")
        return

    # Run steps manually
    gray = to_grayscale(original)
    denoised = remove_noise(gray)
    thresh = apply_thresholding(denoised)
    inverted = cv2.bitwise_not(thresh)
    deskewed = deskew(inverted)
    final_image = morphological_operations(cv2.bitwise_not(deskewed))

    # Setup matplotlib grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.canvas.manager.set_window_title('Pipeline Steps Visualization')

    def show_img(ax, img, title, is_gray=True):
        if is_gray:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

    show_img(axes[0, 0], original, '1. Original', is_gray=False)
    show_img(axes[0, 1], gray, '2. Grayscale')
    show_img(axes[0, 2], denoised, '3. Denoised (Median Blur)')
    show_img(axes[1, 0], thresh, '4. Adaptive Threshold')
    show_img(axes[1, 1], deskewed, '5. Deskewed (Inverted)')
    show_img(axes[1, 2], final_image, '6. Final Morphological')

    plt.tight_layout()
    print("Close the window to continue...")
    plt.show()

if __name__ == "__main__":
    visualize_all_steps("data/raw/synthetic_noisy_test.jpg")
