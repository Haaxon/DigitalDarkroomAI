import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
manual_image_inverter.py

A script for manually inverting and color-correcting scanned photographic negatives.

Overview:
---------
This tool provides a step-by-step workflow for manually converting a photographic negative into a positive image,
with fine-tuned control over color channel levels and color balance. It leverages OpenCV and NumPy for
image processing, and Matplotlib for visualization.

Workflow:
---------
1. Load a scanned negative image.
2. Display the original image.
3. Invert the image to create a positive.
4. Adjust red, green, and blue channel levels using customizable min/max values.
5. Apply color balance corrections (cyan-red, magenta-green, yellow-blue).
6. Display each step for visual feedback.

Usage:
------
- Set the `INPUT_IMAGE` variable to your image file path.
- Optionally adjust the channel min/max and color balance constants to suit your negative.
- Run the script. Each processing step will be shown using Matplotlib.
- To save the final result, uncomment the `cv2.imwrite` line at the end.

Example:
--------
    INPUT_IMAGE = "path/to/your/negative.jpg"
    # Adjust RED_MIN, RED_MAX, etc. as needed
    # Run the script to process and visualize the result.
"""

INPUT_IMAGE = ""        # ranges suggest by Geppetto:
RED_MIN     = 60        # 40-80
RED_MAX     = 250       # 230-255
GREEN_MIN   = 45        # 30-70
GREEN_MAX   = 240       # 220-250
BLUE_MIN    = 30        # 10-40
BLUE_MAX    = 200       # 180-220
CYAN_RED        = -15   # -30-20
MAGENTA_GREEN   = 0     # -20-20
YELLOW_BLUE     = 10    # -20-15
 
def show_image(image, title="Image"):
    # Display an image using matplotlib (converts BGR to RGB).
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis("off")
    plt.show()

def invert_image(image):
    inverted = 255 - image
    return inverted

def adjust_channel_levels(image,r_min=0, r_max =255,
                                g_min=0, g_max=255,
                                b_min=0, b_max=255):
    
    image_float = image.astype(np.float32)

    def remap_channel(channel, black, white):
        channel = (channel - black) / (white - black) * 255.0
        return np.clip(channel, 0, 255)

    image_float[..., 0] = remap_channel(image_float[..., 0], b_min, b_max)
    image_float[..., 1] = remap_channel(image_float[..., 1], g_min, g_max)
    image_float[..., 2] = remap_channel(image_float[..., 2], r_min, r_max)

    adjusted = image_float.astype(np.uint8)
    return adjusted

def apply_color_balance(image, cyan_red=0.0, magenta_green=0.0, yellow_blue=0.0):
    image_float = image.astype(np.float32)

    image_float[..., 2] += cyan_red
    image_float[..., 1] += magenta_green
    image_float[..., 0] += yellow_blue

    balanced = np.clip(image_float, 0, 255).astype(np.uint8)
    
    return balanced

image = cv2.imread(INPUT_IMAGE)
if image is None:
    raise ValueError(f"Could not load image at {INPUT_IMAGE}")

show_image(image, "Original Negative")

inverted = invert_image(image)
show_image(inverted, "Inverted Image")

leveled = adjust_channel_levels(inverted,
                                RED_MIN, RED_MAX,
                                GREEN_MIN, GREEN_MAX,
                                BLUE_MIN, BLUE_MAX)
show_image(leveled, "After Channel Level Adjustment")

color_corrected = apply_color_balance(leveled, CYAN_RED, MAGENTA_GREEN, YELLOW_BLUE)
show_image(color_corrected, "After Color Balance")

#cv2.imwrite(output_path, color_corrected)
#print(f"Saved corrected image to {output_path}")
