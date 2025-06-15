import os
import cv2
from random import gauss
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

"""
This script generates synthetic imitations of C41 color film scans from standard positive images.

The main purpose is to simulate the visual characteristics of scanned color negatives, including:
- The orange mask tint typical of C41 film (by shifting color channels).
- Reduced contrast and compressed dynamic range in each color channel.
- Inversion of the image to mimic the negative format.

Key Features:
- Processes all PNG images in a specified input directory.
- Applies randomized color tints and channel compression to each image to simulate film variability.
- Inverts the processed image to produce a negative.
- Saves the resulting synthetic negative images to an output directory.
- Supports configuration for color shifts, channel compression ranges, and output overwriting.

Configuration:
- INPUT_DIR: Directory containing input images.
- OUTPUT_DIR: Directory to save processed images.
- OVERWRITE: Whether to overwrite existing output files.
- STD_DEV: Standard deviation for randomization of color and channel parameters.
- CYAN_RED, MAGENTA_GREEN, YELLOW_BLUE: Parameters for simulating the orange mask.
- RED_MIN, RED_MAX, GREEN_MIN, GREEN_MAX, BLUE_MIN, BLUE_MAX: Channel compression ranges.

Usage:
- Set INPUT_DIR and OUTPUT_DIR.
- Run the script to generate synthetic C41 negative scans.
"""

# CONFIGURATION
INPUT_DIR       = ""    # Directory containing input images.
OUTPUT_DIR      = ""    # Directory to save processed images.
OVERWRITE       = False # Whether to overwrite existing output files.
STD_DEV         = 15        # Standard deviation for randomization of color and channel parameters.
CYAN_RED        = -100   # Add red (simulates orange mask tint).
MAGENTA_GREEN   = 5      # Slight magenta tint (simulates orange mask tint).
YELLOW_BLUE     = 50     # Reduce blue (simulates orange mask tint).
RED_MIN         = 50    # Minimum red channel value (simulates low contrast of scanned negatives).
RED_MAX         = 220   # Maximum red channel value (simulates low contrast of scanned negatives).
GREEN_MIN       = 50    # Minimum green channel value (simulates low contrast of scanned negatives).
GREEN_MAX       = 210   # Maximum green channel value (simulates low contrast of scanned negatives).
BLUE_MIN        = 60    # Minimum blue channel value (simulates low contrast of scanned negatives).
BLUE_MAX        = 200   # Maximum blue channel value (simulates low contrast of scanned negatives).

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_image_file(path):
    return path.lower().endswith(".png")

def show_image(image, title="Image"):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis("off")
    plt.show()

def apply_color_tint(image, cyan_red=0, magenta_green=0, yellow_blue=0):
    image_float = image.astype(np.float32)
    image_float[..., 2] += cyan_red      # Red channel
    image_float[..., 1] += magenta_green # Green channel
    image_float[..., 0] += yellow_blue   # Blue channel
    return np.clip(image_float, 0, 255).astype(np.uint8)

def degrade_channel_levels(image, r_min=80, r_max=180,
                                  g_min=80, g_max=180,
                                  b_min=80, b_max=180):
    image_float = image.astype(np.float32)

    def compress_channel(channel, out_min, out_max):
        # Map from [0, 255] → [out_min, out_max]
        channel = (channel / 255.0) * (out_max - out_min) + out_min
        return channel

    image_float[..., 0] = compress_channel(image_float[..., 0], b_min, b_max)
    image_float[..., 1] = compress_channel(image_float[..., 1], g_min, g_max)
    image_float[..., 2] = compress_channel(image_float[..., 2], r_min, r_max)

    return np.clip(image_float, 0, 255).astype(np.uint8)

def invert_image(image):
    return 255 - image

def process_image(input_dir, output_path):
    image = cv2.imread(input_dir)
    if image is None:
        print(f"Warning: Could not load {input_dir}")
        return

    # Randomize per-image settings
    cyan_red        = gauss(CYAN_RED, STD_DEV)
    magenta_green   = gauss(MAGENTA_GREEN, STD_DEV)
    yellow_blue     = gauss(YELLOW_BLUE, STD_DEV)
    red_min     = gauss(RED_MIN, STD_DEV)
    red_max     = gauss(RED_MAX, STD_DEV)
    green_min   = gauss(GREEN_MIN, STD_DEV)
    green_max   = gauss(GREEN_MAX, STD_DEV)
    blue_min    = gauss(BLUE_MIN, STD_DEV)
    blue_max    = gauss(BLUE_MAX, STD_DEV)

    # Pipeline
    tinted     = apply_color_tint(image, cyan_red, magenta_green, yellow_blue)
    flattened  = degrade_channel_levels(tinted, red_min, red_max, green_min, green_max, blue_min, blue_max)
    negative   = invert_image(flattened)

    cv2.imwrite(output_path, negative)
    print(f"Saved: {output_path}")


# Process all images
print(f"Looking for PNG images in: {INPUT_DIR}")
input_files = [f for f in glob(os.path.join(INPUT_DIR, "*")) if is_image_file(f)]
print(f"Found {len(input_files)} PNG files")

counter = 0
for input_path in input_files:

    filename = os.path.basename(input_path)
    output_path = os.path.join(OUTPUT_DIR, filename)

    if not OVERWRITE and os.path.exists(output_path):
        print(f"Skipping (exists): {filename}")
        continue

    process_image(input_path, output_path)
