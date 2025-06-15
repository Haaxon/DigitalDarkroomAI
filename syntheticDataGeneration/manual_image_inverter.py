import cv2
import numpy as np
import matplotlib.pyplot as plt

INPUT_IMAGE = "fake_negative.jpg"
RED_MIN     = 60        # ranges suggest by Geppetto: 40-80
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
