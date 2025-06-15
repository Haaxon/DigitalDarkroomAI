from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Dropout, MaxPooling2D, GlobalAveragePooling2D, Dense, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow import keras
import tensorflow as tf

# Enable GPU memory growth
if tf.config.experimental.list_physical_devices('GPU'):
    try:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Constants
LOSS_METHOD     = "mse"         # mse:Mean Squared Error, mae:Mean Average Error
ACTIVATIONMETHOD= 'relu'
INPUTSHAPE      = (150,150,3)   # Image size, and number of channels
BASEFILTERS     = 32
DROPOUTRATE     = 0.3
INITIALIZER     = 'he_normal'
OUTPUTUNITS     = 9             # How many output values
BATCH_SIZE      = 64
EPOCH_SIZE      = 100
CHANNEL_COUNT   = 3             # 3 color channels, no transparency
SOURCE_DIR      = "../dataset/source/"
TARGET_DIR      = "../dataset/target/"
TEST_DIR        = "../Images/21229.png"
VAL_PERCENT     = 0.2           # Percentage of dataset used for validation
IMAGES_TO_LOG   = 15


datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=VAL_PERCENT)

source_train = datagen.flow_from_directory(
    SOURCE_DIR,
    target_size=INPUTSHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode=None,        # No labels for input images
    shuffle=False,           # Maintain order
    subset='training'
)

source_val = datagen.flow_from_directory(
    SOURCE_DIR,
    target_size=INPUTSHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode=None,        # No labels for input images
    shuffle=False,           # Maintain order
    subset='validation'
)

target_train = datagen.flow_from_directory(
    TARGET_DIR,
    target_size=INPUTSHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode=None,        # No labels for target images
    shuffle=False,           # Maintain order
    subset='training'
)

target_val = datagen.flow_from_directory(
    TARGET_DIR,
    target_size=INPUTSHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode=None,        # No labels for target images
    shuffle=False,           # Maintain order
    subset='validation'
)

# Combine the generators into a single generator
train_generator = (
    (X_batch, Y_batch) for X_batch, Y_batch in zip(source_train, target_train)
)

val_generator = (
    (X_batch, Y_batch) for X_batch, Y_batch in zip(source_val, target_val)
)

# Custom layer
@register_keras_serializable()

# class applyParameters(tf.keras.layers.Layer):
#     def invert_image(image):
#         inverted = 255 - image
#         return inverted

#     def channel_exposure_adjust(image, red_min, red_max, green_min, green_max, blue_min, blue_max):

#         def remap_channel(channel, minimum, maximum):
#             channel = (channel - minimum) / (maximum - minimum) * 255.0
#             return np.clip(channel, 0, 255)
    
#         output_image(redchannel) = remap_channel(image_float[..., 0], b_min, b_max)
#         output_image(bluechannel) = remap_channel(image_float[..., 1], g_min, g_max)
#         output_image(greenchannel) = remap_channel(image_float[..., 2], r_min, r_max)

#         return output_image
    
#     def color_balance(image, cyan_red, magenta_green, yellow_blue):
#         out_image = image

#         out_image(redchannel) += cyan_red
#         out_image(bluechannel) += magenta_green
#         out_image(greenchannel) += yellow_blue

#         return out_image

        

#     def call(self, inputs):
#         image, parameters = inputs  # image: (batch, 256, 256, 3), matrix: (?)
#         # Take 9 parameters as input
#         # Also take the original image as input
#         # apply the parametrs to the image

#         finalImage = invert_image(image)
#         finalImage = channel_exposure_adjust(finalImage, parameters[0], parameters[1], parameters[3], parameters[4], parameters[5])
#         finalImage = color_balance(finalImage, parameters[6], parameters[7], parameters[8])
#         return finalImage

class ApplyParameters(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



    def call(self, inputs):
        input_image, parameters = inputs

        # Ensure image is in range [0, 1]
        with tf.control_dependencies([
            tf.debugging.assert_greater_equal(input_image, 0.0, message="Image must be >= 0"),
            tf.debugging.assert_less_equal(input_image, 1.0, message="Image must be <= 1")
        ]):
            input_image = tf.identity(input_image)

        def expand(tensor):
            # tensor shape: (batch,)
            # expand to (batch, 1, 1) for broadcasting over image spatial dims
            return tf.expand_dims(tf.expand_dims(tensor, axis=1), axis=2)
        
        red_scale      = expand(parameters[:, 0])
        red_offset     = expand(parameters[:, 1])
        green_scale    = expand(parameters[:, 2])
        green_offset   = expand(parameters[:, 3])
        blue_scale     = expand(parameters[:, 4])
        blue_offset    = expand(parameters[:, 5])
        cyan_red       = expand(parameters[:, 6])
        magenta_green  = expand(parameters[:, 7])
        yellow_blue    = expand(parameters[:, 8])

        # -----------------------------
        # Invert the image
        inverted = 1.0 - input_image #! please do not use 'x', it's hard to read what this is

        # Extract individual channels
        r = (inverted[..., 0])
        g = (inverted[..., 1])
        b = (inverted[..., 2])

        # Rescale channel (exposure adjustment)
        r = r * red_scale + red_offset
        g = g * green_scale + green_offset
        b = b * blue_scale + blue_offset

        # Clip to [0, 1]
        r = tf.clip_by_value(r, 0.0, 1.0)
        g = tf.clip_by_value(g, 0.0, 1.0)
        b = tf.clip_by_value(b, 0.0, 1.0)

        # Color balance adjustment
        r = tf.clip_by_value(r + cyan_red, 0.0, 1.0)
        g = tf.clip_by_value(g + magenta_green, 0.0, 1.0)
        b = tf.clip_by_value(b + yellow_blue, 0.0, 1.0)

        # Final image
        return tf.stack([r, g, b], axis=-1)

# Define input layer
input_image = keras.Input(shape=INPUTSHAPE, name="image_input")

# Encoder block 1
x = Conv2D(BASEFILTERS, (3, 3), activation=ACTIVATIONMETHOD, kernel_initializer=INITIALIZER, padding='same')(input_image)
x = BatchNormalization()(x)
x = Dropout(DROPOUTRATE)(x)
x = Conv2D(BASEFILTERS, (3, 3), activation=ACTIVATIONMETHOD, kernel_initializer=INITIALIZER, padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Encoder block 2
x = Conv2D(BASEFILTERS * 2, (3, 3), activation=ACTIVATIONMETHOD, kernel_initializer=INITIALIZER, padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUTRATE)(x)
x = Conv2D(BASEFILTERS * 2, (3, 3), activation=ACTIVATIONMETHOD, kernel_initializer=INITIALIZER, padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Encoder block 3
x = Conv2D(BASEFILTERS * 4, (3, 3), activation=ACTIVATIONMETHOD, kernel_initializer=INITIALIZER, padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUTRATE)(x)
x = Conv2D(BASEFILTERS * 4, (3, 3), activation=ACTIVATIONMETHOD, kernel_initializer=INITIALIZER, padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Encoder block 4
x = Conv2D(BASEFILTERS * 8, (3, 3), activation=ACTIVATIONMETHOD, kernel_initializer=INITIALIZER, padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUTRATE)(x)
x = Conv2D(BASEFILTERS * 8, (3, 3), activation=ACTIVATIONMETHOD, kernel_initializer=INITIALIZER, padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Bottleneck
x = Conv2D(BASEFILTERS * 16, (3, 3), activation=ACTIVATIONMETHOD, kernel_initializer=INITIALIZER, padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUTRATE)(x)
x = Conv2D(BASEFILTERS * 16, (3, 3), activation=ACTIVATIONMETHOD, kernel_initializer=INITIALIZER, padding='same')(x)
x = BatchNormalization()(x)

# Global pooling + dense projection
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation=ACTIVATIONMETHOD)(x)
x = Dense(256, activation=ACTIVATIONMETHOD)(x)
parameters = Dense(9, activation='sigmoid', name='parameters')(x)  # Final layer with 9 values in [0, 1]
output_image = ApplyParameters()([input_image,parameters])


model = tf.keras.Model(inputs=input_image, outputs=output_image)

# Compiling the model
model.compile(optimizer=Adam(), loss=LOSS_METHOD)

# Stop training if model does not improve.
callback = EarlyStopping(
    verbose     = 1,
    monitor     = "loss",
    mode        = "min",
    patience    = 20,
    min_delta   = 0.0001,
    restore_best_weights    = True,
    start_from_epoch        = 20
)

# Logging functionality
# To host server, run following command in terminal:
#   tensorboard --logdir=logs/fit
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

class ImageLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, sample_images, sample_targets, freq=1, num_images_to_log=3):
        """
        Logs input, predicted, and target images to TensorBoard during training.

        Args:
            log_dir (str): Base directory where logs will be written.
            sample_images (np.array): Input images to log predictions for.
            sample_targets (np.array): Ground-truth target images.
            freq (int): Frequency (in epochs) at which to log images.
            num_images_to_log (int): Number of images to log each time.
        """
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir + '/images')
        self.sample_images = sample_images[:num_images_to_log]
        self.sample_targets = sample_targets[:num_images_to_log]
        self.freq = freq
        self.num_images_to_log = num_images_to_log

    def on_epoch_end(self, epoch, logs=None):
        # Only log images every 'freq' epochs
        if epoch % self.freq != 0:
            return

        # Run predictions on the cached validation sample
        pred_images = self.model.predict(self.sample_images)

        # Clip predicted images to [0, 1] for valid display
        pred_images = np.clip(pred_images, 0.0, 1.0)

        # Write images to TensorBoard
        with self.file_writer.as_default():
            tf.summary.image("Input", self.sample_images, step=epoch, max_outputs=self.num_images_to_log)
            tf.summary.image("Output", pred_images, step=epoch, max_outputs=self.num_images_to_log)
            tf.summary.image("Target", self.sample_targets, step=epoch, max_outputs=self.num_images_to_log)


# Fetch and store sample validation images before training
sample_val_images, sample_val_targets = next(zip(source_val, target_val))

# Create the image logging callback with the correct arguments
image_logger = ImageLoggerCallback(
    log_dir=log_dir,
    sample_images=sample_val_images,
    sample_targets=sample_val_targets,
    freq=1,  # Log every epoch
    num_images_to_log=IMAGES_TO_LOG
)

# Print information about the model
model.summary()

#sys.exit()

# Train the model
model.fit(
    train_generator, 
    steps_per_epoch     = source_train.samples // BATCH_SIZE,
    validation_data     = val_generator,
    validation_steps    = source_val.samples // BATCH_SIZE,
    epochs              = EPOCH_SIZE,
    callbacks           = [tensorboard_callback, image_logger]
)

# Print information about the model
model.summary()

# Save the trained model
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
model.save(f"trainedModels/trained_model_{current_time}.keras")
print(f"Model saved to trainedModels/trained_model_{current_time}.keras")

# --- LOAD AND PREPROCESS A TEST IMAGE ---
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img = load_img(TEST_DIR, target_size=INPUTSHAPE[:2])  # Resize to match input
img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension → (1, 256, 256, 3)

# --- PREDICT ---
output_array = model.predict(img_array)  # shape: (1, 256, 256, 3)
output_image = output_array[0]  # Remove batch dimension

# --- DISPLAY INPUT AND OUTPUT ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array[0])
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(np.clip(output_image, 0, 1))  # Ensure values are in displayable range
plt.title("Transformed Output")
plt.axis("off")

plt.show()


