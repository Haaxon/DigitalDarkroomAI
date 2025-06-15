from datetime import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt

"""
================================================================================
Color Correction Matrix Model Generator
================================================================================

This script defines and trains a deep learning model for learning a color 
correction matrix (CCM) that transforms input images to match target images. 
It is designed for use in digital darkroom or image color correction tasks.

--------------------------------------------------------------------------------
Usage Notes:
--------------------------------------------------------------------------------
- Set `SOURCE_DIR` and `TARGET_DIR` to the directories containing your source 
  and target images, respectively.
- To monitor training progress and image outputs, run TensorBoard with:
        tensorboard --logdir=logs/fit
- The model expects images of shape (150, 150, 3) and outputs images of the 
  same shape after applying the learned color correction matrix.

================================================================================
"""

# Enable GPU memory growth
if tf.config.experimental.list_physical_devices('GPU'):
    try:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Constants
LOSS_METHOD     = "mse"         # mse:Mean Squared Error, mae:Mean Average Error
ACTIVATIONMETHOD= 'tanh'
INPUTSHAPE      = (150,150,3)   # Image size, and number of channels
OUTPUTUNITS     = 9             # How many output values
BATCH_SIZE      = 128
EPOCH_SIZE      = 30
INPUT_SIZE      = (150,150)     
CHANNEL_COUNT   = 3             # 3 color channels, no transparency
SOURCE_DIR      = ""
TARGET_DIR      = ""
VAL_PERCENT     = 0.2           # Percentage of dataset used for validation
IMAGES_TO_LOG   = 15


datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=VAL_PERCENT)

source_train = datagen.flow_from_directory(
    SOURCE_DIR,
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,        # No labels for input images
    shuffle=False,           # Maintain order
    subset='training'
)

source_val = datagen.flow_from_directory(
    SOURCE_DIR,
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,        # No labels for input images
    shuffle=False,           # Maintain order
    subset='validation'
)

target_train = datagen.flow_from_directory(
    TARGET_DIR,
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,        # No labels for target images
    shuffle=False,           # Maintain order
    subset='training'
)

target_val = datagen.flow_from_directory(
    TARGET_DIR,
    target_size=INPUT_SIZE,
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
def scale_to_range(x):
    return x * 2.0

class MatrixMultiply(tf.keras.layers.Layer):
    def call(self, inputs):
        image, matrix = inputs  # image: (batch, 256, 256, 3), matrix: (batch, 3, 3, 1)
        return tf.einsum('bijc,bcd->bijd', image, matrix)

input_image = tf.keras.Input(shape=INPUTSHAPE, name="image_input")
x = Conv2D(32,(3,3), activation=ACTIVATIONMETHOD, input_shape=INPUTSHAPE)(input_image)
x = GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='tanh')(x)
x = tf.keras.layers.Dense(1024, activation='tanh')(x)
x = tf.keras.layers.Dense(9, activation='tanh')(x)
x = tf.keras.layers.Reshape((3, 3))(x)
x = tf.keras.layers.Lambda(scale_to_range)(x)           # Generated color correction matrix
output = MatrixMultiply()([input_image, x])             # Multiply matrix with input image

model = tf.keras.Model(inputs=input_image, outputs=output)

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

test_image_path = "../dataset/source/20057.png"
img = load_img(test_image_path, target_size=INPUT_SIZE)  # Resize to match input
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


