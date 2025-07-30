import pydicom
import numpy as np
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from skimage.transform import resize # make sure to have scikit-image installed
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras
from keras import layers

# Group files by subtype and patient
dicom_files_grouped = defaultdict(lambda: defaultdict(list))
dicom_folder_path = r"path\to\ct_scan_series_folder"

for root, dirs, files in os.walk(dicom_folder_path):
    for f in files:
        if f.endswith('.dcm'):
            full_path = os.path.join(root, f)
            parts = full_path.split(os.sep)
            lung_subtype = parts[-3]
            patient = parts[-2]
            dicom_files_grouped[lung_subtype][patient].append(full_path)

# Load and sort slices
def load_patient_volume(dicom_file_list):
    slices = [pydicom.dcmread(f) for f in dicom_file_list]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    volume = np.stack([s.pixel_array for s in slices], axis=0)
    return volume

# Normalize (Z-score)
def z_normalize(volume):
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / std if std != 0 else volume - mean

# Cropping and adding padding to create homogenous volumes
def crop_depth(volume, target_depth=64):
    current_depth = volume.shape[2]
    if current_depth > target_depth:
        start = (current_depth - target_depth)//2
        return volume[:,:,start:start + target_depth]
        return volume

def pad_depth(volume, target_depth=64):
    current_depth = volume.shape[2]
    if current_depth < target_depth:
        pad_before = (target_depth - current_depth) // 2
        pad_after = target_depth - current_depth - pad_before
        volume = np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant')
    return volume

def resize_volume(volume, target_shape=(128, 128, 64)):
    volume = resize(volume, target_shape, mode='constant', preserve_range=True, anti_aliasing=True)
    return volume.astype(np.float32)

# Build volume & label lists
volumes = []
labels = []

subtype_to_label = {subtype: idx for idx, subtype in enumerate(dicom_files_grouped.keys())}

for subtype, patients in dicom_files_grouped.items():
    for patient, files in patients.items():
        volume = load_patient_volume(files)
        volume = z_normalize(volume)
        volume = resize_volume(volume)  # (height, width, depth)
        volume = np.expand_dims(volume, axis=-1)  # Add channel dim: (height, width, depth, 1)
        volumes.append(volume)
        labels.append(subtype_to_label[subtype])

# Convert to numpy arrays
x = np.array(volumes)
y = np.array(labels)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# Choose a sample from x_train
volume = x_train[0]  # shape: (depth, height, width, 1)
volume = np.squeeze(volume)  # shape: (depth, height, width)

# Pick a slice index in the middle of the volume
slice_index = volume.shape[0] // 2

# Visualize
plt.imshow(volume[slice_index], cmap="gray")
plt.title(f"Normalized CT Volume Slice {slice_index}")
plt.axis('off')
plt.show()

# Defining a 3D convolutional neural network
def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()


# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["accuracy"],
    run_eagerly=True,
)


# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.keras", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, mode="max")

# Train the model, doing validation at the end of each epoch
epochs = 100
history = model.fit(
    x_train,
    y_train_cat,
    validation_data=(x_test, y_test_cat),
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)


# Plot accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
