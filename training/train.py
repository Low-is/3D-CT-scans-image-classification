import numpy as np
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# As long as Python knows the root folder '3D-CT-scans-image-classification/...', other scripts and their functions can be accessed
from models.cnn_3d import get_model

# ------------------------------------------------------------------
# ASSUMES YOU ALREADY HAVE THESE FUNCTIONS IN YOUR PROJECT
# ------------------------------------------------------------------
from preprocessing import load_patient_volume, z_normalize, resize_volume
from dataset import dicom_files_grouped  # your dictionary structure
# ------------------------------------------------------------------


def build_dataset():
    volumes = []
    labels = []

    subtype_to_label = {
        subtype: idx for idx, subtype in enumerate(dicom_files_grouped.keys())
    }

    for subtype, patients in dicom_files_grouped.items():
        for patient, files in patients.items():

            volume = load_patient_volume(files)
            volume = z_normalize(volume)
            volume = resize_volume(volume)

            volume = np.expand_dims(volume, axis=-1)  # (H, W, D, 1)

            volumes.append(volume)
            labels.append(subtype_to_label[subtype])

    x = np.array(volumes)
    y = np.array(labels)

    return x, y


def main():

    # -----------------------
    # Build dataset
    # -----------------------
    x, y = build_dataset()

    # -----------------------
    # Train/test split
    # -----------------------
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    num_classes = y_train_cat.shape[1]

    # -----------------------
    # Class weights
    # -----------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights_dict = dict(enumerate(class_weights))

    # -----------------------
    # Build model
    # -----------------------
    model = get_model(
        width=128,
        height=128,
        depth=64,
        num_classes=num_classes
    )

    # -----------------------
    # Compile model
    # -----------------------
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # -----------------------
    # Callbacks
    # -----------------------
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3dcnn_best.keras",
        save_best_only=True,
        monitor="val_accuracy"
    )

    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        mode="max"
    )

    # -----------------------
    # Train
    # -----------------------
    history = model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_test, y_test_cat),
        epochs=50,
        batch_size=2,
        shuffle=True,
        class_weight=class_weights_dict,
        verbose=2,
        callbacks=[checkpoint_cb, early_stop_cb]
    )

    # -----------------------
    # Save final model
    # -----------------------
    model.save("3dcnn_final.keras")


if __name__ == "__main__":
    main()
