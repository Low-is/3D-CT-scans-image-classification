import numpy as np

# To access other Python scripts and their functions, make sure both files are in the same folder
from dicom_loader import group_dicom_files
from preprocessing import (
    load_patient_volume,
    z_normalize,
    resize_volume
)

# Frameworks like TensorFlow/Keras expect input like:
# (batch_size, H, W, D, channels)

def build_dataset(config):
    # -----------------------
    # LOAD PATH FROM CONFIG
    # -----------------------
    dataset_path = config["data"]["dataset_path"]
    target_shape = tuple(config["data"]["image_shape"])
    
    dicom_files_grouped = group_dicom_files(dataset_path)

    volumes = []
    labels = []

    subtypes = sorted(dicom_files_grouped.keys())
    subtype_to_label = {s: i for i, s in enumerate(subtypes)}

    for subtype, patients in dicom_files_grouped.items():
        for patient, files in patients.items():
            if len(files) == 0:
                continue

            volume = load_patient_volume(files)

            # -----------------------
            # NORMALIZATION (CONFIG)
            # -----------------------
            if config["preprocessing"]["normalize"] == "z-score":
                volume = z_normalize(volume)

            # -----------------------
            # RESIZE (CONFIG SHAPE)
            # -----------------------
            volume = resize_volume(volume, target_shape)

            volume = np.expand_dims(volume, axis=-1) # Adding channel dim

            volumes.append(volume)
            labels.append(subtype_to_label[subtype])

    return np.array(volumes), np.array(labels)
