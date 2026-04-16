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

def build_dataset(dicom_folder_path):
    """
    Builds X (volumes) and y (labels) from DICOM folder.

    Returns:
        X: numpy array (N, 128, 128, 64, 1)
        y: numpy array (N,)
    """

    dicom_files_grouped = group_dicom_files(dicom_folder_path)

    volumes = []
    labels = []

    # map subtype → numeric label
    subtype_to_label = {
        subtype: idx for idx, subtype in enumerate(dicom_files_grouped.keys())
    }

    for subtype, patients in dicom_files_grouped.items():
        for patient, files in patients.items():

            # skip empty cases (safety)
            if len(files) == 0:
                continue

            volume = load_patient_volume(files)
            volume = z_normalize(volume)
            volume = resize_volume(volume)

            # add channel dim
            volume = np.expand_dims(volume, axis=-1) # inserting a new dimension of size 1, and will be added at the end (axis=-1)

            volumes.append(volume)
            labels.append(subtype_to_label[subtype])

    X = np.array(volumes)
    y = np.array(labels)

    return X, y, subtype_to_label
