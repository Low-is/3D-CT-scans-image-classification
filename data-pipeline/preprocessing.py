import numpy as np
import pydicom
from skimage.transform import resize

#############################
# Loading DICOM -> 3D volume
#############################
def load_patient_volume(dicom_file_list):
    slices = [pydicom.dcmread(f) for f in dicom_file_list]

    # sort by slice order
    slices.sort(key=lambda x: int(x.InstanceNumber))

    # lambda x: is the start of an anonymous function

    # stack into 3D volume (D, H, W)
    volume = np.stack([s.pixel_array for s in slices], axis=0)

    return volume


#############################
# Z-score normalization
#############################
def z_normalize(volume):
    mean = volume.mean()
    std = volume.std()

    if std == 0:
        return volume - mean

    return (volume - mean) / std


#############################
# Depth cropping (optional)
#############################
def crop_depth(volume, target_depth=64):
    current_depth = volume.shape[0]

    if current_depth > target_depth:
        start = (current_depth - target_depth) // 2
        return volume[start:start + target_depth, :, :]

    return volume


#############################
# Depth padding (optional)
#############################
def pad_depth(volume, target_depth=64):
    current_depth = volume.shape[0]

    if current_depth < target_depth:
        pad_before = (target_depth - current_depth) // 2
        pad_after = target_depth - current_depth - pad_before

        volume = np.pad(
            volume,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode="constant"
        )

    return volume


#############################
# Resize volume (for CNN input)
#############################
def resize_volume(volume, target_shape=(128, 128, 64)):
    volume = resize(
        volume,
        target_shape,
        mode="constant",
        preserve_range=True,
        anti_aliasing=True
    )

    return volume.astype(np.float32)
