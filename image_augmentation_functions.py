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
        start = (current_depth - target_depth) // 2
        return volume[:, :, start:start + target_depth]
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
