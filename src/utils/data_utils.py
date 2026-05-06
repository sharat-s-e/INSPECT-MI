import numpy as np
import nibabel as nib

def zscore_normalization(image):
    mean = np.mean(image)
    std = np.std(image)
    normalized_img = (image - mean) / std if std != 0 else image - mean
    return normalized_img

def load_mri(img_path, transform=None):
    img = nib.load(img_path).get_fdata().astype(np.float32)
    img = zscore_normalization(img)
    if transform:
        img = transform(img)
    img = np.expand_dims(img, axis = 0)
    return img

def load_segmap(img_path):
    img = nib.load(img_path).get_fdata().astype(int)
    return img