from torch.utils.data import Dataset
import pandas as pd
from src.utils.data_utils import load_mri, load_segmap
from scipy.ndimage import gaussian_filter, binary_erosion
import torch

class MRIDatasetFromDF(Dataset):
    def __init__(self, df: pd.DataFrame, atlas_loc: int | None = None):
        self.df = df
        self.atlas_index = atlas_loc

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['data_path']
        diagnosis = row['diagnosis']
        img = load_mri(img_path)
        if self.atlas_index:
            seg_path = row['seg_path']
            atlas = load_segmap(seg_path)
            atlas_mask = (atlas == self.atlas_index)
            region_mean = np.mean(scan[:, atlas_mask])
            inner_core_mask = binary_erosion(atlas_mask, iterations=3)
            img[:, inner_core_mask] = region_mean
            blurred_img = gaussian_filter(img, sigma=2.0)
            img[:, atlas_mask] = blurred_img[:, atlas_mask]
        diagnosis = row['diagnosis']
        return torch.tensor(img, dtype=torch.float32), torch.tensor(diagnosis, dtype=torch.long)
