import os
import numpy as np
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(self, args, index_file):
        """
        Args:
            modalities (_type_): The list of modalities
            classes (_type_): The list of classes
            index_file (_type_): The list of sample file names
            sample_path (_type_): The base sample path.

        Sample:
            - label: Tensor
            - flag
                - phone
                    - audio: True
                    - acc: False
            - data:
                -phone
                    - audio: Tensor
                    - acc: Tensor
        """
        self.args = args
        
        if not os.path.exists(index_file):
            raise NotImplementedError(f"Index file {index_file} not found. MultiModalDataset not implemented yet.")
        self.sample_files = list(np.loadtxt(index_file, dtype=str))

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        raise NotImplementedError("Dataloader implemented yet.")