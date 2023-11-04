import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(self, args, index_file, transform=None):
        self.args = args
        
        if not os.path.exists(index_file):
            raise NotImplementedError(f"Index file {index_file} not found. Please make sure that you change the index file path in src/data/config.yaml file")
        self.sample_files = list(np.loadtxt(index_file, dtype=str))
        self.transform = transform
        
        
        self.local_directory = "/Users/tomoyoshikimura/Documents/fa23/cs546/comic-gen/data/sample"
    
    def __replace_path(self, path):
        path = path[path.find("sample"):]
        path = path.replace("sample", self.local_directory)
        # sanity check windows back slashes on mac and linux
        path = path.replace("\\", "/")
        return path

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        image_filepath = self.sample_files[idx]
        image_filepath = self.__replace_path(image_filepath)
        
        if not os.path.exists('/'.join(image_filepath.split('/')[:-1])):
            print(image_filepath)
            raise Exception("Ensure that you change the local directory in this file to your local directory.")

        image_folder_path = f"{'/'.join(image_filepath.split('/')[:-1])}"
        panels = torch.load(f"{image_folder_path}/images.pt")
        text_path = f"{image_folder_path}/text_tokens.pt"
        text = torch.load(text_path)
        return panels, text