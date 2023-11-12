import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from data.multi_modal_dataset import MultiModalDataset

def collate_fn(batch):
    panels = torch.stack([x[0] for x in batch])
    texts = np.array([x[1] for x in batch])
    return panels, texts

def create_dataloader(dataloader_option, args, batch_size=64, workers=5):
    index_file = args.dataset_config["data"][f"{dataloader_option}_index_file"] # train, val, test index files
    index_path = args.dataset_config["data_path"]
    full_index_file = os.path.join(index_path, index_file)
    dataset = MultiModalDataset(args, full_index_file)
    batch_size = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn)

    return dataloader
