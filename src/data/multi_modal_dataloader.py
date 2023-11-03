import torch
from torch.utils.data import DataLoader
from data.multi_modal_dataset import MultiModalDataset

def collate_fn(batch):
    return {
        'panels': torch.stack([x[0] for x in batch]),
        'texts': [x[1] for x in batch]
    }

def create_dataloader(dataloader_option, args, batch_size=64, workers=5):
    index_file = args.dataset_config["data"][f"{dataloader_option}_index_file"] # train, val, test index files
    dataset = MultiModalDataset(args, index_file)
    batch_size = min(batch_size, len(dataset))

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(dataloader_option == "train"), num_workers=workers, collate_fn=collate_fn
    )

    return dataloader
