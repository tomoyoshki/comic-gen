from torch.utils.data import DataLoader
from data.multi_modal_dataset import MultiModalDataset


def create_dataloader(dataloader_option, args, batch_size=64, workers=5):
    index_file = args.dataset_config["data"][f"{dataloader_option}_index_file"] # train, val, test index files
    dataset = MultiModalDataset(args, index_file) # unsure if we are doing it this way yet
    batch_size = min(batch_size, len(dataset))

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(dataloader_option == "train"), num_workers=workers
    )

    return dataloader
