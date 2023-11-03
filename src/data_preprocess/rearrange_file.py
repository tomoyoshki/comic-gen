import os
import random
import shutil
import torch
from torchvision import transforms

import cv2
from transforms.resize_pad import ResizeOrPad

VERBOSE = False
PROCESSED_DIR = "C:/Users/Tomoyoshi/Documents/cs546/processed_data"



transform = transforms.Compose([
    transforms.ToTensor(),
    ResizeOrPad((256, 256))
])
def copy_from_paths(path, target_dir):

    data_t = ResizeOrPad((256, 256))
    for source_image_idx_path in path:
        source_image_idx_path = source_image_idx_path.replace("\n", "")
        image_idx_path = "\\".join(source_image_idx_path.split("\\")[-2:])
        target_image_idx_path = os.path.join(target_dir, image_idx_path)
        target_image_folder_path = f"\\".join(target_image_idx_path.split("\\")[:-1])
        
        if not os.path.exists(target_image_folder_path):
            os.makedirs(target_image_folder_path)
        
        if VERBOSE:
            print(f"Storing images in {target_image_folder_path}")

        for i in range(0, 4):
            source_image_path = f"{source_image_idx_path}_{i}.jpg"
            target_image_path = f"{target_image_idx_path}_{i}.pt"
            
            if VERBOSE:
                print(f"Copying {source_image_path} to {target_image_path}")
            
            panel = cv2.imread(source_image_path)
            panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
            panel = transform(panel)
            torch.save(panel, target_image_path)
    
    target_dir_elem = target_dir.split("/")
    index_path = "/".join(target_dir_elem[:-1])
    index_path = os.path.join(index_path, "index_files")
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    index_path = os.path.join(index_path, f"{target_dir_elem[-1]}.txt")
    with open(index_path, "w") as f:
        for source_image_idx_path in path:
            source_image_idx_path = os.path.join(target_dir, "\\".join(source_image_idx_path.split("\\")[-2:]))
            f.write(f"{source_image_idx_path}")
        

def resample_data(index_file, percent):
    with open(index_paths) as f:
        paths = f.readlines()

    original_paths = paths.copy()
    sampled_paths = paths.copy()

    random.shuffle(sampled_paths)

    N = len(sampled_paths)
    sample_N = int(N * percent)

    print(f"Total samples: {N}")
    print(f"{percent * 100}% is taken: {sample_N}")

    sampled_paths = sampled_paths[:sample_N]

    return sampled_paths

if __name__ == "__main__":
    index_paths = "C:/Users/Tomoyoshi/Documents/cs546/data_preprocess/index.txt"
    sample_percent = 0.0005
    selected_paths = resample_data(index_paths, sample_percent)

    N = len(selected_paths)
    for stage, begin, end in [("train", 0, 0.8), ("val", 0.8, 0.9), ("test", 0.9, 1.0), ("all", 0, 1.0)]:
        stage_paths = selected_paths[int(N * begin):int(N * end)].copy()
        target_dir = os.path.join(PROCESSED_DIR, f"sample/{stage}")
        copy_from_paths(stage_paths, target_dir)


