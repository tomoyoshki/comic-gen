import os
import json
import logging

def check_paths(path_list):
    for p in path_list:
        if not os.path.exists(p):
            os.mkdir(p)

def remove_files(path_list):
    for p in path_list:
        if os.path.exists(p):
            os.remove(p)
            
def set_model_weight_folder(args):
    # get basic paths
    base_path = f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/weights"
    dataset_model_path = os.path.join(base_path, f"{args.framework}")
    check_paths([base_path, dataset_model_path])
    
    # find experiment weight folder under the model folder
    newest_id = -1
    newest_weight = None
    existing_paths = os.listdir(dataset_model_path)

    reversed(existing_paths)
    for weight in existing_paths:
        weight_id = int(weight.split("_")[0][3:])
        if weight_id > newest_id:
            newest_id = weight_id
    

    tag_suffix = "" if args.tag is None else f"_{args.tag}"
    weight_folder = os.path.join(dataset_model_path, f"exp{newest_id + 1}{tag_suffix}")
    check_paths([weight_folder])
    framework_config = args.dataset_config[args.framework]
    with open(os.path.join(weight_folder, "framework_config.json"), "w") as f:
        f.write(json.dumps(framework_config, indent=4))
    
    args.train_log_file = os.path.join(weight_folder, f"pretrain_log.txt")
    remove_files([args.train_log_file])
    
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.FileHandler(args.train_log_file), logging.StreamHandler()], force=True
    )

    logging.info(f"=\t[Model weights path]: {weight_folder}")
    pass