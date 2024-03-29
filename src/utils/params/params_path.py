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
    
    if args.model_weight is not None:
        args.weight_folder = args.model_weight
    else:
        # get basic paths
        base_path = f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/weights"
        dataset_model_path = os.path.join(base_path, f"{args.framework}" if args.framework != "Baseline" else f"{args.framework}_{args.baseline}")
        check_paths([base_path, dataset_model_path])
        
        # find experiment weight folder under the model folder
        newest_id = -1
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
        
        args.weight_folder = weight_folder
    
    args.train_log_file = os.path.join(args.weight_folder, f"pretrain_{args.stage}_log.txt")
    remove_files([args.train_log_file])
    
    model_weight = os.path.join(args.weight_folder, f"model_weights")
    args.model_weight = model_weight
    check_paths([args.model_weight])
    
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.FileHandler(args.train_log_file), logging.StreamHandler()], force=True
    )

    logging.info(f"=[Model weights path]: {args.weight_folder}")
    return args

def set_model_weight_file(args):
    """Automatically select the classifier weight during the testing"""
    args.framework_weight = os.path.join(
        args.model_weight,
        f"{args.framework}_best.pt",
    )

    logging.info(f"=\t[Classifier weight file]: {os.path.basename(args.classifier_weight)}")

    return args