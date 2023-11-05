import torch

def load_model_weight(args, model):
    
    if args.stage in {"encode", "decode"}:
        weight_file = f"{args.weight_folder}/model_weights/{args.framework}_encode_best.pt"
    elif args.stage in {"generate"}:
        weight_file = f"{args.weight_folder}/model_weights/{args.framework}_decode_best.pt"

    trained_dict = torch.load(weight_file, map_location=args.device)
    model_dict = model.state_dict()    
    load_dict = {k: v for k, v in trained_dict.items() if k in model_dict}
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)

    return model