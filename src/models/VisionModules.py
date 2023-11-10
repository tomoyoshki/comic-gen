import torch
import torch.nn as nn

from transformers import ViTModel, ViTImageProcessor

class VisionEncoder(nn.Module):
    def __init__(self, args):
        super(VisionEncoder, self).__init__()

        self.args = args
        self.config = args.dataset_config["VisionEncoder"]
        
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", do_rescale=False)
        self.vision_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def forward(self, panels):
        '''
            panels: (batch_size, seq_length, 3, 256, 256)
        '''
        merged_batch = panels.view(-1, 3, 256, 256)
        
        inputs = self.image_processor(merged_batch, return_tensors="pt")
        outputs = self.vision_model(**inputs.to("cuda"))
        
        embeddings = outputs.last_hidden_state
        cls_embeddings = embeddings[:, 0, :]
        
        return cls_embeddings.view(panels.shape[0], panels.shape[1], -1)