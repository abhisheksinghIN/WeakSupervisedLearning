################################ Working SegFormer ##########################################
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn.functional as F

class SegFormer(nn.Module):
    def __init__(self, num_classes=9, in_channels=3):
        super(SegFormer, self).__init__()
        
        # Create a Segformer configuration with specific number of classes
        config = SegformerConfig(num_labels=num_classes)
        
        # Initialize the Segformer model for semantic segmentation
        self.segformer = SegformerForSemanticSegmentation(config)
        
        # Adjust the input layer to accept the specified number of input channels
        old_conv = self.segformer.segformer.encoder.patch_embeddings[0].proj
        self.segformer.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
    
    def forward(self, x):
        # Forward pass through the Segformer model for semantic segmentation
        outputs = self.segformer(pixel_values=x)
        
        # Extract the logits for semantic segmentation
        logits = outputs.logits
        
        # Upsample the output to match the target size
        x = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)
        return x


#import torch
#import torch.nn as nn
#from transformers import SegformerModel,SegformerForSemanticSegmentation
#import torch.nn.functional as F
#
#class SegFormer(nn.Module):
#    def __init__(self, num_classes=11, pretrained_model_name="nvidia/segformer-b0-finetuned-ade-512-512", in_channels=3):
##    def __init__(self, num_classes=11, pretrained_model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024", in_channels=3):
#        super(SegFormer, self).__init__()
#        self.segformer = SegformerModel.from_pretrained(pretrained_model_name)
#        
#        # Assuming the first embedding layer is in a list:
#        old_conv = self.segformer.encoder.patch_embeddings[0].proj  # Adjust this if the actual layer differs
#        self.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
#            in_channels,
#            old_conv.out_channels,
#            kernel_size=old_conv.kernel_size,
#            stride=old_conv.stride,
#            padding=old_conv.padding
#        )
#        
#        self.segmentation_head = nn.Conv2d(self.segformer.config.hidden_sizes[-1], num_classes, kernel_size=1)  
#    
#    def forward(self, x):
#        outputs = self.segformer(pixel_values=x)
#        x = outputs.last_hidden_state
#        x = self.segmentation_head(x)
#        
#        # Upsample the output to match the target size
#        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
#        return x
       
##model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
##print(model)
