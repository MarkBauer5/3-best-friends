import timm
import torch.nn as nn

# Put predefined model architectures here for training or evaluation

# Pretty terrible
swinModel = nn.Sequential(
    timm.create_model('swin_tiny_patch4_window7_224', pretrained=False),
    nn.Linear(in_features=1000, out_features=2)
)


# 55% after 3 epochs
vitModel = nn.Sequential(
    timm.create_model('vit_base_patch16_224', pretrained=False),
    nn.Linear(in_features=1000, out_features=2)
)



