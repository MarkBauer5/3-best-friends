import timm
import torch.nn as nn
from blocks import *

# Put predefined model architectures here for training or evaluation

# 92% accuracy after full training
# 28M params
swinModel = nn.Sequential(
    timm.create_model('swin_tiny_patch4_window7_224', pretrained=False),
    nn.Linear(in_features=1000, out_features=2)
)


# 91.X? I think on test set? I forgot
# 86M parameters
vitModel = nn.Sequential(
    timm.create_model('vit_base_patch16_224', pretrained=False),
    nn.Linear(in_features=1000, out_features=2)
)

# First CNN I pretty much copy pasted from my CS 444 project
# NOT TESTED
# 49M Params
testCNN1 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlockNorm(in_channels=32, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualBlock(channelCount=32, activation=nn.PReLU()) for _ in range(3)],
            )
        ) for _ in range(3)]
    ], averageChannels=True),
    
    # Expand channel count
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BranchBlockNorm(in_channels=64, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualBlock(channelCount=64, activation=nn.PReLU()) for _ in range(3)],
            )
        ) for _ in range (3)]
    ], averageChannels=True),
    
    # Expand channels again
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlockNorm(in_channels=128, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualBlock(channelCount=128, activation=nn.PReLU()) for _ in range(3)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BranchBlockNorm(in_channels=256, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualBlock(channelCount=256, activation=nn.PReLU()) for _ in range(3)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BranchBlockNorm(in_channels=256, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualBlock(channelCount=256, activation=nn.PReLU()) for _ in range(3)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),
    
    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),

    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.PReLU(),
    
    nn.Linear(in_features=256, out_features=2)
)

# Implements DWS convolutions for a theoretical 9x speedup
# 31M Params
DWSConvNet1 = nn.Sequential(
    DepthwiseSeparableConv2d(in_channels=3, out_channels=33, kernel_size=5, stride=2, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BranchBlockNorm(in_channels=33, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=33) for _ in range(3)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),
    
    
    # DepthwiseSeparableConv2d(in_channels=32, out_channels=64, kernel_size=3),
    nn.Conv2d(in_channels=33, out_channels=64, kernel_size=3),
    nn.BatchNorm2d(num_features=64),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlockNorm(in_channels=64, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=64) for _ in range(3)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),
    
    
    DepthwiseSeparableConv2d(in_channels=64, out_channels=128, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlockNorm(in_channels=128, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=128) for _ in range(3)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),
    

    DepthwiseSeparableConv2d(in_channels=128, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlockNorm(in_channels=256, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),


    DepthwiseSeparableConv2d(in_channels=256, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlockNorm(in_channels=256, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),
    
    
    DepthwiseSeparableConv2d(in_channels=256, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlockNorm(in_channels=256, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),
    
    
    BranchBlockNorm(in_channels=256, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
            )
        ) for _ in range(18)]
    ], averageChannels=True),
    
    BranchBlockNorm(in_channels=256, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
            )
        ) for _ in range(18)]
    ], averageChannels=True),
    
    BranchBlockNorm(in_channels=256, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
            )
        ) for _ in range(18)]
    ], averageChannels=True),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.PReLU(),
    
    nn.Linear(in_features=256, out_features=2)
)