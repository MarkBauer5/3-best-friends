import timm
import torch.nn as nn
from blocks import *
from transformers import ViTForImageClassification

# Put predefined model architectures here for training or evaluation

# 92% accuracy after full training
# 28M params
# swinModel = nn.Sequential(
#     timm.create_model('swin_tiny_patch4_window7_224', pretrained=False),
#     nn.Linear(in_features=1000, out_features=2)
# )

# 92.67% Validation, pretty good
swinModel = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)


# 91.X? I think on test set? I forgot
# 86M parameters
# vitModel = nn.Sequential(
#     timm.create_model('vit_base_patch16_224', pretrained=False),
#     nn.Linear(in_features=1000, out_features=2)
# )

# Only 85.36% Val, 85.11% Test
vitModel = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)

# Gets 86.55% on test set
# TODO: This is pretty sketchy, maybe there's a better way to get this model untrained but I don't know of it.
class VisualizableVIT(nn.Module):
    def __init__(self):
        super().__init__()
        
        _visualizableVIT = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224', return_dict=False)

        # Clear the weights since I have no idea how to get just the architecture without weights
        for layer in _visualizableVIT.children():
            layer.apply(_visualizableVIT._init_weights)
        
        self.vit = _visualizableVIT
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=1000),
            nn.ReLU(),
            
            nn.Linear(1000, 256),
            nn.LayerNorm(normalized_shape=256),
            nn.ReLU(),
            
            nn.Linear(256, 2),
        )
        
    def forward(self, x):
        rawOutput = self.vit(x)[0]
        return self.classifier(rawOutput)






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
# 31M Params, 90.8% accuracy on test set
DWSConvNet1 = nn.Sequential(
    DepthwiseSeparableConv2d(in_channels=3, out_channels=33, kernel_size=5, stride=2, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BranchBlockNorm(in_channels=33, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=33) for _ in range(3)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),
    
    
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

# 87% Test I think
DWSConvNet2_learnedPoolingNoHwy = nn.Sequential(
    DepthwiseSeparableConv2d(in_channels=3, out_channels=18, kernel_size=5, stride=2, padding=1),

    BranchBlockNorm(in_channels=18, branches=[
        *[ResidualDWSeparableConv2d(in_channels=18) for _ in range(6)]
    ], averageChannels=True),
    
    DepthwiseSeparableConv2d(in_channels=18, out_channels=54, kernel_size=3, stride=2, padding=1),

    BranchBlockNorm(in_channels=54, branches=[
        *[ResidualDWSeparableConv2d(in_channels=54) for _ in range(6)]
    ], averageChannels=True),

    DepthwiseSeparableConv2d(in_channels=54, out_channels=108, kernel_size=3, stride=2, padding=1),

    BranchBlockNorm(in_channels=108, branches=[
        *[ResidualDWSeparableConv2d(in_channels=108) for _ in range(12)]
    ], averageChannels=True),

    DepthwiseSeparableConv2d(in_channels=108, out_channels=216, kernel_size=3, stride=2, padding=1),

    BranchBlockNorm(in_channels=216, branches=[
        *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
    ], averageChannels=True),

    DepthwiseSeparableConv2d(in_channels=216, out_channels=432, kernel_size=3, stride=2, padding=1),

    BranchBlockNorm(in_channels=432, branches=[
        *[ResidualDWSeparableConv2d(in_channels=432) for _ in range(36)]
    ], averageChannels=True),
    
    DepthwiseSeparableConv2d(in_channels=432, out_channels=864, kernel_size=3, padding='same'),

    BranchBlockNorm(in_channels=864, branches=[
        *[ResidualDWSeparableConv2d(in_channels=864) for _ in range(36)]
    ], averageChannels=True),
    
    
    nn.AvgPool2d(kernel_size=5, stride=5, padding=0),

    nn.Flatten(),
    
    nn.Linear(in_features=864, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.PReLU(),
    
    nn.Linear(in_features=256, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.PReLU(),
    
    nn.Linear(in_features=256, out_features=2)
)




# 86.27% Test accuracy, kinda bad TBH. Why was the first one so much better?
DWSConvNet3_learnedPoolingNoHwy = nn.Sequential(
    DepthwiseSeparableConv2d(in_channels=3, out_channels=18, kernel_size=5, stride=2, padding=1),

    BranchBlockNorm(in_channels=18, branches=[
        *[ResidualDWSeparableConv2d(in_channels=18) for _ in range(3)]
    ], averageChannels=True),
    
    BranchBlockNorm(in_channels=18, branches=[
        *[ResidualDWSeparableConv2d(in_channels=18) for _ in range(3)]
    ], averageChannels=True),
    
    DepthwiseSeparableConv2d(in_channels=18, out_channels=54, kernel_size=3, stride=2, padding=1),

    BranchBlockNorm(in_channels=54, branches=[
        *[ResidualDWSeparableConv2d(in_channels=54) for _ in range(6)]
    ], averageChannels=True),

    BranchBlockNorm(in_channels=54, branches=[
        *[ResidualDWSeparableConv2d(in_channels=54) for _ in range(6)]
    ], averageChannels=True),

    DepthwiseSeparableConv2d(in_channels=54, out_channels=108, kernel_size=3, stride=2, padding=1),

    BranchBlockNorm(in_channels=108, branches=[
        *[ResidualDWSeparableConv2d(in_channels=108) for _ in range(12)]
    ], averageChannels=True),

    BranchBlockNorm(in_channels=108, branches=[
        *[ResidualDWSeparableConv2d(in_channels=108) for _ in range(12)]
    ], averageChannels=True),

    DepthwiseSeparableConv2d(in_channels=108, out_channels=216, kernel_size=3, stride=2, padding=1),

    BranchBlockNorm(in_channels=216, branches=[
        *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
    ], averageChannels=True),

    DepthwiseSeparableConv2d(in_channels=216, out_channels=216, kernel_size=3, stride=2, padding=1),

    BranchBlockNorm(in_channels=216, branches=[
        *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
    ], averageChannels=True),
    
    BranchBlockNorm(in_channels=216, branches=[
        *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
    ], averageChannels=True),
    
    BranchBlockNorm(in_channels=216, branches=[
        *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
    ], averageChannels=True),
    
    BranchBlockNorm(in_channels=216, branches=[
        *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
    ], averageChannels=True),
    
    BranchBlockNorm(in_channels=216, branches=[
        *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
    ], averageChannels=True),
    
    BranchBlockNorm(in_channels=216, branches=[
        *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
    ], averageChannels=True),
    
    nn.AvgPool2d(kernel_size=5, stride=5, padding=0),

    nn.Flatten(),
    
    nn.Linear(in_features=216, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.PReLU(),
    
    nn.Linear(in_features=256, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.PReLU(),
    
    nn.Linear(in_features=256, out_features=2)
)



# Killed early since this is like 30 minutes/epoch
# 91.01% accuracy on test set
DWSConvNet3_learnedPoolingHwy = nn.Sequential(
    DepthwiseSeparableConv2d(in_channels=3, out_channels=33, kernel_size=5, stride=2, padding=1),

    BranchBlockNorm(in_channels=33, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=33) for _ in range(2)],
            )
        ) for _ in range(3)]
    ], averageChannels=True),
    
    DepthwiseSeparableConv2d(in_channels=33, out_channels=66, kernel_size=3, stride=2, padding=1),
    
    BranchBlockNorm(in_channels=66, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=66) for _ in range(2)],
            )
        ) for _ in range(3)]
    ], averageChannels=True),
    
    DepthwiseSeparableConv2d(in_channels=66, out_channels=132, kernel_size=3, stride=2, padding=1),

    BranchBlockNorm(in_channels=132, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=132) for _ in range(3)],
            )
        ) for _ in range(3)]
    ], averageChannels=True),

    DepthwiseSeparableConv2d(in_channels=132, out_channels=264, kernel_size=3, stride=2, padding=1),

    BranchBlockNorm(in_channels=264, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=264) for _ in range(6)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),

    DepthwiseSeparableConv2d(in_channels=264, out_channels=264, kernel_size=3, stride=2, padding=1),
    
    BranchBlockNorm(in_channels=264, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=264) for _ in range(6)],
            )
        ) for _ in range(6)]
    ], averageChannels=True),
    
    DepthwiseSeparableConv2d(in_channels=264, out_channels=264, kernel_size=3, stride=2, padding=1),
    
    *[BranchBlockNorm(in_channels=264, branches=[
        *[HighwayBlock(highwaySequence=nn.Sequential(
            *[ResidualDWSeparableConv2d(in_channels=264) for _ in range(6)],
            )
        ) for _ in range(6)]
    ], averageChannels=True) for _ in range(18)],
    
    nn.Flatten(),
    
    nn.Linear(in_features=1056, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.PReLU(),
    
    nn.Linear(in_features=256, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.PReLU(),
    
    nn.Linear(in_features=256, out_features=2)
)

# TODO: Try a better CNN lul


def main():
    
    # Test model loading here
    
    customModel = VisualizableVIT()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    customModel.to(device)
    
    dummyInput = torch.randn((4, 3, 224, 224)).to(device)
    
    out = customModel(dummyInput)
    print(out)
    print(f'{out.shape=}')
    
    # print(list(visualizableVIT.children()))
    pass

if __name__ == '__main__':
    main()