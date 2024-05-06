import timm
import torch.nn as nn
from blocks import *
from transformers import ViTForImageClassification
from swin_transformer_v2 import SwinTransformerV2
from modelUtils import profileModel
import torchvision.transforms.v2 as v2

# Put predefined model architectures here for training or evaluation

# 92% accuracy after full training
# 28M params
# swinModel = nn.Sequential(
#     timm.create_model('swin_tiny_patch4_window7_224', pretrained=False),
#     nn.Linear(in_features=1000, out_features=2)
# )

# 92.67% Validation, pretty good
# swinModel = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)


# 91.X? I think on test set? I forgot
# 86M parameters
# vitModel = nn.Sequential(
#     timm.create_model('vit_base_patch16_224', pretrained=False),
#     nn.Linear(in_features=1000, out_features=2)
# )

# Only 85.36% Val, 85.11% Test
# vitModel = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)

# Gets 86.55% on test set
# TODO: This is pretty sketchy, maybe there's a better way to get this model untrained but I don't know of it.
# 94.942 Train, 85.945 Val, 86.257 Test 7367s training
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


# 93.73% Test 94.06% Validation
# 92.206 Train, 94.515 Val 94.485 Test 10096s training
# 97.84 Train 94.955 Val, 96.215 Test Took 7367s to train 
class VisualizableSWIN(nn.Module):
    
    def __init__(self):
        
        super(VisualizableSWIN, self).__init__()
        
        class Args:
            def __init__(self) -> None:
                # data arguments
                self.num_classes = 2
                self.img_size = 256
                self.num_train_data = 10000
                self.num_test_data = 2000
                self.dataset_path = "datasets/140k Real vs Fake/real_vs_fake/real-vs-fake/"

                # training arguments
                self.learning_rate =  1e-4
                self.epochs = 10
                self.scheduler = True
                self.sch_step_size = 2
                self.sch_gamma = 0.1

                # model arguments
                self.drop_path_rate = 0.2
                self.embed_dim = 96
                self.depths = (2, 2, 6, 2)
                self.num_heads = (3, 6, 12, 24)
                self.window_size = 16
                self.load_model_path = "CollectedData/Models/swinv2_tiny_patch4_window16_256.pth"
                self.save_model_path = "CollectedData/Models/swinv2_tiny_patch4_window16_256.pth"

                # output arguments
                self.output_path = "../output/"

        args = Args()

        _visualizableSWIN = SwinTransformerV2(img_size=args.img_size,
                                drop_path_rate=args.drop_path_rate,
                                embed_dim=args.embed_dim,
                                depths=args.depths,
                                num_heads=args.num_heads,
                                window_size=args.window_size)
        state_dict = torch.load(args.load_model_path)
        _visualizableSWIN.load_state_dict(state_dict["model"])
        _visualizableSWIN.head = torch.nn.Linear(_visualizableSWIN.head.in_features, args.num_classes)
        
        # Reset model parameters to train from scratch
        for layer in _visualizableSWIN.children():
            layer.apply(_visualizableSWIN._init_weights)
            
        self.model = _visualizableSWIN
        
    def forward(self, x):
        return self.model(x)


# First CNN I pretty much copy pasted from my CS 444 project
# NOT TESTED
# 49M Params
# testCNN1 = nn.Sequential(
#     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
#     nn.BatchNorm2d(num_features=32),
#     nn.PReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
    
#     BranchBlockNorm(in_channels=32, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualBlock(channelCount=32, activation=nn.PReLU()) for _ in range(3)],
#             )
#         ) for _ in range(3)]
#     ], averageChannels=True),
    
#     # Expand channel count
#     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
#     nn.BatchNorm2d(num_features=64),
#     nn.PReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),

#     BranchBlockNorm(in_channels=64, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualBlock(channelCount=64, activation=nn.PReLU()) for _ in range(3)],
#             )
#         ) for _ in range (3)]
#     ], averageChannels=True),
    
#     # Expand channels again
#     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#     nn.BatchNorm2d(num_features=128),
#     nn.PReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
    
#     BranchBlockNorm(in_channels=128, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualBlock(channelCount=128, activation=nn.PReLU()) for _ in range(3)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True),
    
#     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#     nn.BatchNorm2d(num_features=256),
#     nn.PReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),

#     BranchBlockNorm(in_channels=256, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualBlock(channelCount=256, activation=nn.PReLU()) for _ in range(3)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True),
    
#     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#     nn.BatchNorm2d(num_features=256),
#     nn.PReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),

#     BranchBlockNorm(in_channels=256, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualBlock(channelCount=256, activation=nn.PReLU()) for _ in range(3)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True),
    
#     nn.AvgPool2d(kernel_size=4, stride=4, padding=0),

#     nn.Flatten(),
    
#     nn.Linear(in_features=256, out_features=256),
#     nn.LayerNorm(normalized_shape=256),
#     nn.PReLU(),
    
#     nn.Linear(in_features=256, out_features=2)
# )

# Implements DWS convolutions for a theoretical 9x speedup
# 31M Params, 90.8% accuracy on test set
# DWSConvNet1 = nn.Sequential(
#     DepthwiseSeparableConv2d(in_channels=3, out_channels=33, kernel_size=5, stride=2, padding=1),
#     nn.MaxPool2d(kernel_size=2, stride=2),

#     BranchBlockNorm(in_channels=33, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=33) for _ in range(3)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True),
    
    
#     nn.Conv2d(in_channels=33, out_channels=64, kernel_size=3),
#     nn.BatchNorm2d(num_features=64),
#     nn.PReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
    
#     BranchBlockNorm(in_channels=64, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=64) for _ in range(3)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True),
    
    
#     DepthwiseSeparableConv2d(in_channels=64, out_channels=128, kernel_size=3),
#     nn.MaxPool2d(kernel_size=2, stride=2),
    
#     BranchBlockNorm(in_channels=128, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=128) for _ in range(3)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True),
    

#     DepthwiseSeparableConv2d(in_channels=128, out_channels=256, kernel_size=3),
#     nn.MaxPool2d(kernel_size=2, stride=2),
    
#     BranchBlockNorm(in_channels=256, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True),


#     DepthwiseSeparableConv2d(in_channels=256, out_channels=256, kernel_size=3),
#     nn.MaxPool2d(kernel_size=2, stride=2),
    
#     BranchBlockNorm(in_channels=256, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True),
    
    
#     DepthwiseSeparableConv2d(in_channels=256, out_channels=256, kernel_size=3),
#     nn.MaxPool2d(kernel_size=2, stride=2),
    
#     BranchBlockNorm(in_channels=256, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True),
    
    
#     BranchBlockNorm(in_channels=256, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
#             )
#         ) for _ in range(18)]
#     ], averageChannels=True),
    
#     BranchBlockNorm(in_channels=256, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
#             )
#         ) for _ in range(18)]
#     ], averageChannels=True),
    
#     BranchBlockNorm(in_channels=256, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=256) for _ in range(3)],
#             )
#         ) for _ in range(18)]
#     ], averageChannels=True),
    
#     nn.Flatten(),
    
#     nn.Linear(in_features=256, out_features=256),
#     nn.LayerNorm(normalized_shape=256),
#     nn.PReLU(),
    
#     nn.Linear(in_features=256, out_features=2)
# )

# 87% Test I think
# DWSConvNet2_learnedPoolingNoHwy = nn.Sequential(
#     DepthwiseSeparableConv2d(in_channels=3, out_channels=18, kernel_size=5, stride=2, padding=1),

#     BranchBlockNorm(in_channels=18, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=18) for _ in range(6)]
#     ], averageChannels=True),
    
#     DepthwiseSeparableConv2d(in_channels=18, out_channels=54, kernel_size=3, stride=2, padding=1),

#     BranchBlockNorm(in_channels=54, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=54) for _ in range(6)]
#     ], averageChannels=True),

#     DepthwiseSeparableConv2d(in_channels=54, out_channels=108, kernel_size=3, stride=2, padding=1),

#     BranchBlockNorm(in_channels=108, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=108) for _ in range(12)]
#     ], averageChannels=True),

#     DepthwiseSeparableConv2d(in_channels=108, out_channels=216, kernel_size=3, stride=2, padding=1),

#     BranchBlockNorm(in_channels=216, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
#     ], averageChannels=True),

#     DepthwiseSeparableConv2d(in_channels=216, out_channels=432, kernel_size=3, stride=2, padding=1),

#     BranchBlockNorm(in_channels=432, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=432) for _ in range(36)]
#     ], averageChannels=True),
    
#     DepthwiseSeparableConv2d(in_channels=432, out_channels=864, kernel_size=3, padding='same'),

#     BranchBlockNorm(in_channels=864, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=864) for _ in range(36)]
#     ], averageChannels=True),
    
    
#     nn.AvgPool2d(kernel_size=5, stride=5, padding=0),

#     nn.Flatten(),
    
#     nn.Linear(in_features=864, out_features=256),
#     nn.LayerNorm(normalized_shape=256),
#     nn.PReLU(),
    
#     nn.Linear(in_features=256, out_features=256),
#     nn.LayerNorm(normalized_shape=256),
#     nn.PReLU(),
    
#     nn.Linear(in_features=256, out_features=2)
# )




# 86.27% Test accuracy, kinda bad TBH. Why was the first one so much better?
# DWSConvNet3_learnedPoolingNoHwy = nn.Sequential(
#     DepthwiseSeparableConv2d(in_channels=3, out_channels=18, kernel_size=5, stride=2, padding=1),

#     BranchBlockNorm(in_channels=18, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=18) for _ in range(3)]
#     ], averageChannels=True),
    
#     BranchBlockNorm(in_channels=18, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=18) for _ in range(3)]
#     ], averageChannels=True),
    
#     DepthwiseSeparableConv2d(in_channels=18, out_channels=54, kernel_size=3, stride=2, padding=1),

#     BranchBlockNorm(in_channels=54, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=54) for _ in range(6)]
#     ], averageChannels=True),

#     BranchBlockNorm(in_channels=54, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=54) for _ in range(6)]
#     ], averageChannels=True),

#     DepthwiseSeparableConv2d(in_channels=54, out_channels=108, kernel_size=3, stride=2, padding=1),

#     BranchBlockNorm(in_channels=108, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=108) for _ in range(12)]
#     ], averageChannels=True),

#     BranchBlockNorm(in_channels=108, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=108) for _ in range(12)]
#     ], averageChannels=True),

#     DepthwiseSeparableConv2d(in_channels=108, out_channels=216, kernel_size=3, stride=2, padding=1),

#     BranchBlockNorm(in_channels=216, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
#     ], averageChannels=True),

#     DepthwiseSeparableConv2d(in_channels=216, out_channels=216, kernel_size=3, stride=2, padding=1),

#     BranchBlockNorm(in_channels=216, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
#     ], averageChannels=True),
    
#     BranchBlockNorm(in_channels=216, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
#     ], averageChannels=True),
    
#     BranchBlockNorm(in_channels=216, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
#     ], averageChannels=True),
    
#     BranchBlockNorm(in_channels=216, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
#     ], averageChannels=True),
    
#     BranchBlockNorm(in_channels=216, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
#     ], averageChannels=True),
    
#     BranchBlockNorm(in_channels=216, branches=[
#         *[ResidualDWSeparableConv2d(in_channels=216) for _ in range(18)]
#     ], averageChannels=True),
    
#     nn.AvgPool2d(kernel_size=5, stride=5, padding=0),

#     nn.Flatten(),
    
#     nn.Linear(in_features=216, out_features=256),
#     nn.LayerNorm(normalized_shape=256),
#     nn.PReLU(),
    
#     nn.Linear(in_features=256, out_features=256),
#     nn.LayerNorm(normalized_shape=256),
#     nn.PReLU(),
    
#     nn.Linear(in_features=256, out_features=2)
# )



# Killed early since this is like 30 minutes/epoch
# 91.01% accuracy on test set
# DWSConvNet3_learnedPoolingHwy = nn.Sequential(
#     DepthwiseSeparableConv2d(in_channels=3, out_channels=33, kernel_size=5, stride=2, padding=1),

#     BranchBlockNorm(in_channels=33, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=33) for _ in range(2)],
#             )
#         ) for _ in range(3)]
#     ], averageChannels=True),
    
#     DepthwiseSeparableConv2d(in_channels=33, out_channels=66, kernel_size=3, stride=2, padding=1),
    
#     BranchBlockNorm(in_channels=66, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=66) for _ in range(2)],
#             )
#         ) for _ in range(3)]
#     ], averageChannels=True),
    
#     DepthwiseSeparableConv2d(in_channels=66, out_channels=132, kernel_size=3, stride=2, padding=1),

#     BranchBlockNorm(in_channels=132, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=132) for _ in range(3)],
#             )
#         ) for _ in range(3)]
#     ], averageChannels=True),

#     DepthwiseSeparableConv2d(in_channels=132, out_channels=264, kernel_size=3, stride=2, padding=1),

#     BranchBlockNorm(in_channels=264, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=264) for _ in range(6)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True),

#     DepthwiseSeparableConv2d(in_channels=264, out_channels=264, kernel_size=3, stride=2, padding=1),
    
#     BranchBlockNorm(in_channels=264, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=264) for _ in range(6)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True),
    
#     DepthwiseSeparableConv2d(in_channels=264, out_channels=264, kernel_size=3, stride=2, padding=1),
    
#     *[BranchBlockNorm(in_channels=264, branches=[
#         *[HighwayBlock(highwaySequence=nn.Sequential(
#             *[ResidualDWSeparableConv2d(in_channels=264) for _ in range(6)],
#             )
#         ) for _ in range(6)]
#     ], averageChannels=True) for _ in range(18)],
    
#     nn.Flatten(),
    
#     nn.Linear(in_features=1056, out_features=256),
#     nn.LayerNorm(normalized_shape=256),
#     nn.PReLU(),
    
#     nn.Linear(in_features=256, out_features=256),
#     nn.LayerNorm(normalized_shape=256),
#     nn.PReLU(),
    
#     nn.Linear(in_features=256, out_features=2)
# )

# TODO: Try a better CNN lul

# 0.9999 Train, 0.9151 Val, 0.91585 Test
# Augmentation: 90.256|89.385|89.72
customResnet = nn.Sequential(
    ResidualBlock4(in_channels=3, out_channels=4),
    ResidualDownsample(in_channels=4), # 112
    
    ResidualBlock4(in_channels=4, out_channels=8),
    ResidualDownsample(in_channels=8), # 56
    
    ResidualBlock4(in_channels=8, out_channels=16),
    ResidualBlock4(in_channels=16, out_channels=32),
    ResidualDownsample(in_channels=32), # 28
    
    ResidualBlock4(in_channels=32, out_channels=64),
    ResidualBlock4(in_channels=64, out_channels=128),
    ResidualBlock4(in_channels=128, out_channels=128),
    ResidualDownsample(in_channels=128), # 14

    ResidualBlock4(in_channels=128, out_channels=256),
    *[ResidualBlock4(in_channels=256, out_channels=256) for _ in range(2)],
    ResidualDownsample(in_channels=256), # 7
    
    ResidualBlock4(in_channels=256, out_channels=512),
    *[ResidualBlock4(in_channels=512, out_channels=512) for _ in range(2)],
    ResidualDownsample(in_channels=512), # 3

    ResidualBlock4(in_channels=512, out_channels=1024),
    *[ResidualBlock4(in_channels=1024, out_channels=1024) for _ in range(2)],

    *[ResidualBlock4(in_channels=1024, out_channels=1024) for _ in range(3)],

    ResidualBlock4(in_channels=1024, out_channels=2048),
    *[ResidualBlock4(in_channels=2048, out_channels=2048) for _ in range(3)],

    ResidualBlock4(in_channels=2048, out_channels=4096),
    *[ResidualBlock4(in_channels=4096, out_channels=4096) for _ in range(2)],

    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    
    nn.Linear(in_features=4096, out_features=2048),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=2048),
    
    nn.Linear(in_features=2048, out_features=1024),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=1024),

    nn.Linear(in_features=1024, out_features=512),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=512),
    
    nn.Linear(in_features=512, out_features=2),
)

# Runs like twice as fast as previous ResNet, but looks like it will have worse performance# Also only uses 9.2 GB when compared to the 22 GB from ResNet 
# TODO: Continue training this later, seems to not have fully fit yet.
# After 15 epochs: Train: 92.414%, Val: 81.015%, Test: 81.035%
# After 30 epochs: Train 99.72, Val: 82.99, Test: 82.6
# superSepNetSmall = nn.Sequential(
#     DoubleResidualDWSeparableConv2d(in_channels=3, out_channels=9),
#     ResidualDownsampleSep(in_channels=9), # 112
    
#     DoubleResidualDWSeparableConv2d(in_channels=9, out_channels=18),
#     ResidualDownsampleSep(in_channels=18), # 56
    
#     DoubleResidualDWSeparableConv2d(in_channels=18, out_channels=36),
#     ResidualDownsampleSep(in_channels=36), # 28
    
#     DoubleResidualDWSeparableConv2d(in_channels=36, out_channels=72),
#     DoubleResidualDWSeparableConv2d(in_channels=72, out_channels=144),
#     DoubleResidualDWSeparableConv2d(in_channels=144, out_channels=144),
#     ResidualDownsampleSep(in_channels=144), # 14

#     DoubleResidualDWSeparableConv2d(in_channels=144, out_channels=288),
#     *[DoubleResidualDWSeparableConv2d(in_channels=288, out_channels=288) for _ in range(2)],
#     ResidualDownsampleSep(in_channels=288), # 7
    
#     DoubleResidualDWSeparableConv2d(in_channels=288, out_channels=576),
#     *[DoubleResidualDWSeparableConv2d(in_channels=576, out_channels=576) for _ in range(2)],
#     ResidualDownsampleSep(in_channels=576), # 3

#     DoubleResidualDWSeparableConv2d(in_channels=576, out_channels=1152),
#     *[DoubleResidualDWSeparableConv2d(in_channels=1152, out_channels=1152) for _ in range(5)],

#     DoubleResidualDWSeparableConv2d(in_channels=1152, out_channels=2304),
#     *[DoubleResidualDWSeparableConv2d(in_channels=2304, out_channels=2304) for _ in range(3)],

#     DoubleResidualDWSeparableConv2d(in_channels=2304, out_channels=4608),
#     *[DoubleResidualDWSeparableConv2d(in_channels=4608, out_channels=4608) for _ in range(2)],

#     nn.AdaptiveAvgPool2d(1),
#     nn.Flatten(),
    
#     nn.Linear(in_features=4608, out_features=2304),
#     nn.ReLU(),
#     nn.LayerNorm(normalized_shape=2304),
    
#     nn.Linear(in_features=2304, out_features=1152),
#     nn.ReLU(),
#     nn.LayerNorm(normalized_shape=1152),

#     nn.Linear(in_features=1152, out_features=512),
#     nn.ReLU(),
#     nn.LayerNorm(normalized_shape=512),
    
#     nn.Linear(in_features=512, out_features=2),
# )

# Didn't train at all, 7751 s train
# Changing to NAdam and much lower LR without AMP got it training
# 99.963 Train, 88.85 Val, 89.485 Test
# Need to cut batch size to 32 with augmentations because they use too much memory lole
# ABSOLUTELY GOATED WITH AUGS 98.65|96.74|96.64 TAKES FOREVER TO TRAIN: 16189s
superSepNetLarge = nn.Sequential(
    DoubleResidualDWSeparableConv2d(in_channels=3, out_channels=18),
    ResidualDownsampleSep(in_channels=18), # 112
    
    DoubleResidualDWSeparableConv2d(in_channels=18, out_channels=36),
    ResidualDownsampleSep(in_channels=36), # 56
    
    DoubleResidualDWSeparableConv2d(in_channels=36, out_channels=72),
    ResidualDownsampleSep(in_channels=72), # 28
    
    DoubleResidualDWSeparableConv2d(in_channels=72, out_channels=144),
    ResidualDownsampleSep(in_channels=144), # 14

    DoubleResidualDWSeparableConv2d(in_channels=144, out_channels=288),
    ResidualDownsampleSep(in_channels=288), # 7
    
    DoubleResidualDWSeparableConv2d(in_channels=288, out_channels=576),
    ResidualDownsampleSep(in_channels=576), # 3

    DoubleResidualDWSeparableConv2d(in_channels=576, out_channels=1152),
    *[DoubleResidualDWSeparableConv2d(in_channels=1152, out_channels=1152) for _ in range(6)],

    DoubleResidualDWSeparableConv2d(in_channels=1152, out_channels=2304),
    *[DoubleResidualDWSeparableConv2d(in_channels=2304, out_channels=2304) for _ in range(6)],

    DoubleResidualDWSeparableConv2d(in_channels=2304, out_channels=4608),
    *[DoubleResidualDWSeparableConv2d(in_channels=4608, out_channels=4608) for _ in range(8)],

    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    
    nn.Linear(in_features=4608, out_features=2304),
    nn.LayerNorm(normalized_shape=2304),
    nn.GELU(),

    nn.Linear(in_features=2304, out_features=1152),
    nn.LayerNorm(normalized_shape=1152),
    nn.GELU(),

    nn.Linear(in_features=1152, out_features=512),
    nn.LayerNorm(normalized_shape=512),
    nn.GELU(),

    nn.Linear(in_features=512, out_features=2),
)


def main():
    
    # Test model loading here
    # customModel = VisualizableSWIN()
    # customModel = VisualizableVIT() # Can't profile this one
    customModel = customResnet
    input_size = (32, 3, 224, 224)
    profileModel(customModel, input_size=input_size)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    customModel.to(device)
    
    TRAIN_TRANSFORM_AUG = v2.Compose([
        v2.Resize((224, 224)),  # Resize images to fit Swin Transformer input dimensions
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(),
        v2.RandomResizedCrop(size=224, scale=(0.7, 1)),
        v2.RandomGrayscale(p=0.05),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    ])
    
    dummyInput = torch.randn((input_size)).to(device)
    
    out = customModel(TRAIN_TRANSFORM_AUG(dummyInput))
    # print(out)
    print(f'{out.shape=}')
    
    # print(list(visualizableVIT.children()))
    pass

if __name__ == '__main__':
    main()