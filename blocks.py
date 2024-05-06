from torch import nn
import torch



"""
Yoinked straight from my CS 444 final project lol

Define custom architecture blocks here
"""


class BranchBlockNorm(nn.Module):
    
    """
    The BranchBlockNorm splits one feature map into several parallel classifiers before concatenating 
    their outputs together again.
    This also creates a residual connection between the inputs and the concatenated branches. 
    The Norm version also normalizes activations via a batch normalization following all the branches
    """
        
    def __init__(self, in_channels:int, branches:list, activation=nn.ReLU(), averageChannels=False) -> None:
        
        super().__init__()
        
        self.activation = activation
        
        self.inputNorm = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            self.activation
        )
        
        self.out_channels = in_channels * len(branches)
        # We need to use a ModuleList to ensure that the .to(device) operation registers these as submodules
        self.branches = nn.ModuleList(branches)
        
        self.averageChannels = averageChannels
    

    def forward(self, x: torch.Tensor):
        normInput = self.inputNorm(x)
        # Run all branches in parallel and stack results into a (numBranches, batch_size, C, H, W) tensor
        rawOutputs = nn.parallel.parallel_apply(self.branches, [normInput] * len(self.branches))
        # We need to use stack() here instead of view since we need all the outputs to be contiguous in memory
        rawOutputs = torch.stack(rawOutputs)
        
        # Reshape rawOutputs to merge the branch dimension with the batch dimension
        # This allows for it to be passed into normalization so it will be (numBranches*batch_size, C, H, W)
        rawOutputsMerged = rawOutputs.view(-1, *rawOutputs.shape[2:])
        normOutputsMerged = self.inputNorm(rawOutputsMerged)

        # Reshape the normalized outputs back to the original shape (numBranches, batch_size, C, H, W)
        normOutputs = normOutputsMerged.view(*rawOutputs.shape)

        # Add the residual connection, we can't use += here since that breaks things for backpropagation
        normOutputs = normOutputs + x.unsqueeze(0)
        
        if self.averageChannels:
            # Average over branches
            y = torch.mean(normOutputs, dim=0)
        else:
            # Merge the branches and channels dimensions to get (batch_size, C*numBranches, H, W)
            y = normOutputs.view(x.size(0), -1, *x.shape[2:])
        
        return y
    
    
    
class HighwayBlock(nn.Module):
    
    
    printOutsize = False
    
    """
    A network block with longer residual connections that allows long stretches of residual connections.
    """
    
    def __init__(self, highwaySequence:nn.Sequential=None, *args, **kwargs) -> None:
        
        """
        Initialize a long highway block with continuous residual connections for long parts of the network
        """
        
        super().__init__(*args, **kwargs)
                
        self.highwaySequence = highwaySequence

        
        
        
    def forward(self, x):
        
        """
        Performs a forward pass on all highwaySequence elements. Also creates a long skip connection which propagates
            the input all the way to the output.
        """
        
        firstPass = True
        output = torch.empty_like(x)
        
        for layer in self.highwaySequence:
                        
            y = torch.empty_like(x)
            
            if firstPass:
                y = layer(x)
                firstPass = False
            else:
                y = layer(output)
            
            # TODO: Add original activations to each layer?
            output = y

        # Add the original residual creating a long skip from the start of the highway to the end
        return output + x
    
    
    
    
    
class ResidualBlock4(nn.Module):
        
    def __init__(self, in_channels, out_channels, activation:nn.Module=nn.GELU(), kernel_size:int=3, 
                stride:int=1, padding:int='same'):
        super().__init__()
        
        
        self.activation = activation
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            self.activation,
        )
        
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            self.activation,
        )
        
        self.outNorm = nn.BatchNorm2d(num_features=out_channels)

        
        if in_channels != out_channels:
            self.residualLayer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.residualLayer = nn.Identity()
                
    def forward(self, x):

        residual = self.residualLayer(x)

        y1 = self.c1(x)
        y = self.c2(y1)
        
        return self.outNorm(y + residual)
    
    
class ResidualDownsample(nn.Module):
    
    def __init__(self, in_channels):
        super(ResidualDownsample, self).__init__()
        
        self.downPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downLearned = nn.Sequential(
            ResidualBlock4(in_channels=in_channels, out_channels=in_channels),
            self.downPool
        )
        
        self.outNorm = nn.BatchNorm2d(num_features=in_channels)
        
    def forward(self, x):
        
        xPool = self.downPool(x)
        downLearned = self.downLearned(x)
        
        return self.outNorm(xPool + downLearned)
    
    
    
    
    
    
    
# TODO: This seems completely broken in pytorch? Every sources says this should be faster but it's way slower
# TODO: TRY THIS WITH AMP, IT MIGHT SUCK LESS!!!
class DepthwiseSeparableConv2d(nn.Module):
    
    """
    Implements depthwise-pointwise convolution for faster performance. I think this was used in EfficientNet
    https://www.youtube.com/watch?v=vVaRhZXovbw&list=WL&index=44
    """
    
    def __init__(self, in_channels, out_channels, activation:nn.Module=nn.GELU(), 
                kernel_size:int=3, stride:int=1, padding:int='same'):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.activation = activation
        
        if stride != 1:
            padding = 0
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            self.activation,
        )
        
    def forward(self, x):
        
        y1 = self.depthwise(x)
        return self.pointwise(y1)
        
        
class ResidualDWSeparableConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation:nn.Module=nn.ReLU(), kernel_size:int=3):
        super(ResidualDWSeparableConv2d, self).__init__()
        
        self.activation = activation
        
        self.conv1 = DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, activation=activation, kernel_size=kernel_size, padding='same')
        self.conv2 = DepthwiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, activation=activation, kernel_size=kernel_size, padding='same')
        
        self.outNorm = nn.BatchNorm2d(num_features=out_channels)

        # If in channels neq out channels, pass through a 1x1 convolution to change channel count. Sizes must be the same still!
        if in_channels != out_channels:
            self.residualLayer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.residualLayer = nn.Identity()
        
    def forward(self, x):
        
        residual = self.residualLayer(x)
        
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        
        # Do the residual connection by adding inputs to outputs
        return self.activation(y2+residual)
    
    















class DoubleDepthwiseSeparableConv2d(nn.Module):
    
    """
    Implements depthwise-pointwise convolution for faster performance. I think this was used in EfficientNet
    https://www.youtube.com/watch?v=vVaRhZXovbw&list=WL&index=44
    
    This Double separable version separates into both vertical and horizontal convolutions for max efficiency
    """
    
    def __init__(self, in_channels, out_channels, activation:nn.Module=nn.GELU(), 
                kernel_size:int=3, stride:int=1, padding:int='same'):
        super(DoubleDepthwiseSeparableConv2d, self).__init__()
        
        self.activation = activation
        
        if stride != 1:
            padding = 0
        
        # Spatial and channelwise separable convolution should improve speed while hopefully keeping performance the same
        # No activation because original paper says that's better
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                    kernel_size=(kernel_size, 1), stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                    kernel_size=(1, kernel_size), stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            self.activation,
        )
        
    def forward(self, x):
        
        y1 = self.depthwise(x)
        return self.pointwise(y1)
        
        
class DoubleResidualDWSeparableConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation:nn.Module=nn.GELU(), kernel_size:int=3):
        super(DoubleResidualDWSeparableConv2d, self).__init__()
        
        self.activation = activation
        
        self.conv1 = DoubleDepthwiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, activation=activation, kernel_size=kernel_size, padding='same')
        self.conv2 = DoubleDepthwiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, activation=activation, kernel_size=kernel_size, padding='same')
        
        self.outNorm = nn.BatchNorm2d(num_features=out_channels)

        # If in channels neq out channels, pass through a 1x1 convolution to change channel count. Sizes must be the same still!
        if in_channels != out_channels:
            self.residualLayer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.residualLayer = nn.Identity()
        
    def forward(self, x):
        
        residual = self.residualLayer(x)
        
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        
        # Do the residual connection by adding inputs to outputs
        return self.activation(y2+residual)
    
    
class ResidualDownsampleSep(nn.Module):
    
    def __init__(self, in_channels):
        super(ResidualDownsampleSep, self).__init__()
        
        self.downPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downLearned = nn.Sequential(
            DoubleResidualDWSeparableConv2d(in_channels=in_channels, out_channels=in_channels),
            self.downPool
        )
        
        self.outNorm = nn.BatchNorm2d(num_features=in_channels)
        
    def forward(self, x):
        
        xPool = self.downPool(x)
        downLearned = self.downLearned(x)
        
        return self.outNorm(xPool + downLearned)