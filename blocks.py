from torch import nn
import torch



"""
Yoinked straight from my CS 444 final project lole

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
    
    
    def forward(self, x:torch.Tensor):
        
        normInput = self.inputNorm(x)
                
        rawOutputs = [branch(normInput) for branch in self.branches]
        normOutputs = [self.inputNorm(raw) + x for raw in rawOutputs]
        if self.averageChannels:
            y = torch.mean(torch.stack(normOutputs), dim=0)
        else:
            y = torch.cat(normOutputs, dim=1)
                
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
    
    
class ResidualBlock(nn.Module):
        
    printOutsize = False
        
    def __init__(self, channelCount, activation:nn.Module=nn.ReLU(), kernel_size:int=3, stride:int=1, padding:int='same'):
        super().__init__()
        
        self.activation = activation
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=channelCount, out_channels=channelCount, kernel_size=kernel_size, 
                      stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=channelCount),
            self.activation,
        )
        
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=channelCount, out_channels=channelCount, kernel_size=kernel_size, 
                      stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=channelCount),
            self.activation,
        )

    def forward(self, x):
        
        if self.printOutsize:
            print(f'x.size(): {x.size()}')
            
        y1 = self.c1(x)
        
        if self.printOutsize:
            print(f'y1.size(): {y1.size()}')
            
        y = self.c2(y1)
        
        if self.printOutsize:
            print(f'y.size(): {y.size()}\n')
            
        y = y + x
        self.outsize = y.size()
        
        return self.activation(y)
    
    
class DepthwiseSeparableConv2d(nn.Module):
    
    """
    Implements depthwise-pointwise convolution for faster performance. I think this was used in EfficientNet
    
    https://www.youtube.com/watch?v=vVaRhZXovbw&list=WL&index=44
    """
    
    def __init__(self, in_channels, out_channels, activation:nn.Module=nn.ReLU(), 
                 kernel_size:int=3, stride:int=1, padding:int='same'):
        super().__init__()
        
        self.activation = activation
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
            nn.BatchNorm2d(num_features=out_channels),
            self.activation,
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            self.activation,
        )
        
    def forward(self, x):
        
        y1 = self.depthwise(x)
        return self.pointwise(y1)
        
        
class ResidualDWSeparableConv2d(nn.Module):
    
    def __init__(self, in_channels, activation:nn.Module=nn.ReLU(), kernel_size:int=3):
        super().__init__()
        
        self.activation = activation
        
        self.conv1 = DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=in_channels, activation=activation, 
                                              kernel_size=kernel_size, padding='same')
        
        self.conv2 = DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=in_channels, activation=activation, 
                                              kernel_size=kernel_size, padding='same')
        
    def forward(self, x):
        
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        
        # Do the residual connection by adding inputs to outputs
        return self.activation(y2+x)