import torch
import torch.nn as nn
import os
import torchinfo

from torch.utils.data import Subset, DataLoader
from datasets import RealVsFake140k, DEFAULT_INITIAL_TRANSFORM

import matplotlib.pyplot as plt

from torch.profiler import profile, record_function, ProfilerActivity
from tabulate import tabulate



def getDataLoaders(splitFraction=1, dataLoaderKwargs={}, trainTransform=DEFAULT_INITIAL_TRANSFORM, valTestTransform=DEFAULT_INITIAL_TRANSFORM) -> tuple[DataLoader, DataLoader, DataLoader]:

    """
    Get Train, Validation, and Test dataloaders for the RealVsFake140k dataset
    
    Arguments:
        splitFraction: What percentage of the overall dataset should we use
        dataLoaderKwargs: What kwargs to use for all the dataloaders like num_workers and such
        trainTransform: What data augmentation to add to the training data
        valTestTransform: What data augmentation to add to the validation and test data
        
    Returns:
        trainLoader, validationLoader, testLoader
    """

    assert splitFraction <= 1 and splitFraction > 0, f"ERROR: Split fraction {splitFraction} is out of bounds"

    # Download and prepare datasets
    trainset = RealVsFake140k(transform=trainTransform, split='train')
    valset =  RealVsFake140k(transform=valTestTransform, split='valid')
    testset =  RealVsFake140k(transform=valTestTransform, split='test')


    # Define train/val/test subset size, set splitFraction = 1 to use the whole thing
    trainset = Subset(trainset, indices=torch.randint(0, RealVsFake140k.TRAIN_SIZE, (int(RealVsFake140k.TRAIN_SIZE*splitFraction),)))
    valset = Subset(valset, indices=torch.randint(0, RealVsFake140k.VALID_SIZE, (int(RealVsFake140k.VALID_SIZE*splitFraction),)))
    testset = Subset(testset, indices=torch.randint(0, RealVsFake140k.TEST_SIZE, (int(RealVsFake140k.TEST_SIZE*splitFraction),)))

    # Dataloaders
    trainLoader = DataLoader(trainset, shuffle=True, **dataLoaderKwargs)
    validationLoader = DataLoader(valset, shuffle=False, **dataLoaderKwargs)
    testLoader = DataLoader(testset, shuffle=False, **dataLoaderKwargs)

    return trainLoader, validationLoader, testLoader


# TODO: Maybe refactor this to allow for kwargs and construct the filename based on that
def getSaveFileName(rootPath: str, epochs: int, batch_size: int, lr: float, momentum: float, modelName: str):

    """
    Produces a save file name for the model and tensorboard data. Automatically handles duplicate names.
    
    Arguments:
        rootPath: The folder this should be saved
        epochs: Number of epochs
        batch_size: Batch size
        lr: Initial Learning Rate
        momentum: Momentum for optimizer
        modelName: The name of the model
        
    Returns: 
        filename: A string representation of the file the data should be written to
    """

    duplicateID = 0
    filename = ''
    
    # Keep adding 1 to the duplicateID until we find one not in use
    while True:        
        filename = os.path.join(rootPath, f'{modelName}-{duplicateID}_Epoch{epochs}_Batch{batch_size}_LR{lr}_Momentum{momentum}')
        duplicateID += 1
        
        if not os.path.exists(filename):
            break

    return filename



def validateModelIO(model:nn.Module, printSummary=True, batchSize=5) -> torchinfo.ModelStatistics:
    
    """
    Validates whether or not an input tensor of a given shape will produce an output of a correct shape.
    Returns a torchinfo.ModelStatistics object which can be used for more profiling.
    
    Arguments:
        model: The model to be evaluated
        printSummary: Whether or not to print a summary of the model
        batchSize: An example batch size to be tested
        
    Returns:
        summaryObject: A torchinfo.ModelStatistics object containing information about the model.
    """
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    
    print(f"Using device: {device}")

    model = model.to(device)

    dummy_input = torch.randn(batchSize, 3, 224, 224, device=device, dtype=torch.float)
    output = model(dummy_input)
    assert output.size() == (batchSize, 2), f"Expected output size ({batchSize}, 2), got {output.size()}!"

    summaryObject = torchinfo.summary(model=model, input_size=(batchSize, 3, 224, 224), device=device, mode='train', depth=20, verbose=0)

    if printSummary:
        # print(model)
        # print(summaryObject)
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")

    print("Test passed!")
    
    return summaryObject


def profileModel(model:nn.Sequential, input_size:tuple, printOriginalTable=False):
    
    """
    Prints and plots relevant model information to get a sense of model size and expected performance.
    
    Arguments:
        model: A Sequential representation of a model
        input_size: The shape of the expected input in the form (B, C, W, H)
        printoriginalTable: Whether or not to print the original tables from the profiling library
        
    """
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    randomInput = torch.randn(input_size, device=device, dtype=torch.float)
    model = model.to(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True, with_modules=True, record_shapes=True, profile_memory=True) as profilerContext:
        
        output = randomInput
        for i, layer in enumerate(model.children()):
            # Get the class name for each layer and add an "_" so we can filter it
            with record_function(f"_{type(layer).__name__} {i}"):
                output = layer(output)

    originalEvents = profilerContext.key_averages()

    # Filter events with a custom name
    events = [event for event in originalEvents if event.key.startswith('_')]
    attributesList = []

    assert len(list(model.children())) == len(events)

    # Manually extract parameters from each event
    for event, layer in zip(events, model.children()):
        cpu_time = event.cpu_time_total
        cuda_time = event.cuda_time_total
        cuda_memory_usage = round(event.cuda_memory_usage / (1024**2), 3)
        param_count = sum([p.numel() for p in layer.parameters()])
        key = event.key
        attributesList.append({
            'key': key,
            'cpu_time': cpu_time,
            'cuda_time': cuda_time,
            'cuda_memory_usage': cuda_memory_usage,
            'param_count': param_count
        })
        
    # Creates a 2D list in the same shape as a table
    table = [[val for _, val in attrs.items()] for attrs in attributesList]
    
    nRows = len(table)
    nCols = len(table[0])
    
    columnTotals = [None, ]
    # Start from 1 since we want to skip the names of the layers
    for cIdx in range(1, nCols):
        currentTotal = 0
        for rIdx in range(nRows):
            currentTotal += table[rIdx][cIdx]
        columnTotals.append(currentTotal)
    
    table.append(columnTotals)
    
    print(tabulate(table, headers=['CPU Time', 'CUDA Time (ms)', 'CUDA Memory Usage (MB)', 'Parameter Count'], tablefmt='outline'))

    cudaTimes = [attr['cuda_time']/1000 for attr in attributesList]    
    cudaMemories = [attr['cuda_memory_usage'] for attr in attributesList]
    param_counts = [attr['param_count'] for attr in attributesList]

    layerCount = len(attributesList)

    plt.bar(range(layerCount), cudaTimes), plt.title('CUDA times'), plt.xlabel('Layer number'), plt.xticks(range(layerCount)), plt.ylabel('Layer time (ms)'), plt.show()
    plt.bar(range(layerCount), cudaMemories), plt.title('CUDA Memory Usage (MB)'), plt.xlabel('Layer number'), plt.xticks(range(layerCount)), plt.ylabel('Memory usage (MB)'), plt.show()
    plt.bar(range(layerCount), param_counts), plt.title('Parameter Counts'), plt.xlabel('Layer number'), plt.xticks(range(layerCount)), plt.ylabel('Parameters'), plt.show()

    if printOriginalTable:
        outputTable = originalEvents.table()
        print(outputTable)
        
        
class WarmupPlateauScheduler():
    
    """
    A simple class that combines both a warmup scheduler and a ReduceLROnPlateau scheduler to simplify 
    scheduler logic. Simply call scheduler.step(metric) and this will handle chaining the warmup and plateau schedulers
    """

    def __init__(self, warmup:torch.optim.lr_scheduler.LinearLR, 
                plateauScheduler:torch.optim.lr_scheduler.ReduceLROnPlateau):
        
        """
        Define a WarmupPlateauScheduler by providing independent warmup and plateau schedulers respectively.
        
        Arguments:
            warmup: The warmup scheduler to be used. Note that this scheduler should have the total_iters kwarg set to determine when to 
                transition from warmup to plateau scheduling. The default defined in most schedulers is 5.
            plateauScheduler: A plateau scheduler with whatever parameters you want.    
        """
        
        self.warmup = warmup
        self.plateauScheduler = plateauScheduler
        self.warmupEpochs = warmup.total_iters
        self._currentEpoch = 0
    
    def step(self, loss):
        if self._currentEpoch < self.warmupEpochs:
            self.warmup.step()
        else:
            self.plateauScheduler.step(loss)
            
        self._currentEpoch += 1
    
    def getLastLR(self):
        if self._currentEpoch < self.warmupEpochs:
            return self.warmup.get_last_lr()[0]
        else:
            return self.plateauScheduler.optimizer.param_groups[0]['lr']  # Directly get LR from optimizer used in plateuScheduler