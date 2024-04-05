import torch
import torch.nn as nn
import os
import torchinfo

from torch.utils.data import Subset, DataLoader
from datasets import RealVsFake140k, DEFAULT_INITIAL_TRANSFORM

import matplotlib.pyplot as plt

from torch.profiler import profile, record_function, ProfilerActivity
from tabulate import tabulate



def getDataLoaders(splitFraction=1, dataLoaderKwargs={}, transform=DEFAULT_INITIAL_TRANSFORM):

    # Download and prepare datasets
    trainset = RealVsFake140k(transform=transform, split='train')
    valset =  RealVsFake140k(transform=transform, split='valid')
    testset =  RealVsFake140k(transform=transform, split='test')


    # Define train/val/test subset size, set SPLIT_FRACTION = 1 to use the whole thing
    trainset = Subset(trainset, indices=torch.randint(0, RealVsFake140k.TRAIN_SIZE, (int(RealVsFake140k.TRAIN_SIZE*splitFraction),)))
    valset = Subset(valset, indices=torch.randint(0, RealVsFake140k.VALID_SIZE, (int(RealVsFake140k.VALID_SIZE*splitFraction),)))
    testset = Subset(testset, indices=torch.randint(0, RealVsFake140k.TEST_SIZE, (int(RealVsFake140k.TEST_SIZE*splitFraction),)))

    # Dataloaders
    trainLoader = DataLoader(trainset, shuffle=True, **dataLoaderKwargs)
    validationLoader = DataLoader(valset, shuffle=False, **dataLoaderKwargs)
    testLoader = DataLoader(testset, shuffle=False, **dataLoaderKwargs)

    return trainLoader, validationLoader, testLoader


def getSaveFileName(rootPath, epochs, batch_size, lr, momentum, modelName):

    duplicateID = 0
    filename = ''
    
    while True:
        
        epochs = epochs
        batch_size = batch_size
        lr = lr
        momentum = momentum
        modelName = modelName
        
        filename = os.path.join(rootPath, f'{modelName}-{duplicateID}_Epoch{epochs}_Batch{batch_size}_LR{lr}_Momentum{momentum}')
        duplicateID += 1
        
        if not os.path.exists(filename):
            break

    return filename



def validateModelIO(model:nn.Module, printSummary=True, batchSize=5) -> torchinfo.ModelStatistics:
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