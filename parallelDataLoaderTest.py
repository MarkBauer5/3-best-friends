import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, TensorDataset, Dataset
from datasets import RealVsFake140k, DEFAULT_INITIAL_TRANSFORM
import time
import numpy as np
import threading
from queue import Queue



def trainEpoch(model:nn.Module, dataloader:DataLoader, scheduler, freezeModel):
    
    time.sleep(1)
    pass

def loadNextSubset(fullDataset, indices: np.ndarray, transform, dataLoaderKwargs: dict, queue: Queue, num=0):
    
    print(f'Loading next subset... ({num})')
    
    subset = Subset(fullDataset, indices)
    features = torch.stack([transform(sample) for sample, _ in subset])
    labels = torch.tensor([label for _, label in subset])
    
    tensorSubset = TensorDataset(features, labels)
    
    # Put the DataLoader in the queue
    loader = DataLoader(tensorSubset, **dataLoaderKwargs)
    loader.i = num
    queue.put(loader)
    print(f'Finished loading ({num})')

def getRandomEpochIndices(numSubsets, validIndices: np.ndarray) -> list:
    
    validIndicesCopy = validIndices.copy()
    np.random.shuffle(validIndicesCopy)
    return np.array_split(validIndicesCopy, numSubsets)



class ParallelDataLoader():


    # TODO: Make this work on the size of individual subsets instead of counting subsets themselves    
    def __init__(self, dataset, validIndices, numSubsets, maxQueueSize=4, transform=None, dataLoaderKwargs:dict=None):
        
        
        self.dataset = dataset
        self.validIndices = validIndices
        self.numSubsets = numSubsets
        self.maxQueueSize = maxQueueSize
        self.transform = transform
        self.dataLoaderKwargs = dataLoaderKwargs
        
        self.randomSubsets = []
        self.threads = []
        self.currentSubset = 0
        self.numComplete = 0
        self.queue = Queue()        
        
    
    
    
    def populateQueue(self):
        
        # Populate queue of DataLoaders to be used in training in parallel.
        while len(self.randomSubsets) > 0 and self.numSubsets - len(self.randomSubsets) - self.numComplete < self.maxQueueSize:

            currentIndices = self.randomSubsets.pop()
            thread = threading.Thread(target=self.loadNextSubset, 
                        args=(self.dataset, currentIndices), daemon=True)
            thread.start()
            self.threads.append(thread)
            self.currentSubset += 1
            
    
    def loadNextSubset(self, fullDataset: Dataset, indices: np.ndarray):
    
        num = self.currentSubset
        print(f'Loading next subset... ({num})')
        
        subset = Subset(fullDataset, indices)
        features = torch.stack([self.transform(sample) for sample, _ in subset])
        labels = torch.tensor([label for _, label in subset])
        
        tensorSubset = TensorDataset(features, labels)
        
        # Put the DataLoader in the queue
        loader = DataLoader(tensorSubset, **self.dataLoaderKwargs)
        loader.i = num
        self.queue.put(loader)
        print(f'Finished loading ({num})')
    
    
    def getRandomEpochIndices(self):
        
        np.random.shuffle(self.validIndices)
        self.randomSubsets = np.array_split(self.validIndices, self.numSubsets)


    def getNextSubset(self):
        
        self.populateQueue()
        self.numComplete += 1
        return self.queue.get()














# if __name__ == '__main__':

#     trainDataset = RealVsFake140k(transform=DEFAULT_INITIAL_TRANSFORM)
#     trainDataset = Subset(trainDataset, indices=torch.randint(0, RealVsFake140k.TRAIN_SIZE, size=(4000,)))

#     trainDatasetSize = len(trainDataset)

#     EPOCH_COUNT = 10
#     SUBSET_COUNT = 17
#     MAX_QUEUE_SIZE = 4

#     for _ in range(EPOCH_COUNT):
        
#         # Get next dataset in parallel and create DataLoader
#         randomSubsets = getRandomEpochIndices(numSubsets=SUBSET_COUNT, 
#                             validIndices=np.arange(trainDatasetSize))
        
#         dataLoaderKwargs = {
#             'batch_size': 64,
#             'shuffle': True
#         }
        
#         dataLoaderQueue = Queue()
#         threads = []
        
#         currentSubset = 0

#         for i in range(SUBSET_COUNT):

#             # Populate queue of DataLoaders to be used in training in parallel.
#             while len(randomSubsets) > 0 and len(threads) - i < MAX_QUEUE_SIZE:

#                 currentIndices = randomSubsets.pop()
#                 thread = threading.Thread(target=loadNextSubset, 
#                             args=(trainDataset, currentIndices, DEFAULT_INITIAL_TRANSFORM, 
#                                 dataLoaderKwargs, dataLoaderQueue, currentSubset), daemon=True)
#                 thread.start()
#                 threads.append(thread)
#                 currentSubset += 1
                        
#             # Get next DataLoader, train, and mark the next item in the queue as done
#             currentDataLoader = dataLoaderQueue.get()
            
#             print(f'Training {currentDataLoader.i}')
#             trainEpoch(model=None, dataloader=currentDataLoader, scheduler=None, freezeModel=False)
#             print(f'Done training {currentDataLoader.i}')

#             # dataLoaderQueue.task_done()

#         # Wait for all processes to finish
#         for thread in threads:
#             thread.join()
            
#         print('NEXT EPOCH')




if __name__ == '__main__':

    trainDataset = RealVsFake140k(transform=DEFAULT_INITIAL_TRANSFORM)
    trainDataset = Subset(trainDataset, indices=torch.randint(0, RealVsFake140k.TRAIN_SIZE, size=(4000,)))

    trainDatasetSize = len(trainDataset)

    dataLoaderKwargs = {
        'batch_size': 64,
        'shuffle': True
    }

    SUBSET_COUNT = 17
    MAX_QUEUE_SIZE = 4

    parallelDataset = ParallelDataLoader(dataset=trainDataset, validIndices=np.arange(trainDatasetSize), numSubsets=SUBSET_COUNT,
                                         maxQueueSize=MAX_QUEUE_SIZE, transform=DEFAULT_INITIAL_TRANSFORM, dataLoaderKwargs=dataLoaderKwargs)


    EPOCH_COUNT = 10


    for _ in range(EPOCH_COUNT):
        
        # Get next dataset in parallel and create DataLoader
        parallelDataset.getRandomEpochIndices()

        for i in range(SUBSET_COUNT):

            # Get next DataLoader, train, and mark the next item in the queue as done
            currentDataLoader = parallelDataset.getNextSubset()
            
            print(f'Training {currentDataLoader.i}')
            trainEpoch(model=None, dataloader=currentDataLoader, scheduler=None, freezeModel=False)
            print(f'Done training {currentDataLoader.i}')

            
        print('NEXT EPOCH')
