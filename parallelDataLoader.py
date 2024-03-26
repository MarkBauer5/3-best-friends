import torch
from torch.utils.data import Subset, DataLoader, TensorDataset, Dataset
import numpy as np
import threading
from queue import Queue



class ParallelDataLoader():


    # TODO: Make this work on the size of individual subsets instead of counting subsets themselves    
    def __init__(self, dataset, validIndices=None, numSubsets=16, maxQueueSize=4, transform=None, dataLoaderKwargs:dict=None, printDebug=False):
        
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.dataset = dataset
        if validIndices is not None:
            self.validIndices = validIndices
        else:
            self.validIndices = np.arange(len(dataset))
        self.numSubsets = numSubsets
        self.maxQueueSize = maxQueueSize
        self.transform = transform
        self.dataLoaderKwargs = dataLoaderKwargs
        
        self.randomSubsets = []
        self.threads = []
        self.currentSubset = 0
        self.numComplete = 0
        self.queue = Queue()        
    
        self.printDebug = printDebug    
    
        
    def _populateQueue(self):
        
        """
        Populate the queue with the next subsets in parallel.
        """
        
        # Populate queue of DataLoaders to be used in training in parallel.
        while len(self.randomSubsets) > 0 and self.numSubsets - len(self.randomSubsets) - self.numComplete < self.maxQueueSize:

            currentIndices = self.randomSubsets.pop()
            thread = threading.Thread(target=self._loadNextSubset, 
                        args=(self.dataset, currentIndices), daemon=True)
            thread.start()
            self.threads.append(thread)
            self.currentSubset += 1
            
    
    def _loadNextSubset(self, fullDataset: Dataset, indices: np.ndarray):
    
        """
        Take the next subset of indices, create a TensorDataset DataLoader and place it in the queue
        """
    
        num = self.currentSubset
        
        if self.printDebug:
            print(f'Loading next subset... ({num})')
        
        subset = Subset(fullDataset, indices)
        features = torch.stack([self.transform(sample) for sample, _ in subset])#.to(self.device)
        labels = torch.tensor([label for _, label in subset])#.to(self.device)
        
        tensorSubset = TensorDataset(features, labels)
        
        # Put the DataLoader in the queue
        loader = DataLoader(tensorSubset, **self.dataLoaderKwargs)
        self.queue.put(loader)
        
        if self.printDebug:
            print(f'Finished loading ({num})')
    
    
    def getRandomEpochIndices(self):
        
        """
        Initializes or re-rolls random indices. This should be done for each epoch of training to ensure the data
        is randomly shuffled.
        """
        
        np.random.shuffle(self.validIndices)
        self.randomSubsets = np.array_split(self.validIndices, self.numSubsets)


    def getNextSubset(self) -> DataLoader:
        
        """
        Gets the next DataLoader subset. This is based on a TensorDataset and should be faster.
        """
        
        self._populateQueue()
        self.numComplete += 1
        return self.queue.get()
