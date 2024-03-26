import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, TensorDataset
from datasets import RealVsFake140k, DEFAULT_INITIAL_TRANSFORM
from tqdm import tqdm
from models import swinModel, vitModel
import time

from parallelDataLoader import ParallelDataLoader



# TODO: This eats VRAM like crazy after the first epoch, find out why and fix it
# TODO: Make sure training actually works, I think it does but I didn't do a full eval
# TODO: Move the WarmupPlataeuScheduler and trainEpoch() function somewhere else???
# TODO: Make the torch flash attention warning fuck off because it's annoying
# TODO: Maybe use a TensorDataset? IDK how much of a speedup you could get but it's a bit slow



def main():
    
    # TODO: Find a way to properly integrate this shit because it SUCKS
    class WarmupPlataeuScheduler():
        
        """
        A simple class that combines both a warmup scheduler and a ReduceLROnPlateau scheduler to simplify 
        scheduler logic.
        """

        def __init__(self, warmup:torch.optim.lr_scheduler.LinearLR, 
                    plateauScheduler:torch.optim.lr_scheduler.ReduceLROnPlateau):
            
            self.warmup = warmup
            self.plateauScheduler = plateauScheduler
            self.warmupEpochs = warmup.total_iters
        
        def step(self, loss):
            if epoch < self.warmupEpochs:
                self.warmup.step()
            else:
                self.plateauScheduler.step(loss)
        
        def getLastLR(self):
            if epoch < self.warmupEpochs:
                return self.warmup.get_last_lr()[0]
            else:
                return self.plateauScheduler.optimizer.param_groups[0]['lr']  # Directly get LR from optimizer used in plateuScheduler


    def trainEpoch(model:nn.Module, parallelDataLoader:ParallelDataLoader, scheduler:WarmupPlataeuScheduler, freezeModel, numSubsets):
        
        if freezeModel:
            model.train()
        else:
            model.eval()
        running_loss = 0.0
                
        pbar = tqdm(range(numSubsets))
        for subsetNum in pbar:
            
            subLoader = parallelDataLoader.getNextSubset()
            
            for images, labels in subLoader:

                images:torch.Tensor; labels:torch.Tensor
                images, labels = images.to(device), labels.to(device)

                if not freezeModel:
                    optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss: torch.Tensor

                if not freezeModel:
                    loss.backward()
                    optimizer.step()

                    # Not sure which clipping works best
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0) # Clip gradients after calculating loss
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                running_loss += loss.item()
                
            currentLr = scheduler.getLastLR()
            pbar.set_description("Running loss: {:.6f}, lr: {:.6f}".format(running_loss/(subsetNum+1), currentLr), refresh=True)


        return running_loss # TODO: Return an accuracy metric as well as loss but I'm tired tonight so do it later




    
    # Define transformations
    transform = DEFAULT_INITIAL_TRANSFORM

    BATCH_SIZE = 64
    NUM_EPOCHS = 3
    NUM_SUBSETS = 64
    warmupEpochs = 4


    # Download and prepare datasets
    trainset = RealVsFake140k(transform=transform, split='train')
    valset =  RealVsFake140k(transform=transform, split='valid')
    testset =  RealVsFake140k(transform=transform, split='test')


    # Define train/val/test subset size, set SPLIT_FRACTION = 1 to use the whole thing
    SPLIT_FRACTION = 0.5
    trainset = Subset(trainset, indices=torch.randint(0, RealVsFake140k.TRAIN_SIZE, (int(RealVsFake140k.TRAIN_SIZE*SPLIT_FRACTION),)))
    valset = Subset(valset, indices=torch.randint(0, RealVsFake140k.VALID_SIZE, (int(RealVsFake140k.VALID_SIZE*SPLIT_FRACTION),)))
    testset = Subset(testset, indices=torch.randint(0, RealVsFake140k.TEST_SIZE, (int(RealVsFake140k.TEST_SIZE*SPLIT_FRACTION),)))


    # Dataloaders
    # trainLoader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    # validationLoader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
    testLoader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    dataLoaderKwargs = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2
    }





    # Define model
    model = vitModel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001) # ADAM IS WASHED, SGD SUPREMACY
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.01, nesterov=True)



    # Use LR warmup schedule and reduce learning rate on loss plateu
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-9, end_factor=1, total_iters=warmupEpochs)
    plateuScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, threshold=1e-2, cooldown=1)

    scheduler = WarmupPlataeuScheduler(warmup=warmup, plateauScheduler=plateuScheduler)

    startTime = time.time()

    # Training loop
    for epoch in range(NUM_EPOCHS):

        # Takes 216s for 10% 3 epochs 64 batch. Takes 595s for 2 epochs 50%. Takes 854s for 3 epochs 50% ~4:20 per train epoch
        parallelTrainLoader = ParallelDataLoader(trainset, numSubsets=NUM_SUBSETS, transform=DEFAULT_INITIAL_TRANSFORM, dataLoaderKwargs=dataLoaderKwargs)
        parallelValidationLoader = ParallelDataLoader(valset, numSubsets=NUM_SUBSETS, transform=DEFAULT_INITIAL_TRANSFORM, dataLoaderKwargs=dataLoaderKwargs)

        parallelTrainLoader.getRandomEpochIndices()
        parallelValidationLoader.getRandomEpochIndices()

        totalTrainLoss = trainEpoch(model, parallelTrainLoader, scheduler, freezeModel=False, numSubsets=NUM_SUBSETS)
        with torch.no_grad():
            totalValidationLoss = trainEpoch(model, parallelValidationLoader, scheduler, freezeModel=True, numSubsets=NUM_SUBSETS)

        # Do plateau scheduler step based on validation loss instead of train loss so we only reduce lr when validation loss stops improving
        scheduler.step(totalValidationLoss/len(valset))

        print(f"Epoch {epoch+1}, Train Loss: {totalTrainLoss/len(trainset)}, Validation Loss: {totalValidationLoss/len(valset)}")

    print("Finished Training")
    print(f'Train time was {time.time() - startTime}')

    # Testing loop
    correct = 0
    total = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in testLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {int(RealVsFake140k.TEST_SIZE*SPLIT_FRACTION)} test images: {100 * correct / total}%')
    
    pass

# Do this because pytorch gets mad when num_workers > 0 and there isn't a main guard
if __name__ == '__main__':
    main()
    
