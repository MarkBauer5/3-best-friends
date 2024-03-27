import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from datasets import RealVsFake140k, DEFAULT_INITIAL_TRANSFORM
from tqdm import tqdm
from models import swinModel, vitModel
import time
import os
from collections import defaultdict

from tensorboardX import SummaryWriter


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


    def trainEpoch(model:nn.Module, dataloader:DataLoader, scheduler:WarmupPlataeuScheduler, freezeModel):
        
        model.train()
        running_loss = 0.0                
        correct = 0
        total = 0

        statsDict = defaultdict(list)
                
        pbar = tqdm(range(len(dataloader)))
        for batchNum, (images, labels) in zip(pbar, dataloader):
            
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
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct/total
            
            currentLr = scheduler.getLastLR()
            
            statsDict['loss'].append(loss.item()) 
            statsDict['lr'].append(currentLr)
            statsDict['accuracy'].append(accuracy)
            
            pbar.set_description("loss: {:.6f}, lr: {:.6f}, accuracy: {:.5f}".format(running_loss/(batchNum+1), currentLr, accuracy), refresh=True)

        
        return running_loss, accuracy, statsDict


    def getDataLoaders(splitFraction = 1, dataLoaderKwargs={}):

        # Download and prepare datasets
        trainset = RealVsFake140k(transform=transform, split='train')
        valset =  RealVsFake140k(transform=transform, split='valid')
        testset =  RealVsFake140k(transform=transform, split='test')


        # Define train/val/test subset size, set SPLIT_FRACTION = 1 to use the whole thing
        trainset = Subset(trainset, indices=torch.randint(0, RealVsFake140k.TRAIN_SIZE, (int(RealVsFake140k.TRAIN_SIZE*splitFraction),)))
        valset = Subset(valset, indices=torch.randint(0, RealVsFake140k.VALID_SIZE, (int(RealVsFake140k.VALID_SIZE*splitFraction),)))
        testset = Subset(testset, indices=torch.randint(0, RealVsFake140k.TEST_SIZE, (int(RealVsFake140k.TEST_SIZE*splitFraction),)))

        # Dataloaders
        trainLoader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, **dataLoaderKwargs)
        validationLoader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, **dataLoaderKwargs)
        testLoader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, **dataLoaderKwargs)

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
    
    #######################################
    # CONFIG        
    #######################################
    
    # Define transformations
    transform = DEFAULT_INITIAL_TRANSFORM

    BATCH_SIZE = 64
    NUM_EPOCHS = 25
    warmupEpochs = 4
    MOMENTUM = 0.9
    LR = 1e-3

    MODEL_NAME = 'SWIN'
    # Define model
    model = swinModel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    RUNS_DIR_TRAIN = r'CollectedData\Runs\Train'
    RUNS_DIR_VALIDATION = r'CollectedData\Runs\Validation'
    MODELS_DIR = r'CollectedData\Models'


    dataLoaderKwargs = {
        'num_workers': 4,
        'prefetch_factor': 4,
        'pin_memory': True
    }


    # Initialize summary writers to save loss and accuracy during training and validation
    TRAIN_WRITER_PATH = getSaveFileName(rootPath=RUNS_DIR_TRAIN, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, momentum=MOMENTUM, modelName=MODEL_NAME)
    VALIDATION_WRITER_PATH = getSaveFileName(rootPath=RUNS_DIR_VALIDATION, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, momentum=MOMENTUM, modelName=MODEL_NAME)

    MODEL_PATH = getSaveFileName(rootPath=MODELS_DIR, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, momentum=MOMENTUM, modelName=MODEL_NAME)
    trainWriter = SummaryWriter(TRAIN_WRITER_PATH, flush_secs=10)
    validationWriter = SummaryWriter(VALIDATION_WRITER_PATH, flush_secs=10)


    splitFraction = 1
    trainLoader, validationLoader, testLoader = getDataLoaders(splitFraction=splitFraction, dataLoaderKwargs=dataLoaderKwargs)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001) # ADAM IS WASHED, SGD SUPREMACY
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=0.01, nesterov=True)

    # Use LR warmup schedule and reduce learning rate on loss plateu
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-9, end_factor=1, total_iters=warmupEpochs)
    plateuScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, threshold=1e-2, cooldown=1)

    scheduler = WarmupPlataeuScheduler(warmup=warmup, plateauScheduler=plateuScheduler)

    startTime = time.time()

    currentTrainBatch = 0
    currentValidationBatch = 0
    # Training loop
    for epoch in range(NUM_EPOCHS):

        trainLoss, trainAccuracy, trainStats = trainEpoch(model, trainLoader, scheduler, freezeModel=False)
        with torch.no_grad():
            validationLoss, valAccuracy, validationStats = trainEpoch(model, validationLoader, scheduler, freezeModel=True)

        # Do plateau scheduler step based on validation loss instead of train loss so we only reduce lr when validation loss stops improving
        scheduler.step(validationLoss)

        for batch in range(len(trainLoader)):
            batchTrainLoss = trainStats['loss'][batch]
            batchTrainAccuracy = trainStats['accuracy'][batch]
            batchTrainLR = trainStats['lr'][batch]
            trainWriter.add_scalar('trainLoss', batchTrainLoss, currentTrainBatch)
            trainWriter.add_scalar('trainAccuracy', batchTrainAccuracy, currentTrainBatch)
            trainWriter.add_scalar('lr', batchTrainLR, currentTrainBatch)
            currentTrainBatch += 1

        for batch in range(len(validationLoader)):
            batchValidationLoss = validationStats['loss'][batch]
            batchValidationAccuracy = validationStats['accuracy'][batch]
            batchValidationLR = validationStats['lr'][batch]
            validationWriter.add_scalar('validationLoss', batchValidationLoss, currentValidationBatch)
            validationWriter.add_scalar('validationAccuracy', batchValidationAccuracy, currentValidationBatch)
            validationWriter.add_scalar('lr', batchValidationLR, currentValidationBatch)
            currentValidationBatch += 1


        print(f"Epoch {epoch+1}, Train Loss: {trainLoss/len(trainLoader)}, Validation Loss: {validationLoss/len(trainLoader)}, Train Accuracy: {trainAccuracy}, Validation Accuracy: {valAccuracy}")

    print("Finished Training")
    print(f'Train time was {time.time() - startTime}')

    torch.save(model, MODEL_PATH)

    with torch.no_grad():
        testLoss, testAccuracy, testStats = trainEpoch(model, testLoader, scheduler, freezeModel=True)

    print(f'Accuracy of the network on the {int(RealVsFake140k.TEST_SIZE*splitFraction)} test images: {100 * testAccuracy}%, loss was {testLoss}')
    

# Do this because pytorch gets mad when num_workers > 0 and there isn't a main guard
if __name__ == '__main__':
    main()
    
