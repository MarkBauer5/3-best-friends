import torch
import torch.nn as nn
import time

from datasets import RealVsFake140k, DEFAULT_INITIAL_TRANSFORM
from collections import defaultdict
from models import *
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from modelUtils import getDataLoaders, getSaveFileName, validateModelIO, profileModel


from tensorboardX import SummaryWriter

# TODO: Find a way to ensure the tensorboard logs can appear on the same graph. 
#   I can't do it easily since there are different numbers of batches in the train and validation sets
#   We could just do logs by epoch but with how few epochs we do they'd look pretty bad.
# TODO: Move the WarmupPlataeuScheduler and trainEpoch() function somewhere else???
# TODO: Make the torch flash attention warning fuck off because it's annoying

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


    #######################################
    # CONFIG        
    #######################################
    
    # MAKE SURE I AM TRUE WHEN WE WANT DATA
    SAVE_STATISTICS = True

    if not SAVE_STATISTICS:
        for _ in range(10):
            print('WARNING: DATA WILL NOT BE SAVED!!!!')
            time.sleep(0.25)

    BATCH_SIZE = 64
    NUM_EPOCHS = 25
    warmupEpochs = 4
    MOMENTUM = 0.9
    LR = 1e-3

    # CHANGE ME IF YOU USE A DIFFERENT MODEL PLEASE
    MODEL_NAME = 'vit'
    # Define model
    model = vitModel
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    # validateModelIO(model)
    # profileModel(model, input_size=(BATCH_SIZE, 3, 224, 224))
    
    RUNS_DIR_TRAIN = r'CollectedData/Runs/Train'
    RUNS_DIR_VALIDATION = r'CollectedData/Runs/Validation'
    MODELS_DIR = r'CollectedData/Models'


    dataLoaderKwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': 4,
        'prefetch_factor': 4,
        'pin_memory': True
    }

    splitFraction = 1
    trainLoader, validationLoader, testLoader = getDataLoaders(splitFraction=splitFraction, dataLoaderKwargs=dataLoaderKwargs)


    # Initialize summary writers to save loss and accuracy during training and validation
    TRAIN_WRITER_PATH = getSaveFileName(rootPath=RUNS_DIR_TRAIN, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, momentum=MOMENTUM, modelName=MODEL_NAME)
    VALIDATION_WRITER_PATH = getSaveFileName(rootPath=RUNS_DIR_VALIDATION, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, momentum=MOMENTUM, modelName=MODEL_NAME)

    MODEL_PATH = getSaveFileName(rootPath=MODELS_DIR, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, momentum=MOMENTUM, modelName=MODEL_NAME)
    if SAVE_STATISTICS:
        trainWriter = SummaryWriter(TRAIN_WRITER_PATH, flush_secs=10)
        validationWriter = SummaryWriter(VALIDATION_WRITER_PATH, flush_secs=10)


    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001) # ADAM IS WASHED, SGD SUPREMACY
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=0.00, nesterov=True)

    # Use LR warmup schedule and reduce learning rate on loss plateu
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-9, end_factor=1, total_iters=warmupEpochs)
    plateuScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, threshold=1e-2, cooldown=1)

    scheduler = WarmupPlataeuScheduler(warmup=warmup, plateauScheduler=plateuScheduler)

    startTime = time.time()

    currentTrainBatch = 0
    currentValidationBatch = 0
    epoch = 0
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
            if SAVE_STATISTICS:
                trainWriter.add_scalar('trainLoss', batchTrainLoss, currentTrainBatch)
                trainWriter.add_scalar('trainAccuracy', batchTrainAccuracy, currentTrainBatch)
                trainWriter.add_scalar('lr', batchTrainLR, currentTrainBatch)
            currentTrainBatch += 1

        for batch in range(len(validationLoader)):
            batchValidationLoss = validationStats['loss'][batch]
            batchValidationAccuracy = validationStats['accuracy'][batch]
            batchValidationLR = validationStats['lr'][batch]
            if SAVE_STATISTICS:
                validationWriter.add_scalar('validationLoss', batchValidationLoss, currentValidationBatch)
                validationWriter.add_scalar('validationAccuracy', batchValidationAccuracy, currentValidationBatch)
                validationWriter.add_scalar('lr', batchValidationLR, currentValidationBatch)
            currentValidationBatch += 1


        print(f"Epoch {epoch+1}, Train Loss: {trainLoss/len(trainLoader)}, Validation Loss: {validationLoss/len(trainLoader)}, Train Accuracy: {trainAccuracy}, Validation Accuracy: {valAccuracy}")

        if batchTrainLR < 1e-7 and epoch >= warmupEpochs:
            print('Learning rate collapsed, ending training!')
            break

    print("Finished Training")
    print(f'Train time was {time.time() - startTime}')

    if SAVE_STATISTICS:
        torch.save(model, MODEL_PATH)

    with torch.no_grad():
        testLoss, testAccuracy, testStats = trainEpoch(model, testLoader, scheduler, freezeModel=True)

    print(f'Accuracy of the network on the {int(RealVsFake140k.TEST_SIZE*splitFraction)} test images: {100 * testAccuracy}%, loss was {testLoss}')
    

# Do this because pytorch gets mad when num_workers > 0 and there isn't a main guard
if __name__ == '__main__':
    main()
    
