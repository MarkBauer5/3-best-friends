import torch
import torch.nn as nn
import time

from datasets import RealVsFake140k, DEFAULT_INITIAL_TRANSFORM
from collections import defaultdict
from models import *
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from modelUtils import getDataLoaders, getSaveFileName, validateModelIO, profileModel, WarmupPlateauScheduler
import torchvision.transforms.v2 as v2


from tensorboardX import SummaryWriter

# TODO: Find a way to ensure the tensorboard logs can appear on the same graph. 
#   I can't do it easily since there are different numbers of batches in the train and validation sets
#   We could just do logs by epoch but with how few epochs we do they'd look pretty bad.

# Run benchmark to use most efficient convolution
torch.backends.cudnn.benchmark = True

# Kill debuggers for training
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)
USE_AMP = True


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



def trainEpoch(model:nn.Module, dataloader:DataLoader, scheduler:torch.optim.lr_scheduler.ReduceLROnPlateau, freezeModel:bool,
            optimizer, scaler, criterion):
    
    if freezeModel:
        model.eval()
    else:
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
        
        # Use AMP for better train speed
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
            outputs = model(images)                
            loss = criterion(outputs, labels)
        
        if not freezeModel:
            scaler.scale(loss).backward() # Do backpropagation on scaled loss from AMP
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = correct/total
        
        # currentLr = scheduler.getLastLR()
        currentLr = scheduler.optimizer.param_groups[0]['lr']
        
        statsDict['loss'].append(loss.item()) 
        statsDict['lr'].append(currentLr)
        statsDict['accuracy'].append(accuracy)
        
        pbar.set_description("loss: {:.6f}, lr: {:.6f}, accuracy: {:.5f}".format(running_loss/(batchNum+1), currentLr, accuracy), refresh=True)

    return running_loss, accuracy, statsDict





def trainModel(trainingKwargs:dict):
    
    model = trainingKwargs['model']
    MODEL_NAME = trainingKwargs['MODEL_NAME']
    NUM_EPOCHS = trainingKwargs['NUM_EPOCHS']
    BATCH_SIZE = trainingKwargs['BATCH_SIZE']
    LR = trainingKwargs['LR']
    MOMENTUM = trainingKwargs['MOMENTUM']
    TRAIN_TRANSFORM = trainingKwargs['TRAIN_TRANSFORM']
    VALTEST_TRANSFORM = trainingKwargs['VALTEST_TRANSFORM']
    dataLoaderKwargs = trainingKwargs['dataLoaderKwargs']
    splitFraction = trainingKwargs['splitFraction']
    RUNS_DIR_TRAIN = trainingKwargs['RUNS_DIR_TRAIN']
    RUNS_DIR_VALIDATION = trainingKwargs['RUNS_DIR_VALIDATION']
    MODELS_DIR = trainingKwargs['MODELS_DIR']
    SAVE_STATISTICS = trainingKwargs['SAVE_STATISTICS']
    
    if not SAVE_STATISTICS:
        for _ in range(10):
            print('WARNING: DATA WILL NOT BE SAVED!!!!')
            time.sleep(0.25)
    
    model.to(device)
    trainLoader, validationLoader, testLoader = getDataLoaders(splitFraction=splitFraction, dataLoaderKwargs=dataLoaderKwargs, trainTransform=TRAIN_TRANSFORM, valTestTransform=VALTEST_TRANSFORM)

    # Initialize summary writers to save loss and accuracy during training and validation
    TRAIN_WRITER_PATH = getSaveFileName(rootPath=RUNS_DIR_TRAIN, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, momentum=MOMENTUM, modelName=MODEL_NAME)
    VALIDATION_WRITER_PATH = getSaveFileName(rootPath=RUNS_DIR_VALIDATION, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, momentum=MOMENTUM, modelName=MODEL_NAME)

    MODEL_PATH = getSaveFileName(rootPath=MODELS_DIR, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, momentum=MOMENTUM, modelName=MODEL_NAME)
    
    trainWriter = None
    validationWriter = None
    if SAVE_STATISTICS:
        trainWriter = SummaryWriter(TRAIN_WRITER_PATH, flush_secs=10)
        validationWriter = SummaryWriter(VALIDATION_WRITER_PATH, flush_secs=10)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=LR) # ADAM IS WASHED, SGD SUPREMACY
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=0, nesterov=True)

    # Use LR warmup schedule and reduce learning rate on loss plateu
    # warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-9, end_factor=1, total_iters=warmupEpochs)
    plateuScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, threshold=1e-2, cooldown=1)
    scheduler = plateuScheduler
    # scheduler = WarmupPlateauScheduler(warmup=warmup, plateauScheduler=plateuScheduler)

    startTime = time.time()

    currentTrainBatch = 0
    currentValidationBatch = 0
    epoch = 0
    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    print(f'Training model {MODEL_NAME}')
    # Training loop
    for epoch in range(NUM_EPOCHS):

        trainLoss, trainAccuracy, trainStats = trainEpoch(model, trainLoader, scheduler, freezeModel=False, optimizer=optimizer, scaler=scaler, criterion=criterion)
        validationLoss = None; valAccuracy = None; validationStats = None
        with torch.no_grad():
            validationLoss, valAccuracy, validationStats = trainEpoch(model, validationLoader, scheduler, freezeModel=True, optimizer=optimizer, scaler=scaler, criterion=criterion)

        # Do plateau scheduler step based on validation loss instead of train loss so we only reduce lr when validation loss stops improving. Stepping on train loss means we only reduce
        #   lr after we've overfit to the data instead of when we actually need to drop lr to learn better
        # TODO: Maybe consider stepping on validation accuracy instead? Stepping on loss may mean we can overfit since we improve confidence, but not actual accuracy
        scheduler.step(validationLoss)

        # Write statistics to file if needed
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

        if batchTrainLR < 1e-6:
            print('Learning rate collapsed, ending training!')
            break

    print("Finished Training")
    print(f'Train time was {time.time() - startTime}')

    if SAVE_STATISTICS:
        torch.save(model, MODEL_PATH)

    with torch.no_grad():
        testLoss, testAccuracy, testStats = trainEpoch(model, testLoader, scheduler, freezeModel=True, optimizer=optimizer, scaler=scaler, criterion=criterion)

    print(f'Accuracy of the network on the {int(RealVsFake140k.TEST_SIZE*splitFraction)} test images: {100 * testAccuracy}%, loss was {testLoss}')





def main():

    #######################################
    # CONFIG        
    #######################################
    
    # MAKE SURE I AM TRUE WHEN WE WANT DATA
    SAVE_STATISTICS = False

    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    MOMENTUM = 0.9
    LR = 1e-3
    # How much of the dataset to use, 1 for all, 0 for none    
    splitFraction = 1

    dataLoaderKwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': 2,
        'prefetch_factor': 1,
        'pin_memory': True
    }


    # CHANGE ME IF YOU USE A DIFFERENT MODEL PLEASE
    # MODEL_NAME = 'SuperSepNet-Small-Contd'
    # model = superSepNetSmall #VisualizableSWIN()
    # model = torch.load('CollectedData\Models\SuperSepNet-Small-0_Epoch15_Batch128_LR0.001_Momentum0.9')
    # model.to(device)
    
    # Validate model will run and profile each layer's computation cost
    # validateModelIO(model)
    # profileModel(model, input_size=(BATCH_SIZE, 3, 224, 224))
    
    RUNS_DIR_TRAIN = r'CollectedData/Runs/Train'
    RUNS_DIR_VALIDATION = r'CollectedData/Runs/Validation'
    MODELS_DIR = r'CollectedData/Models'

    randomCenterCrop = v2.RandomApply(transforms=[v2.CenterCrop(150)], p=0.1)

    TRAIN_TRANSFORM_AUG = v2.Compose([
        v2.Resize((224, 224)),  # Resize images to fit Swin Transformer input dimensions
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        # v2.RandomAdjustSharpness(sharpness_factor=20, p=1),
        # randomCenterCrop,
        # v2.GaussianBlur(kernel_size=17, sigma=(2, 2)),
        v2.RandomHorizontalFlip(),
        v2.RandomResizedCrop(size=224, scale=(0.7, 1)),
        v2.RandomGrayscale(p=0.05),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        # v2.Resize((224, 224)), # ENABLE ME IF WE USE CENTER CROP
    ])
    
    VALTEST_TRANSFORM = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),  # Resize images to fit Swin Transformer input dimensions
        ])
    
    
    
    BATCH_MODEL_PARAMETERS = [
        # {
        #     'model': superSepNetLarge,
        #     'MODEL_NAME': 'superSepNetLarge-NAdam-Aug',
        #     'NUM_EPOCHS': NUM_EPOCHS,
        #     'BATCH_SIZE': BATCH_SIZE,
        #     'LR': 1e-4,
        #     'MOMENTUM': MOMENTUM,
        #     'TRAIN_TRANSFORM': TRAIN_TRANSFORM,
        #     'VALTEST_TRANSFORM': VALTEST_TRANSFORM,
        #     'dataLoaderKwargs': dataLoaderKwargs,
        #     'splitFraction': splitFraction,
        #     'RUNS_DIR_TRAIN': RUNS_DIR_TRAIN,
        #     'RUNS_DIR_VALIDATION': RUNS_DIR_VALIDATION,
        #     'MODELS_DIR': MODELS_DIR,
        #     'SAVE_STATISTICS': SAVE_STATISTICS
        # },
        {
            'model': customResnet,
            'MODEL_NAME': 'customResnet-Aug-NAdam',
            'NUM_EPOCHS': NUM_EPOCHS,
            'BATCH_SIZE': 4, # For some reason, augmentation eats a TON of VRAM when compared to normal training
            'LR': 1e-4, # Probably need to start this at 1e-2 or even 5e-2 next time.
            'MOMENTUM': MOMENTUM,
            'TRAIN_TRANSFORM': TRAIN_TRANSFORM_AUG,
            'VALTEST_TRANSFORM': VALTEST_TRANSFORM,
            'dataLoaderKwargs': dataLoaderKwargs,
            'splitFraction': splitFraction,
            'RUNS_DIR_TRAIN': RUNS_DIR_TRAIN,
            'RUNS_DIR_VALIDATION': RUNS_DIR_VALIDATION,
            'MODELS_DIR': MODELS_DIR,
            'SAVE_STATISTICS': SAVE_STATISTICS
        },
        # {
        #     'model': VisualizableVIT(),
        #     'MODEL_NAME': 'VisualizableVIT',
        #     'NUM_EPOCHS': NUM_EPOCHS,
        #     'BATCH_SIZE': BATCH_SIZE,
        #     'LR': LR,
        #     'MOMENTUM': MOMENTUM,
        #     'TRAIN_TRANSFORM': TRAIN_TRANSFORM,
        #     'VALTEST_TRANSFORM': VALTEST_TRANSFORM,
        #     'dataLoaderKwargs': dataLoaderKwargs,
        #     'splitFraction': splitFraction,
        #     'RUNS_DIR_TRAIN': RUNS_DIR_TRAIN,
        #     'RUNS_DIR_VALIDATION': RUNS_DIR_VALIDATION,
        #     'MODELS_DIR': MODELS_DIR,
        #     'SAVE_STATISTICS': SAVE_STATISTICS
        # },
        # {
        #     'model': VisualizableSWIN(),
        #     'MODEL_NAME': 'VisualizableSWIN',
        #     'NUM_EPOCHS': NUM_EPOCHS,
        #     'BATCH_SIZE': BATCH_SIZE,
        #     'LR': LR,
        #     'MOMENTUM': MOMENTUM,
        #     'TRAIN_TRANSFORM': v2.Compose([
        #         v2.ToImage(),
        #         v2.ToDtype(torch.float32, scale=True),
        #         v2.Resize((256, 256)),  # Resize images to fit Swin Transformer input dimensions
        #     ]),
        #     'VALTEST_TRANSFORM': v2.Compose([
        #         v2.ToImage(),
        #         v2.ToDtype(torch.float32, scale=True),
        #         v2.Resize((256, 256)),  # Resize images to fit Swin Transformer input dimensions
        #     ]),
        #     'dataLoaderKwargs': dataLoaderKwargs,
        #     'splitFraction': splitFraction,
        #     'RUNS_DIR_TRAIN': RUNS_DIR_TRAIN,
        #     'RUNS_DIR_VALIDATION': RUNS_DIR_VALIDATION,
        #     'MODELS_DIR': MODELS_DIR,
        #     'SAVE_STATISTICS': SAVE_STATISTICS
        # },
        # {
        #     'model': torch.load('CollectedData\Models\VisualizableSWIN-0_Epoch25_Batch64_LR0.001_Momentum0.9'),
        #     'MODEL_NAME': 'VisualizableSWIN-Contd',
        #     'NUM_EPOCHS': NUM_EPOCHS,
        #     'BATCH_SIZE': BATCH_SIZE,
        #     'LR': LR,
        #     'MOMENTUM': MOMENTUM,
        #     'TRAIN_TRANSFORM': v2.Compose([
        #         v2.ToImage(),
        #         v2.ToDtype(torch.float32, scale=True),
        #         v2.Resize((256, 256)),  # Resize images to fit Swin Transformer input dimensions
        #     ]),
        #     'VALTEST_TRANSFORM': v2.Compose([
        #         v2.ToImage(),
        #         v2.ToDtype(torch.float32, scale=True),
        #         v2.Resize((256, 256)),  # Resize images to fit Swin Transformer input dimensions
        #     ]),
        #     'dataLoaderKwargs': dataLoaderKwargs,
        #     'splitFraction': splitFraction,
        #     'RUNS_DIR_TRAIN': RUNS_DIR_TRAIN,
        #     'RUNS_DIR_VALIDATION': RUNS_DIR_VALIDATION,
        #     'MODELS_DIR': MODELS_DIR,
        #     'SAVE_STATISTICS': SAVE_STATISTICS
        # },
    ]
    
    
    
    for trainingKwargs in BATCH_MODEL_PARAMETERS:
        trainModel(trainingKwargs=trainingKwargs)

# Do this because pytorch gets mad when num_workers > 0 and there isn't a main guard
if __name__ == '__main__':
    main()

