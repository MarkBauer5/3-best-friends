import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from datasets import RealVsFake140k, DEFAULT_INITIAL_TRANSFORM
from tqdm import tqdm
import timm
import torch.optim as optim
from models import swinModel, vitModel



class ReduceLROnPlateauWrapper(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_loss = float('inf')

    def step(self, loss=None):
        if loss is None:
            loss = self.last_loss
        self.last_loss = loss
        super().step(loss)



def trainEpoch(dataloader: DataLoader):
    
    model.train()
    runningLoss = 0.0
    currentLr = None
    
    pbar = tqdm(range(len(dataloader)))
    for batchNum in pbar:
        
        optimizer.zero_grad()

        images, labels = next(iter(trainLoader))
        images, labels = images.to(device), labels.to(device)

        # Forward pass and calculate loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # If
        if epoch < warmupEpochs:
            currentLr = warmup.get_last_lr()[0]
        else:
            currentLr = plateuScheduler.optimizer.param_groups[0]['lr']  # Directly get LR from optimizer used in plateuScheduler

        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0) # Clip gradients after calculating loss
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        runningLoss += loss.item()
        
        pbar.set_description("loss: {:.6f}, lr: {:.6f}".format(loss, currentLr), refresh=True)

    if epoch < warmupEpochs:
        warmup.step()
    else:
        plateuScheduler.step(runningLoss)
    
    
    
    return totalLoss / N, correct / N



# Define transformations
# REMEMBER TO FUCKING NORMALIZE THIS SHIT OR IT WILL BE HYPER MEGA ASS
transform = DEFAULT_INITIAL_TRANSFORM

DATA_PATH = r'datasets\\tempDataset'
BATCH_SIZE = 64
NUM_EPOCHS = 10
warmupEpochs = 4

# TODO: Make this an actual train/val/test training setup. We should set learning rate based on validation loss, not train loss.
# Download and prepare datasets
trainset = RealVsFake140k(transform=transform, split='train')
valset =  RealVsFake140k(transform=transform, split='valid')

trainset = Subset(trainset, indices=torch.randint(0, 100000, (640,)))
valset = Subset(valset, indices=torch.randint(0, 20000, (4000,)))

# Dataloaders
trainLoader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validationLoader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)



# Load a pre-trained Swin Transformer model and modify the classifier
# backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
backbone = timm.create_model('vit_base_patch16_224', pretrained=False)

model = vitModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001) # ADAM IS WASHED, SGD SUPREMACY
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.01, nesterov=True)



# Use LR warmup schedule and reduce learning rate on loss plateu
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-9, end_factor=1, total_iters=warmupEpochs)
plateuScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, threshold=1e-2, cooldown=1)

# TODO: See if we can use a chained scheduler to connect these two, I forget if this was possible or not
chainedScheduler = optim.lr_scheduler.ChainedScheduler([warmup, plateuScheduler])

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(range(len(trainLoader)))
    for batchNum in pbar:
        
        images, labels = next(iter(trainLoader))
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if epoch < warmupEpochs:
            currentLr = warmup.get_last_lr()[0]
        else:
            currentLr = plateuScheduler.optimizer.param_groups[0]['lr']  # Directly get LR from optimizer used in plateuScheduler

        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0) # Clip gradients after calculating loss
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        running_loss += loss.item()
        
        pbar.set_description("loss: {:.6f}, lr: {:.6f}".format(loss, currentLr), refresh=True)

    if epoch < warmupEpochs:
        warmup.step()
    else:
        plateuScheduler.step(running_loss)



    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainLoader)}")

print("Finished Training")

correct = 0
total = 0
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for images, labels in validationLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')