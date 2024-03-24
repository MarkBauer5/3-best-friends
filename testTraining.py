import torch, os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.v2 as v2
from datasets import RealVsFake140k, DEFAULT_INITIAL_TRANSFORM
from tqdm import tqdm

# Define transformations
# REMEMBER TO FUCKING NORMALIZE THIS SHIT OR IT WILL BE HYPER MEGA ASS
transform = DEFAULT_INITIAL_TRANSFORM

DATA_PATH = r'datasets\\tempDataset'
BATCH_SIZE = 64
NUM_EPOCHS = 3


# Download and prepare datasets
trainset = RealVsFake140k(transform=transform, split='train')
valset =  RealVsFake140k(transform=transform, split='valid')

# trainset = Subset(trainset, indices=torch.randint(0, 100000, (4000,)))
# valset = Subset(valset, indices=torch.randint(0, 20000, (4000,)))

# Dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valLoader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

import timm
import torch.nn as nn

# Load a pre-trained Swin Transformer model and modify the classifier
backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
# model.head = nn.Linear(model.head.in_features, 2)

model = nn.Sequential(
    backbone,
    nn.Linear(in_features=1000, out_features=2)
)

import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop


for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(range(len(trainloader)))
    for batchNum in pbar:
        
        images, labels = next(iter(trainloader))
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        pbar.set_description("loss: {:.6f}".format(loss), refresh=True)


    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

print("Finished Training")

correct = 0
total = 0
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for images, labels in valLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')