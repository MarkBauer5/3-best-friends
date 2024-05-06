import torch
from torch.utils.data import DataLoader
from modelUtils import getDataLoaders
from torch import nn
import torchvision.transforms.v2 as v2

from tqdm import tqdm


"""
Load and evaluate a pretrained model on the test data
"""

def main():

    criterion = nn.CrossEntropyLoss()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    def testEpoch(model:nn.Module, dataloader:DataLoader):
        
        with torch.no_grad():
            
            model.eval()
            running_loss = 0.0                
            correct = 0
            total = 0
                    
            pbar = tqdm(range(len(dataloader)))
            for batchNum, (images, labels) in zip(pbar, dataloader):
                
                images:torch.Tensor; labels:torch.Tensor
                images, labels = images.to(device), labels.to(device)


                outputs = model(images)
                loss = criterion(outputs, labels)
                loss: torch.Tensor

                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = correct/total
                
                pbar.set_description("loss: {:.4f}, accuracy: {:.4f}".format(running_loss/(batchNum+1), accuracy), refresh=True)

        return running_loss, accuracy


    # model, IM_SIZE = torch.load(r'CollectedData\Models\customResnet-Aug-0_Epoch25_Batch1_LR0.001_Momentum0.9'), (224, 224) # RCNN
    # model, IM_SIZE = torch.load(r'CollectedData\Models\superSepNetLarge-NAdam-Aug-0_Epoch25_Batch32_LR0.0001_Momentum0.9'), (224, 224) # SepNet
    model, IM_SIZE = torch.load(r'CollectedData\Models\VisualizableVIT-0_Epoch25_Batch64_LR0.001_Momentum0.9'), (224, 224) # ViT
    # model, IM_SIZE = torch.load(r'CollectedData\Models\VisualizableSWIN-Contd-0_Epoch25_Batch32_LR0.001_Momentum0.9'), (256, 256) # SWIN

    dataLoaderKwargs = {
        'batch_size': 32,
        'num_workers': 4,
        'prefetch_factor': 4,
        'pin_memory': True
    }


    TRAIN_TRANSFORM_AUG = v2.Compose([
        v2.Resize(IM_SIZE),  # Resize images to fit Swin Transformer input dimensions
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(),
        v2.RandomResizedCrop(size=224, scale=(0.7, 1)),
        v2.RandomGrayscale(p=0.05),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    ])
    
    randomCenterCrop = v2.RandomApply(transforms=[v2.CenterCrop(160)], p=1)

    VALTEST_TRANSFORM = v2.Compose([
            v2.Resize(IM_SIZE),  # Resize images to fit Swin Transformer input dimensions
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # randomCenterCrop,
            v2.Resize(IM_SIZE),  # Resize images to fit Swin Transformer input dimensions
            # v2.RandomGrayscale(p=1),
            # v2.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.2),
            # v2.GaussianBlur(kernel_size=17, sigma=(1, 1)),
            v2.RandomAdjustSharpness(sharpness_factor=10, p=1),
        ])

    _, _, testDataloader = getDataLoaders(dataLoaderKwargs=dataLoaderKwargs, trainTransform=TRAIN_TRANSFORM_AUG, valTestTransform=VALTEST_TRANSFORM)

    runningLoss, accuracy = testEpoch(model, testDataloader)
    
    print(f'Finished with accuracy of {round(accuracy,4)*100}%')
    
    
# Do this because pytorch gets mad when num_workers > 0 and there isn't a main guard
if __name__ == '__main__':
    main()