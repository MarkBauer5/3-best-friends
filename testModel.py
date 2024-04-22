import torch
from torch.utils.data import DataLoader
from modelUtils import getDataLoaders
from torch import nn

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



    model = torch.load(r'CollectedData\Models\visualizableVIT-0_Epoch25_Batch64_LR0.001_Momentum0.9')

    print(model)

    dataLoaderKwargs = {
        'batch_size': 32,
        'num_workers': 4,
        'prefetch_factor': 4,
        'pin_memory': True
    }

    _, _, testDataloader = getDataLoaders(dataLoaderKwargs=dataLoaderKwargs)

    runningLoss, accuracy = testEpoch(model, testDataloader)
    
    print(f'Finished with accuracy of {accuracy}')
    
    
# Do this because pytorch gets mad when num_workers > 0 and there isn't a main guard
if __name__ == '__main__':
    main()