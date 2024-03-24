import torch, os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import csv


# TODO: Somehow convert these all to a TensorDataset so the GPU runs faster.

# TODO: Make this a default wrapper transform or something to easily be applied to all datasets.
#   This should let us modify the default transform for each dataset in one spot while still allowing
#   custom transforms on individual datasets if needed.

# TODO: Find a way to merge all these different datasets together, we really want one big dataset
#   comprised of all the individual datasets.

# TODO: FUCKING NORMALIZE PER DATASET DURING TRAINING
DEFAULT_INITIAL_TRANSFORM = v2.Compose([
    v2.Resize((224, 224)),  # Resize images to fit Swin Transformer input dimensions
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
    ]
)


"""
Data should have label 1 if it's real, 0 if fake. 

BASIC INTERFACE:
"""

class ExampleDatasetInterface(Dataset):
    
    """
    A special Dataset class for reading the <DATASET>
    """
    
    DATA_PATH = r'PATH'
    LABELS_PATH = r'PATH'
    
    def __init__(self, transform=None, split='train') -> None:
        super().__init__()
        
        self.transform = transform
        self.dataPath = ExampleDatasetInterface.DATA_PATH
        
        assert split in ['train', 'valid', 'test'], 'You dense fucking dumbass, you stupid fucking cretin. (Check split name)'
        self.split = split
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> list[tuple]:

        """
        Loads dataset annotations specific to the current dataset
        
        Returns:
            annotations: A list of tuples in the form (imagePath, label)
        """
        annotations = None
        return annotations

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:

        imagePath, label = self.annotations[index]
        image = Image.open(imagePath).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        return image, label










class LFWDataset(Dataset):
    
    """
    A special Dataset class for reading the LFW celebrities dataset
    """
    
    DATA_PATH = r'datasets\\lfw - Real'
    
    def __init__(self, transform:v2=None, split='train'):
        
        """
        Initializes the LFW Dataset.
        
        Arguments:
            transform: The transform used on all samples from this dataset.
        """
        
        super().__init__()
        self.dataPath = LFWDataset.DATA_PATH
        self.transform = transform
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> list[tuple]:

        """
        Loads dataset annotations specific to the current dataset
        
        Returns:
            annotations: A list of tuples in the form (imagePath, label)
        """
        
        annotations = []
        
        for root, _, files in os.walk(self.dataPath):
            for file in files:
                if file.endswith('.jpg'):
                    image_path = os.path.join(root, file)
                    label = 1 # All data samples are real here
                    annotations.append((image_path, label))
                    
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        
        image_path, label = self.annotations[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    
    
    

class RealVsFake140k(Dataset):
    
    """
    A special Dataset class for reading the 140k Real vs Fake dataset
    """
    
    DATA_PATH = r'datasets\140k Real vs Fake\real_vs_fake\real-vs-fake'
    LABELS_PATH = r'datasets\140k Real vs Fake'
    VALID_SPLITS = ['train', 'valid', 'test']
    
    def __init__(self, transform=None, split='train') -> None:
        super().__init__()
        
        self.transform = transform
        self.dataPath = RealVsFake140k.DATA_PATH
        
        assert split in self.VALID_SPLITS, f'You dense fucking dumbass, you stupid fucking cretin. Split {split} not in {self.VALID_SPLITS}'
        self.split = split
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> list[tuple]:

        """
        Loads dataset annotations specific to the current dataset
        
        Returns:
            annotations: A list of tuples in the form (imagePath, label)
        """

        csvFileName = os.path.join(self.LABELS_PATH, self.split+'.csv')
        
        dataDict = {}
        with open(csvFileName, mode='r') as f:
            dataDict = csv.DictReader(f)
            annotations = [(os.path.normpath(os.path.join(self.DATA_PATH, line['path'])), int(line['label'])) for line in dataDict]
            f.close()

            return annotations

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:

        imagePath, label = self.annotations[index]
        image = Image.open(imagePath).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    



class RealVsFake2k(Dataset):
    
    """
    A special Dataset class for reading the 2k Real vs Fake Small dataset
    """
    
    DATA_PATH = r'datasets\2k Real vs Fake Small\real_and_fake_face_detection'
    VALID_SPLITS = ['train']
    
    def __init__(self, transform=None, split='train') -> None:
        super().__init__()
        
        self.transform = transform
        self.dataPath = RealVsFake2k.DATA_PATH
        
        assert split in self.VALID_SPLITS, f'You dense fucking dumbass, you stupid fucking cretin. Split {split} not in {self.VALID_SPLITS}'
        self.split = split
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> list[tuple]:

        """
        Loads dataset annotations specific to the current dataset
        
        Returns:
            annotations: A list of tuples in the form (imagePath, label)
        """
        
        annotations = []
        
        for root, _, files in os.walk(self.dataPath):
            for file in files:
                if not file.endswith('.jpg'):
                    continue
                
                # Fake is 0. Real is 1.
                if root.split('\\')[-1] == 'training_fake':
                    annotations.append((os.path.join(root, file), 0))

                elif root.split('\\')[-1] == 'training_real':
                    annotations.append((os.path.join(root, file), 1))

        return annotations

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:

        imagePath, label = self.annotations[index]
        image = Image.open(imagePath).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        return image, label


    
    
def main():
    
    # Debug dataset initalization and setup here 
    newTest = RealVsFake2k()
    
    print('SWAGA')
    
    
if __name__ == '__main__':
    main()