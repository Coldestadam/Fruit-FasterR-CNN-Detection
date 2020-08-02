import xml.etree.ElementTree as ET
import glob

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

import vision.references.detection.transforms as T

#Class to create the Datasets
class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms
        
        #Image and XML file directories
        self.imgs_list = sorted(glob.glob("{}/*.jpg".format(path)))
        self.XML_list = sorted(glob.glob("{}/*.xml".format(path)))
        
    def __getitem__(self, idx):
        img_path = self.imgs_list[idx]
        XML_path = self.XML_list[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        #Uses the get_target function to get the labels and boxes
        target = get_target(XML_path)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.imgs_list)
    
# Function to get the labels and boxes of each XML file
def get_target(path):
        
    #Reads XML File
    tree = ET.parse(path)
    root = tree.getroot()
        
    boxes = []
    labels = [] 
    
    for i in range(len(root)):
        # Getting all boxes and labels for every object in image
        
        if root[i].tag == 'object':
            for j in range(len(root[i])):
                if root[i][j].tag == 'name':
                    label = root[i][j].text
                    if label == 'apple':
                        labels.append(1)
                    elif label == 'banana':
                        labels.append(2)
                    elif label == 'orange':
                        labels.append(3)
                    else:
                        labels.append(None)
                            
                elif root[i][j].tag == 'bndbox':
                    x1 = int(root[i][j][0].text)
                    y1 = int(root[i][j][1].text)
                    x2 = int(root[i][j][2].text)
                    y2 = int(root[i][j][3].text)

                    box = [x1, y1, x2, y2]
                    boxes.append(box)
                        
    target = {}
    target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
    target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
    
            
    return target


def get_transforms(train):
    transforms = []
    
    transforms.append(T.ToTensor())

    if train == True:
        transforms.append(T.RandomHorizontalFlip(0.5))
        
    return T.Compose(transforms)


def collate_fn(batch):
    images = []
    targets = []
    
    for sample in batch:
        image = sample[0]
        target = sample[1]
        
        images.append(image)
        targets.append(target)
    
    return images, targets


def getDataloaders(train_path, test_path, train_val_split=0.8, train_batch_size=1, test_batch_size=1):
    
    #Reading Training Data
    train_dataset = FruitDataset(train_path, get_transforms(train=True))
    
    #Splitting Training Data into training and validation sets
    train_count = int(len(train_dataset) * train_val_split)
    val_count = int(len(train_dataset) - train_count)
    train_dataset, val_dataset = random_split(train_dataset, [train_count, val_count])

    #Reading Testing Data
    test_dataset = FruitDataset(test_path, get_transforms(train=False))

    # Get Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader, test_dataloader