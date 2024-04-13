# import some packages you need here

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from torchvision.transforms import RandomRotation, RandomHorizontalFlip

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):

        # write your codes here
        self.data_dir = data_dir
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)])

    def __len__(self):

        # write your codes here
        return len(self.filenames)

    def __getitem__(self, idx):

        # write your codes here
        img_path = os.path.join(self.data_dir, self.filenames[idx])
        img = Image.open(img_path).convert('L') # convert to gray scale
        img = self.transform(img)
        label = int(self.filenames[idx].split('_')[1].split('.')[0])
        
        return img, label

if __name__ == '__main__':

    # write test codes to verify your implementations
    
    data_dir = '/home/ishwang/hw/ishwang/mnist-classification/data'
    dataset = MNIST(data_dir)
    print("Dataset size :", len(dataset))
    
    img, label = dataset[0]
    print("First image shape:", img.shape)
    print("First image label:", label)


