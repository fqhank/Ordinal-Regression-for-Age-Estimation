import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import glob
from torchvision.io import read_image
import PIL.Image as Image
from torchvision.transforms.functional import normalize
from torchvision.transforms.transforms import RandomRotation, ToTensor

class AgeDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.num_imgs = len(glob.glob(path+'\*\*'))
        self.img_list = glob.glob(path+'\*\*')

    def __len__(self):
        return self.num_imgs

    def __getitem__(self,idx):
        img = Image.open(self.img_list[idx])
        temp_list = self.img_list[idx].split('\\')
        age = int(temp_list[-2])
        label = torch.zeros(72-15,2)
        label[:age-15] = torch.tensor([1,0])
        label[age-15:] = torch.tensor([0,1])
        img = (transforms.ToTensor()(img)-0.5)*2
        return img,label,age