import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if opt.phase == 'test':
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
            self.A_paths = make_dataset(self.dir_A)
            self.A_paths = sorted(self.A_paths)
            self.A_size = len(self.A_paths)
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

            self.A_paths = make_dataset(self.dir_A)
            self.B_paths = make_dataset(self.dir_B)

            self.A_paths = sorted(self.A_paths)
            self.B_paths = sorted(self.B_paths)
            self.A_size = len(self.A_paths)
            self.B_size = len(self.B_paths)

        
    def __getitem__(self, index):
        if self.opt.phase == 'test':
            A_path = self.A_paths[index % self.A_size] #H&E
            A_img = Image.open(A_path).convert('RGB')
            A = transforms.ToTensor()(A_img)
            A = transforms.Resize((1024,1024))(A)
            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            return {'A': A, 'A_paths': A_path}
        else:
            A_path = self.A_paths[index % self.A_size] #H&E
            B_path = self.B_paths[index % self.B_size] #IHC
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            A = transforms.ToTensor()(A_img)
            B = transforms.ToTensor()(B_img)
            A = transforms.Resize((1024,1024))(A)
            B = transforms.Resize((1024,1024))(B)
            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
      
            return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
