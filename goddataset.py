import os
import cv2
import time
import argparse
import easydict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# pip install albumentations
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Flip, Normalize, Resize
# list of transformations
# https://vfdev-5-albumentations.readthedocs.io/en/docs_pytorch_fix/api/augmentations.html

# ====================
# ||   EDA results  ||
# ====================
# EDA results, you can check them in EDA.ipynb
def edaresult(path='god_dataset'):
    classes = os.listdir('god_dataset')
    paths = [os.path.join('god_dataset',cls,path) for cls in classes for path in \
                        os.listdir(os.path.join('god_dataset', cls)) if path.endswith('.jpg')]
    paths.sort()
    cl1 = [path for path in paths if path.split('/')[1] == classes[0]]
    cl2 = [path for path in paths if path.split('/')[1] == classes[1]]
    counts = [len(cl1), len(cl2)]
    return classes, paths, counts

# ====================
# ||     Dataset    ||
# ====================
class GodDataset(Dataset):
    '''
        __init__, __len__, __getitem__ must be implemented
        __init__
                1. initialize the input parameters, the self.xxx syntax pass the 
                   parameter to other attributes.
                2. train/val/text split
                3. label assign
        __len__
                determine the size (len) of this dataset
        __getitem__
                1. transform the data into torch
                2. data augmentation
    '''
    def __init__(self, path='god_dataset', mode='train', val_split=0.3, test_split=0.1, image_size=256, seed=42):
        # assert the assigned mode is only in 'train','val','test'
        assert mode in ['train','val','test']
        
        # must inherit Dataset class
        super().__init__()

        classes, paths, counts = edaresult(path=path)
        self.mode = mode
        self.image_size = image_size
        self.paths  = paths
        self.labels = [0 if path.split('/')[1] == 'god' else 1 for path in self.paths]
        # print(self.labels)
        
        np.random.seed(seed)  # fix the seed for consistent train val test set
        rand_index = np.random.permutation([i for i in range(len(self.paths))])
        self.paths = [self.paths[i] for i in rand_index]
        self.labels= [self.labels[i] for i in rand_index]

        # split data here
        end_train_idx = round(len(self.paths)*(1-val_split-test_split))
        end_val_idx   = round(len(self.paths)*(1-test_split))
        if mode == 'train':
            self.paths = self.paths[:end_train_idx]
            self.labels= self.labels[:end_train_idx]
        elif mode == 'val':
            self.paths = self.paths[end_train_idx:end_val_idx]
            self.labels= self.labels[end_train_idx:end_val_idx]
        else:
            self.paths = self.paths[end_val_idx:]
            self.labels= self.labels[end_val_idx:]

    def __len__(self):
        return len(self.paths)
    
    def train_transfrom(self, image, image_size):
        # simple augmentation
        transform = Compose([
                            Resize(image_size,image_size, interpolation=cv2.INTER_AREA),
                            Flip(),
                            Normalize(mean =[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]),
                            ToTensorV2()
                            ])
        image_transform = transform(image = image)['image']
        return image_transform

    def val_transfrom(self, image, image_size):
        transform = Compose([
                            Resize(image_size,image_size, interpolation=cv2.INTER_AREA),
                            Normalize(mean =[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]),
                            ToTensorV2()
                            ])
        image_transform = transform(image = image)['image']
        return image_transform

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx])
        path  = self.paths[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == 'train':
            image = self.train_transfrom(image, self.image_size)
        else:
            image = self.val_transfrom(image, self.image_size)

        return image, label


# ====================
# ||   Dataloader   ||
# ====================
def datasets(args):
    train_dataset = GodDataset(mode='train', image_size=args.image_size, val_split=args.val_split, 
                                test_split=args.test_split, seed=args.seed)
    val_dataset   = GodDataset(mode='val', image_size=args.image_size, 
                                val_split=args.val_split,test_split=args.test_split, seed=args.seed)
    test_dataset  = GodDataset(mode='test', image_size=args.image_size, 
                                val_split=args.val_split, test_split=args.test_split, seed=args.seed)
    '''
        pin_memory:
                   samples on CPU and like to push it to the GPU during training
                   this speed up. This lets your DataLoader allocate the samples
                   in page-locked memory, which speeds-up the transfer.
        num_workers:
                   each worker loads a single batch and returns it only once it’s ready.
                   some say `num_worker = 4 * num_GPU`, but I set it equal to the 
                   number of threads CPU has, since I cannot feel the difference.
                   In the validation stage, the double of the num_worker speeds up
                   the overall speed. The reason is that GPU does not need to 
                   calculate the gradient anymore, the speed bottleneck is resided
                   in CPU only.
        collate_fn:
                   process the list of samples (original batch) to form a batch.
                   The batch argument is a list with all your samples, and output
                   the batch you really "want" to return from your dataloader.
                   Take an example from object detection,
                   Example:
                            def yolo_dataset_collate(batch):
                                images = []
                                bboxes = []
                                for img, box in batch:
                                    images.append(img)
                                    bboxes.append(box)
                                images = np.array(images)
                                return images, bboxes
                            loaded = DataLoader(dataset, collate_fn = yolo_dataset_collate)
                   See the thread https://discuss.pytorch.org/t/how-to-use-collate-fn/27181 for more detail

    '''
    train_loader = DataLoader(train_dataset, pin_memory=True, num_workers=os.cpu_count(),batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, pin_memory=True, num_workers=2*os.cpu_count(), batch_size=args.batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, pin_memory=True, num_workers=2*os.cpu_count(), batch_size=args.batch_size, shuffle=True)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    dataset = GodDataset(mode='val')
    print(len(dataset))
    # print(next(iter(dataset)))
    # train_loader = DataLoader(dataset, batch_size=20)
    # imgs, labels = next(iter(train_loader))
    # print(imgs)
    # print(labels)
    # classes, paths, counts = edaresult()
    # print(paths)
    # print(classes)
    # print(counts)
