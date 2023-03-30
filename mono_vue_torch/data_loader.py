import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np


data_config={'ROOT_DIR' : '/home/onyxia/work/data/Images_ROI/Images_ROI_Typ_for_multiview_echant/RECTO',
'VALID_SPLIT' : 0.2,
'TEST_SPLIT' : 0.2,
'IMAGE_SIZE' : 224, # Image size of resize when applying transforms.
'BATCH_SIZE' : 32,
'NUM_WORKERS' : 0}

torch.manual_seed(0)
torch.cuda.manual_seed(0)
class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = sorted(os.listdir(main_dir))
        
    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image

class MonoInputLoader():
    def __init__(self,data_config=data_config,pretrained=True):
        self.root_dir = data_config["ROOT_DIR"]
        self.valid_split = data_config["VALID_SPLIT"]
        self.test_split = data_config["TEST_SPLIT"]
        self.image_size = data_config["IMAGE_SIZE"]
        self.batch_size = data_config["BATCH_SIZE"]
        self.num_workers = data_config["NUM_WORKERS"]
        self.pretrained=pretrained

    def get_train_transform(self):
        train_transform = transforms.Compose([
        transforms.Resize((self.image_size, self.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        self.normalize_transform()
        ])
        return train_transform

    # Validation transforms
    def get_valid_transform(self):
        valid_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            self.normalize_transform()
        ])
        return valid_transform

    # Test transforms
    def get_test_transform(self):
        test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            self.normalize_transform()
        ])
        return test_transform

    # Image normalization transforms.
    def normalize_transform(self):
        if self.pretrained: # Normalization for pre-trained weights.
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        else: # Normalization when training from scratch.
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        return normalize


    def get_datasets(self):
        """
        Function to prepare the Datasets.
        :param pretrained: Boolean, True or False.
        Returns the training and validation datasets along 
        with the class names.
        """
        if self.test_split + self.valid_split == 0:
            dataset = CustomDataSet(
                self.root_dir, 
                transform=(self.get_test_transform())
            )
            return dataset
        
        dataset = datasets.ImageFolder(
            self.root_dir, 
            transform=(self.get_train_transform())
        )
        dataset_valid = datasets.ImageFolder(
            self.root_dir, 
            transform=(self.get_valid_transform())
        )

        dataset_test = datasets.ImageFolder(
            self.root_dir, 
            transform=(self.get_test_transform())
        )
        
        g1 = torch.Generator()
        g1.manual_seed(0)
        dataset_size = len(dataset)
        # Calculate the test dataset size.
        test_size = int(self.test_split*dataset_size)
        # Radomize the data indices.
        # Radomize the data indices.
        train_val_idx, test_idx = train_test_split(np.arange(len(dataset)),
                                             test_size=self.test_split,
                                             random_state=0,
                                             shuffle=True,
                                             stratify=dataset.targets)

        train_idx, val_idx = train_test_split(train_val_idx,
                                             test_size=self.valid_split,
                                             random_state=0,
                                             shuffle=True,
                                             stratify= np.array(dataset.targets)[train_val_idx])
        # Training and validation sets.
        dataset_train = Subset(dataset, train_idx)
        dataset_valid = Subset(dataset_valid, val_idx)
        dataset_test = Subset(dataset_test, test_idx)

        return dataset_train, dataset_valid, dataset.classes, dataset_test


    def get_data_loaders(self):
        """
        Prepares the training and validation data loaders.
        :param dataset_train: The training dataset.
        :param dataset_valid: The validation dataset.
        Returns the training and validation data loaders.
        """
        if self.test_split + self.valid_split == 0:
            dataset= self.get_datasets()
            loader = DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers
            )
            return loader
        
        dataset_train, dataset_valid, dataset_class, dataset_test = self.get_datasets()
        train_loader = DataLoader(
            dataset_train, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers
        )
        valid_loader = DataLoader(
            dataset_valid, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            dataset_test, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers
        )
    
        return train_loader, valid_loader, test_loader