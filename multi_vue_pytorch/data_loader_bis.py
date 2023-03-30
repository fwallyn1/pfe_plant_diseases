import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import random
data_config={'ROOT_DIR' : '/home/onyxia/work/data/Images_ROI/Images_ROI_Typ_for_multiview_echant/',
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
    
    def get_img_path(self,idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        return img_loc
    
    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image

class MultiInputLoader():
    """
    Class for loading the two views data
    """
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
        Returns datasets for each view for purpose : train/validation/test
        """
        if self.test_split + self.valid_split == 0:
            datasetRecto = CustomDataSet(
                self.root_dir + 'RECTO', 
                transform=(self.get_test_transform())
            )
            datasetVerso = CustomDataSet(
                self.root_dir + 'VERSO', 
                transform=(self.get_test_transform())
            )
            return datasetRecto, datasetVerso

        datasetRecto = datasets.ImageFolder(
            self.root_dir + 'RECTO', 
            transform=(self.get_train_transform())
        )
        datasetVerso = datasets.ImageFolder(
            self.root_dir + 'VERSO', 
            transform=(self.get_train_transform())
        )
        datasetRecto_valid = datasets.ImageFolder(
            self.root_dir+'RECTO', 
            transform=(self.get_valid_transform())
        )

        datasetVerso_valid = datasets.ImageFolder(
            self.root_dir+'VERSO', 
            transform=(self.get_valid_transform())
        )

        datasetRecto_test = datasets.ImageFolder(
            self.root_dir+'RECTO', 
            transform=(self.get_test_transform())
        )
        datasetVerso_test = datasets.ImageFolder(
            self.root_dir+'VERSO', 
            transform=(self.get_test_transform())
        )

        # Radomize the data indices.
        train_val_idx, test_idx = train_test_split(np.arange(len(datasetRecto)),
                                             test_size=self.test_split,
                                             random_state=0,
                                             shuffle=True,
                                             stratify=datasetRecto.targets)

        train_idx, val_idx = train_test_split(train_val_idx,
                                             test_size=self.valid_split,
                                             random_state=0,
                                             shuffle=True,
                                             stratify= np.array(datasetRecto.targets)[train_val_idx])
        # Training and validation sets.
        #Recto
        datasetRecto_train = Subset(datasetRecto, train_idx)
        datasetRecto_valid = Subset(datasetRecto_valid, val_idx)
        datasetRecto_test = Subset(datasetRecto_test, test_idx)
        #Verso
        datasetVerso_train = Subset(datasetVerso, train_idx)
        datasetVerso_valid = Subset(datasetVerso_valid, val_idx)
        datasetVerso_test = Subset(datasetVerso_test, test_idx)

        return datasetRecto_train, datasetRecto_valid, datasetVerso_train, datasetVerso_valid, datasetRecto.classes, datasetRecto_test, datasetVerso_test


    def get_data_loaders(self):
        """
        Prepares the training and validation data loaders.
        :param dataset_train: The training dataset.
        :param dataset_valid: The validation dataset.
        Returns the training and validation data loaders.
        """
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        if self.test_split + self.valid_split == 0:
            datasetRecto, datasetVerso = self.get_datasets()
            Recto_loader = DataLoader(
            datasetRecto, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers,
                worker_init_fn=seed_worker,generator=g
            )
            Verso_loader = DataLoader(
            datasetVerso, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers,
            worker_init_fn=seed_worker,generator=g
            )
            return Recto_loader, Verso_loader
        
        datasetRecto_train, datasetRecto_valid, datasetVerso_train, datasetVerso_valid, dataset_class, datasetRecto_test, datasetVerso_test = self.get_datasets()
        trainRecto_loader = DataLoader(
            datasetRecto_train, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers,
            worker_init_fn=seed_worker,generator=g
        )
        validRecto_loader = DataLoader(
            datasetRecto_valid, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers,
            worker_init_fn=seed_worker,generator=g
        )
        trainVerso_loader = DataLoader(
            datasetVerso_train, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers,
            worker_init_fn=seed_worker,generator=g
        )
        validVerso_loader = DataLoader(
            datasetVerso_valid, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers,
            worker_init_fn=seed_worker,generator=g
        )
        testRecto_loader = DataLoader(
            datasetRecto_test, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers,
            worker_init_fn=seed_worker,generator=g
        )
        testVerso_loader = DataLoader(
            datasetVerso_test, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers,
            worker_init_fn=seed_worker,generator=g
        )
        return trainRecto_loader, validRecto_loader, trainVerso_loader, validVerso_loader, testRecto_loader, testVerso_loader
