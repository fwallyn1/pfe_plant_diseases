import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_model
from datasets import get_datasets, get_data_loaders
import torchvision.models as models
import torch.nn as nn 
from torch.nn import functional as F
from data_loader_bis import MultiInputLoader
from multi_view_classification_network import MultiInputModel
import matplotlib
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed(0)
class TrainMultiInputModel():
    """
    Class for training and testing MultiInputModel
    """
    
    def __init__(self,model_config,data_config,pretrained=True, fine_tune=False, num_classes=7):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_recto = model_config['model_recto']['model']
        self.model_verso = model_config['model_verso']['model']
        self.model = MultiInputModel(model_config['model_recto'],model_config['model_verso'],pretrained=model_config['pretrained'], fine_tune=model_config['fine_tune'], num_classes=7).to(self.device)
        self.pretrained = pretrained
        self.fine_tune = model_config['fine_tune']
        self.num_classes = num_classes
        self.loader = MultiInputLoader(data_config,pretrained)
        self.trainRectoloader, self.validRectoloader, self.trainVersoloader, self.validVersoloader, self.testRectoloader, self.testVersoloader = self.loader.get_data_loaders()
        self.datasetRecto_train, self.datasetRecto_valid, self.datasetVerso_train, self.datasetVerso_valid, self.dataset_classes, self.datasetRecto_test, self.datasetVerso_test = self.loader.get_datasets()
        self.epochs = model_config['epochs']
        self.lr = model_config['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.main_path = '/home/onyxia/work/pfe-deep-learning-maladies-plantes/multi_vue_pytorch'
        self.batch_size = data_config["BATCH_SIZE"]
        
    def train(self):
        self.model.train()
        print('Training')
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        for i, (recto,verso) in tqdm(enumerate(zip(self.trainRectoloader,self.trainVersoloader)), total=len(self.trainRectoloader)):
            counter += 1
            image_recto, labels = recto
            image_recto = image_recto.to(self.device)
            labels = labels.to(self.device)
            image_verso, _ = verso
            image_verso = image_verso.to(self.device)
            self.optimizer.zero_grad()
            # Forward pass.
            outputs = self.model([image_recto,image_verso])
            # Calculate the loss.
            loss = self.criterion(outputs, labels)
            train_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
            # Backpropagation
            loss.backward()
            # Update the weights.
            self.optimizer.step()
            torch.cuda.empty_cache()

        # Loss and accuracy for the complete epoch.
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(self.trainRectoloader.dataset))
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        print('Validation')
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0
        with torch.no_grad():
            for i, (recto,verso) in tqdm(enumerate(zip(self.validRectoloader,self.validVersoloader)), total=len(self.validRectoloader)):
                counter += 1
                image_recto, labels = recto
                image_recto = image_recto.to(self.device)
                labels = labels.to(self.device)
                image_verso, _ = verso
                image_verso = image_verso.to(self.device)
                # Forward pass.
                outputs = self.model([image_recto,image_verso])
                # Calculate the loss.
                loss = self.criterion(outputs, labels)
                valid_running_loss += loss.item()
                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()

        # Loss and accuracy for the complete epoch.
        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / len(self.validRectoloader.dataset))
        return epoch_loss, epoch_acc
    
    def save_plots(self,train_acc,valid_acc,train_loss,valid_loss):
        """
        Function to save the loss and accuracy plots to disk.
        """
        model_recto_name = self.model.modelRecto_name
        model_verso_name = self.model.modelVerso_name
        # accuracy plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_acc, color='green', linestyle='-', 
            label='train accuracy'
        )
        plt.plot(
            valid_acc, color='blue', linestyle='-', 
            label='validataion accuracy'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f"{self.main_path}/outputs/accuracy_model_recto_{model_recto_name}_model_verso_{model_verso_name}_epochs_{self.epochs}_lr_{self.lr}_batch_size_{self.batch_size}_pretrained_{self.pretrained}_fine_tune_{self.fine_tune}.png")

        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_loss, color='orange', linestyle='-', 
            label='train loss'
        )
        plt.plot(
            valid_loss, color='red', linestyle='-', 
            label='validataion loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{self.main_path}/outputs/loss_pretrained_model_recto_{model_recto_name}_model_verso_{model_verso_name}_epochs_{self.epochs}_lr_{self.lr}_batch_size_{self.batch_size}_pretrained_{self.pretrained}_fine_tune_{self.fine_tune}.png")

    def train_and_validate(self):
        print(f"[INFO]: Number of training images: {len(self.datasetVerso_train)},{len(self.datasetVerso_train)}")
        print(f"[INFO]: Number of validation images: {len(self.datasetRecto_valid)},{len(self.datasetVerso_valid)}")
        print(f"[INFO]: Class names: {self.dataset_classes}\n")
        print(f"Computation device: {self.device}")
        print(f"Learning rate: {self.lr}")
        print(f"Epochs to train for: {self.epochs}\n")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        # Start the training.
        for epoch in range(self.epochs):
            print(f"[INFO]: Epoch {epoch+1} of {self.epochs}")
            train_epoch_loss, train_epoch_acc = self.train()
            valid_epoch_loss, valid_epoch_acc = self.validate()
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-'*50)
            time.sleep(5)

        # Save the trained model weights.
        model_recto_name = self.model.modelRecto_name
        model_verso_name = self.model.modelVerso_name
        torch.save({'epoch': self.epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.criterion,
                    }, f"{self.main_path}/outputs/model_recto_{model_recto_name}_model_verso_{model_verso_name}_epochs_{self.epochs}_lr_{self.lr}_batch_size_{self.batch_size}_pretrained_{self.pretrained}_fine_tune_{self.fine_tune}.pth")
        # Save the loss and accuracy plots.
        self.save_plots(train_acc, valid_acc, train_loss, valid_loss)
        
    def predict (self, recto_data_loader, verso_data_loader):
        """
        :param tensor_normalized_image:
        Example of normalized image:

            pil_img = PIL.Image.open(img_dir)

            torch_img = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(pil_img)

            tensor_normalized_image = tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]


        :return:

        Return

        predict = tensor with prediction (mean distance of query image and support set)
        torch_max [1] = predicted class index

        """
        self.model.eval()
        with torch.no_grad():
            counter=0
            outputs_test_preds= []
            outputs_test_probs = []
            outputs_all_test_probs = []
            real_labels = []
            for i, (recto,verso) in tqdm(enumerate(zip(recto_data_loader,verso_data_loader)), total=len(recto_data_loader)):
                counter += 1
                image_recto = recto[0] if isinstance(recto,list) else recto
                image_recto = image_recto.to(self.device)
                image_verso = verso[0] if isinstance(verso,list) else verso
                image_verso = image_verso.to(self.device)
                # Forward pass.
                outputs_test = self.model([image_recto,image_verso])
                outputs_test_prob = F.softmax(outputs_test,dim=1)
                outputs_all_test_probs.append(outputs_test_prob.tolist())
                probs , preds = torch.max(outputs_test_prob.data, 1)
                outputs_test_preds.append(preds.tolist())
                outputs_test_probs.append(probs.tolist())
            outputs_test_preds = [item for sublist in outputs_test_preds for item in sublist]
            outputs_test_probs = [item for sublist in outputs_test_probs for item in sublist]
            outputs_all_test_probs = [item for sublist in outputs_all_test_probs for item in sublist]
        return outputs_test_preds, outputs_test_probs, outputs_all_test_probs

       