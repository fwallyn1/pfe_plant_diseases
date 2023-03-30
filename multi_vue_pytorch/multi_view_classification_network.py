import torch
import torch.nn as nn
from typing import Dict, List
torch.manual_seed(0)
torch.cuda.manual_seed(0)
class MultiInputModel(nn.Module):
    """
    Multi input model (need to be adapted for more that two views)
    """
    def __init__(self,model_recto:Dict,model_verso:Dict,pretrained=True, fine_tune=False, num_classes=7) -> None:
        """
        Constructs Two Input model with two model dictionnary, like those in config.py

        Args:
            model_recto (Dict): model recto description
            model_verso (Dict): model verso description
            pretrained (bool, optional): pretrained model or not. Defaults to True.
            fine_tune (bool, optional): fine tune during training or not. Defaults to False.
            num_classes (int, optional): number of classes to predict. Defaults to 7.
        """
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        super().__init__()
        if pretrained:
            print('[INFO]: Loading pre-trained weights')
        else:
            print('[INFO]: Not loading pre-trained weights')
        #self.modelRecto = model_recto['model']
        self.avg_pool_recto = model_recto['avgpool']
        self.last_layer_size_recto = model_recto['last_layer_size']
        self.modelRecto_name = model_recto['model_name']
        self.modelRecto_features = model_recto['model'].features if not model_recto['features'] else model_recto['features']
                                               
        #self.modelVerso = model_verso['model']
        self.avg_pool_verso = model_verso['avgpool']
        self.modelVerso_name = model_verso['model_name']
        self.modelVerso_features = model_verso['model'].features if not model_verso['features'] else model_verso['features']
        
        self.last_layer_size_verso = model_verso['last_layer_size']
        self.drop = nn.Dropout(0.2)
        self.name = f"{self.modelRecto_name}_{self.modelVerso_name}"
        self.classifier = nn.Linear(self.last_layer_size_recto+self.last_layer_size_verso,num_classes)
        #self.classifier = nn.Sequential(nn.Linear(self.last_layer_size_recto+self.last_layer_size_verso,1024), 
                       #                 nn.ReLU(), nn.Dropout(0.2),nn.Linear(1024,num_classes))
        if fine_tune:
            print('[INFO]: Fine-tuning last features layers...')
            for params in self.modelRecto_features.parameters():
                params.requires_grad = True
            for params in self.modelVerso_features.parameters():
                params.requires_grad = True
            #self.modelRecto_features[-1].requires_grad_(requires_grad=True)
            #self.modelVerso_features[-3:].requires_grad_(requires_grad=True)
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in self.modelRecto_features.parameters():
                params.requires_grad = False
            for params in self.modelVerso_features.parameters():
                params.requires_grad = False
            
    def forward(self, two_views:List[torch.Tensor]):
        """
        Args:
            two_views (List[torch.Tensor]): Lis of the two images tensors

        Returns:
            _type_: _description_
        """
        recto = two_views[0]
        verso = two_views[1]
        x1 = self.modelRecto_features(recto) 
        x1 = self.avg_pool_recto(x1) if self.avg_pool_recto else x1
        x1 = torch.flatten(x1,1)
        x1 = self.drop(x1)
        x2 = self.modelVerso_features(verso)
        x2 = self.avg_pool_verso(x2) if self.avg_pool_verso else x2
        x2 = torch.flatten(x2,1)
        x2 = self.drop(x2)
        x_cat = torch.cat((x1, x2), dim=1)
        x_fc = self.classifier(x_cat)
        return x_fc