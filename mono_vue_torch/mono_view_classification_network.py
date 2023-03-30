import torch
import torch.nn as nn
torch.manual_seed(0)
torch.cuda.manual_seed(0)

class MonoInputModel(nn.Module):
    """
    Define Mono Input model : 
    - Feature extraction with CNN
    - Classifer
    """
    def __init__(self, model, pretrained=True, fine_tune=False, num_classes=7):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        super().__init__()
        if pretrained:
            print('[INFO]: Loading pre-trained weights')
        else:
            print('[INFO]: Not loading pre-trained weights')
        self.name = model["model_name"]
        self.model_features = model['model'].features if not model['features'] else model['features']
        self.avg_pool = model['avgpool']
        self.last_layer_size = model['last_layer_size']
        self.classifier = nn.Linear(self.last_layer_size,num_classes)
        self.drop = nn.Dropout(0.2)
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in self.model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in self.model_features.parameters():
                params.requires_grad = False
    
    def forward(self, inputs):
        x = self.model_features(inputs)
        x = self.avg_pool(x) if self.avg_pool else x
        x = torch.flatten(x,1)
        x = self.drop(x)
        x_fc = self.classifier(x)
        return x_fc