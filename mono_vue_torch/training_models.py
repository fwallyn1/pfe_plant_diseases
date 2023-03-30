from train_monoview_network import TrainMonoInputModel
from mono_view_classification_network import MonoInputModel
from config import get_config

import torch
from tqdm import tqdm
from torchvision import datasets
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CUDA_LAUNCH_BLOCKING=1

model_config, data_config = get_config()

#trainer = TrainMonoInputModel(model_config,data_config)
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def modified_resnet(model):
    model.fc = Identity()
    return model

def modified_densenet(model):
    model.classifier = Identity()
    return model 

import torchvision
models = [{'model':torchvision.models.efficientnet_b5(weights='IMAGENET1K_V1'),
                                      'avgpool':torch.nn.AdaptiveAvgPool2d(1),
                                      'last_layer_size':2048,
                                      'model_name' : 'EfficientNet',
                                       'features' : None
          },
          {'model':torchvision.models.densenet121(weights='IMAGENET1K_V1'),
                                  'avgpool':None,
                                  'last_layer_size':1024,
                                  'model_name' : 'DenseNet121',
                                   'features' : modified_densenet(torchvision.models.densenet121(weights='IMAGENET1K_V1')),
            },
          {'model':torchvision.models.resnet50(weights='IMAGENET1K_V1'),
                                  'avgpool':None,
                                  'last_layer_size':2048,
                                  'model_name' : 'ResNet50',
                                   'features' : modified_resnet(torchvision.models.resnet50(weights='IMAGENET1K_V1'))
            },
          {'model':torchvision.models.vgg16(weights='IMAGENET1K_V1'),
                                  'avgpool':torch.nn.AdaptiveAvgPool2d((1,1)),
                                  'last_layer_size':512,
                                  'model_name' : 'VGG',
                                   'features' : None
            },
          
          {'model':torchvision.models.resnet152(weights='IMAGENET1K_V1'),
                                  'avgpool':None,
                                  'last_layer_size':2048,
                                  'model_name' : 'ResNet152',
                                   'features' : modified_resnet(torchvision.models.resnet152(weights='IMAGENET1K_V1'))
            }
          
]
import os


for model in models :
    torch.cuda.empty_cache()
    model_config = {'learning_rate' : 0.001,
                      "epochs" : 30,
                      "pretrained":True,
                      "fine_tune":False,
                        "model":model,
                     }
    trainer = TrainMonoInputModel(model_config,data_config,pretrained=True,fine_tune=False)
    save_file = f"model_{model['model_name']}_mono_{trainer.side}_epochs_{trainer.epochs}_lr_{trainer.lr}_batch_size_{trainer.batch_size}_pretrained_{trainer.pretrained}_fine_tune_{trainer.fine_tune}.pth"
    if save_file not in os.listdir('/home/onyxia/work/pfe-deep-learning-maladies-plantes/mono_vue_torch/outputs'):
        print(model['model_name'])
        trainer.train_and_validate()


model_config, data_config = get_config(side="VERSO")
for model in models :
    torch.cuda.empty_cache()
    model_config = {'learning_rate' : 0.001,
                      "epochs" : 30,
                      "pretrained":True,
                      "fine_tune":False,
                        "model":model,
                     }
    trainer = TrainMonoInputModel(model_config,data_config,pretrained=True,fine_tune=False)
    save_file = f"model_{model['model_name']}_mono_{trainer.side}_epochs_{trainer.epochs}_lr_{trainer.lr}_batch_size_{trainer.batch_size}_pretrained_{trainer.pretrained}_fine_tune_{trainer.fine_tune}.pth"
    if save_file not in os.listdir('/home/onyxia/work/pfe-deep-learning-maladies-plantes/mono_vue_torch/outputs'):
        print(model['model_name'])
        trainer.train_and_validate()