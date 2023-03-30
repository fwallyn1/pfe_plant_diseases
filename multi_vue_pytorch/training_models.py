from train_multiview_network import TrainMultiInputModel
from multi_view_classification_network import MultiInputModel
from config import get_config
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from torchvision import datasets
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptflops as ptf
import os


########################### SCRIPT FOR TRAINING MULTIVIEW MODELS ##########################

model_config, data_config = get_config()

trainer = TrainMultiInputModel(model_config,data_config)

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
                                       'features' : modified_resnet(torchvision.models.resnet50(weights='IMAGENET1K_V1')),
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

for modelR in models :
    for modelV in models :
        save_file = f'model_recto_{modelR["model_name"]}_model_verso_{modelV["model_name"]}_epochs_30_lr_0.001_batch_size_32_pretrained_True_fine_tune_False.pth'
        #if save_file not in os.listdir('/home/onyxia/work/pfe-deep-learning-maladies-plantes/multi_vue_pytorch/outputs'):
        model_config = {'learning_rate' : 0.001,
                        "epochs" : 30,
                        "pretrained":True,
                        "fine_tune":False,
                        "model_recto":modelR,
                        "model_verso":modelV
                        }
        trainer = TrainMultiInputModel(model_config,data_config)
        trainer.train_and_validate()