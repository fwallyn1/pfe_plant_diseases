from torch import nn, optim
import torchvision

class Identity(nn.Module):
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

def get_config():
        DATA_CONFIG={
                'ROOT_DIR': '/home/onyxia/work/data/Images_ROI/Images_ROI_Typ_for_multiview/',
                'VALID_SPLIT': 0.2,
                'TEST_SPLIT': 0.5,
                'IMAGE_SIZE': 224,  # Image size of resize when applying transforms.
                'BATCH_SIZE': 32,
                'NUM_WORKERS': 0
                }

        MODEL_CONFIG={'learning_rate' : 0.001,
                      "epochs" : 30,
                      "pretrained":True,
                      "fine_tune":False,
                      "model_recto": {'model':torchvision.models.resnet50(weights='IMAGENET1K_V1'),
                                  'avgpool':None,
                                  'last_layer_size':2048,
                                  'model_name' : 'ResNet50',
                                   'features' : modified_resnet(torchvision.models.resnet50(weights='IMAGENET1K_V1'))
                                },
                        "model_verso":{'model':torchvision.models.resnet50(weights='IMAGENET1K_V1'),
                                  'avgpool':None,
                                  'last_layer_size':2048,
                                  'model_name' : 'ResNet50',
                                   'features' : modified_resnet(torchvision.models.resnet50(weights='IMAGENET1K_V1'))
                                },
                     }
        return MODEL_CONFIG,DATA_CONFIG


def get_aty_data_config():
        DATA_CONFIG={'ROOT_DIR' : '/home/onyxia/work/data/Images_ROI/Images_ROI_Aty_for_multiview/',
        'VALID_SPLIT' : 0,
        'TEST_SPLIT' : 0,
        'IMAGE_SIZE' : 224, # Image size of resize when applying transforms.
        'BATCH_SIZE' : 32,
        'NUM_WORKERS' : 0}

        return DATA_CONFIG

def get_ext_data_config():
        DATA_CONFIG={'ROOT_DIR' : '/home/onyxia/work/data/Images_ROI/Images_ROI_Ext_for_multiview/',
        'VALID_SPLIT' : 0,
        'TEST_SPLIT' : 0,
        'IMAGE_SIZE' : 224, # Image size of resize when applying transforms.
        'BATCH_SIZE' : 32,
        'NUM_WORKERS' : 0}

        return DATA_CONFIG