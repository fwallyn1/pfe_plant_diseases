from torch import nn, optim
import torchvision

def get_config(side = "RECTO"):
        DATA_CONFIG={'ROOT_DIR' : '/home/onyxia/work/data/Images_ROI/Images_ROI_Typ_for_multiview/'+side,
        'VALID_SPLIT' : 0.2,
        'TEST_SPLIT' : 0.5,
        'IMAGE_SIZE' : 224, # Image size of resize when applying transforms.
        'BATCH_SIZE' : 32,
        'NUM_WORKERS' : 0}

        MODEL_CONFIG={'learning_rate' : 0.001,
                "epochs" : 30,
                "pretrained":True,
                "fine_tune":False,
                "model": {'model':torchvision.models.efficientnet_b5(weights='IMAGENET1K_V1'),
                                'avgpool':nn.AdaptiveAvgPool2d(1),
                                'last_layer_size':2048,
                                'model_name' : 'VGG',
                                'features' : None}}
        return MODEL_CONFIG,DATA_CONFIG


def get_aty_data_config(side = "RECTO"):
        DATA_CONFIG={'ROOT_DIR' : '/home/onyxia/work/data/Images_ROI/Images_ROI_Aty_for_multiview/'+side,
        'VALID_SPLIT' : 0,
        'TEST_SPLIT' : 0,
        'IMAGE_SIZE' : 224, # Image size of resize when applying transforms.
        'BATCH_SIZE' : 32,
        'NUM_WORKERS' : 0}
        return DATA_CONFIG

def get_ext_data_config(side = "RECTO"):
        DATA_CONFIG={'ROOT_DIR' : '/home/onyxia/work/data/Images_ROI/Images_ROI_Ext_for_multiview/'+side,
        'VALID_SPLIT' : 0,
        'TEST_SPLIT' : 0,
        'IMAGE_SIZE' : 224, # Image size of resize when applying transforms.
        'BATCH_SIZE' : 32,
        'NUM_WORKERS' : 0}
        return DATA_CONFIG

