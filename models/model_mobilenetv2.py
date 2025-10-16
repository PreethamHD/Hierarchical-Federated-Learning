import torch.nn as nn
import torchvision.models as models

def get_mobilenetv2(num_classes=10, pretrained=True):
    model = models.mobilenet_v2(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
