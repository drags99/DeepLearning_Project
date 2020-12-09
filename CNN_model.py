import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast

class VGG(nn.Module):
    def __init__(self,input_channels=3,W=64,H=64,num_classes=1):
        super(VGG, self).__init__()
        self.H=H
        self.W=W
        self.input_channels=input_channels
        self.num_classes=num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        #how layers before fully connected layers will be defined
        cfgs=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.features=C_layers(cfgs,True)


        #last layers, extract feature before this function is called in forward
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        #add function to extract features here
        #####################################

        #####################################

        x = self.classifier(x)
        return x

    #layers before classifier, extract features after this
def C_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)