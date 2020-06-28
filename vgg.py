##
 #  Worg
 ##

import torch
import torch.nn as nn
import torch.nn.functional as F

#  Source of inspiration for this model:
#  https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
VGG16_layers = [16, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class VGG_16(nn.Module):
    def __init__(self, in_channels = 1):
        super(VGG_16, self).__init__()
        self.in_channels = 1
        self.conv_layers = self.generate_conv_layers(VGG16_layers)
        self.fc_layers = nn.Sequential(
                    nn.Linear(512 * 4 * 4, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2),
                    nn.Softmax(dim = 1)
                )

    def generate_conv_layers(self, layer_architecture):
        layers = []
        in_channels = self.in_channels


        for x in layer_architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (3, 3),
                                        stride = (1, 1), padding = (1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()
                        ]
                in_channels = x
            else:
                layers += [nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))]

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x

