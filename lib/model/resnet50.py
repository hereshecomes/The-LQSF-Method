import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Residual(nn.Module): #@save
    def __init__(self, input_channels, num_channels, output_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=1, 
                               padding=0, stride=1)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv3 = nn.Conv1d(num_channels, output_channels, kernel_size=1)
        if use_1x1conv:
            self.conv4 = nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=1)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.bn3 = nn.BatchNorm1d(output_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
)

def resnet_block(input_channels, num_channels, output_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, output_channels,  use_1x1conv=True))
        else:
            blk.append(Residual(input_channels, num_channels, output_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 256, 1),
                  *resnet_block(256, 64, 256, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(256, 128, 512, 1),
                  *resnet_block(512, 128, 512, 3, first_block=True))
b4 = nn.Sequential(*resnet_block(512, 256, 1024, 1),
                  *resnet_block(1024, 256, 1024, 5, first_block=True))
b5 = nn.Sequential(*resnet_block(1024, 512, 2048, 1),
                  *resnet_block(2048, 512, 2048, 2, first_block=True))

net = nn.Sequential(
                    b1, 
                    b2, 
                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                    b3, 
                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                    b4, 
                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                    b5,
                    nn.AdaptiveAvgPool1d(2),
                    nn.Flatten(), 
                    nn.Sequential(
                            nn.Linear(2048*2, 4096), nn.ReLU(), ### 这边需要改，输入的大小
                            nn.BatchNorm1d(4096),
                            # nn.Dropout(p=0.5),
                            nn.Linear(4096, 1024), nn.ReLU(),
                            nn.BatchNorm1d(1024),
                            # nn.Dropout(p=0.5),
                            nn.Linear(1024, 256), nn.ReLU(),
                            nn.BatchNorm1d(256),
                            # nn.Dropout(p=0.5),
                            # nn.Linear(512, 256), nn.ReLU(),
                            nn.Linear(256, 64), nn.ReLU(),
                            nn.BatchNorm1d(64),
                            # nn.Linear(64, 16), nn.ReLU(),
                            nn.Linear(64, 1), nn.Sigmoid()
                        )
)

def ResNet50():
    return net
