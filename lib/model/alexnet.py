import torch 
from torch.utils.data import Dataset
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 7), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=2),
            nn.Conv2d(64, 192, kernel_size=(1, 5)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=2),
            nn.Conv2d(192, 384, kernel_size=(1, 3)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(1,3)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size= (1, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size= (1, 3), stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 1 * 4, 2048), nn.ReLU(), ### 这边需要改，输入的大小
            nn.BatchNorm1d(2048),
            # nn.Dropout(p=0.5),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.BatchNorm1d(1024),
            # nn.Dropout(p=dropout),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.BatchNorm1d(512),
            # nn.Dropout(p=dropout),
            # nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(512, 64), nn.ReLU(),
            nn.BatchNorm1d(64),
            # nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(64, num_classes), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x