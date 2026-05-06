import torch.nn as nn

class CNN3D(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, dropout=0.3):
        super(CNN3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(num_groups=4, num_channels=16), #nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(num_groups=8, num_channels=32), #nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(num_groups=8, num_channels=64), #nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )   
        self.fc_block = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), # Reduces input from [(193//8), (229//8), (193//8), 64] to [64, 1, 1, 1]
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x
