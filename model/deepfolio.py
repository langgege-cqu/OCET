import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, input_channels, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, input_channels, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(input_channels),
        )
    
    def forward(self, x):
        return x + self.conv(x)

class DeepFolio(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_positional_embedding = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(4, 1), padding='same'),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        
        self.resblock1 = ResBlock(16)
        self.resblock2 = ResBlock(16)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )

        self.resblock3 = ResBlock(16)
        self.resblock4 = ResBlock(16)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 10), stride=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )

        self.resblock5 = ResBlock(16)
        self.resblock6 = ResBlock(16)


         # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # GRU layers

        self.gru = nn.GRU(input_size=96, hidden_size=64,  batch_first=True)

        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = x + self.conv_positional_embedding(x)

        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)

        x = self.conv2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = self.conv3(x)
        x = self.resblock5(x)
        x = self.resblock6(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        forecast_y = torch.softmax(x, dim=1)

        return forecast_y