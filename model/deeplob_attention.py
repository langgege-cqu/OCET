import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepLOBAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # convolution blocks
        # self.conv_positional = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=(4, 1), padding='same'),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     # nn.BatchNorm2d(1),
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32)
        )

        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.encoder = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)

        self.decoder = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.bn = nn.BatchNorm1d(64, momentum=0.6)
        self.fc1 = nn.Linear(64, 3)


    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).cuda()
        c0 = torch.zeros(1, x.size(0), 64).cuda()
        # x = x + self.conv_positional(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        #  x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        # seq to seq
        encoder_output, (state_h, state_c) = self.encoder(x, (h0, c0))
        encoder_state_h = state_h.permute(1, 0, 2)        
        decoder_output, (state_h, state_c) = self.decoder(encoder_state_h, (state_h, state_c))
        
        # attentions
        attention = torch.mul(encoder_output, decoder_output)
        attention = torch.sum(attention, dim=-1)
        attention = F.softmax(attention, dim=-1)
        attention = torch.unsqueeze(attention, dim=-1)
        
        output = torch.mul(encoder_output, attention)
        output = torch.sum(output, dim=1)
        output = self.bn(output)
        output = self.fc1(output)
        forecast_y = torch.softmax(output, dim=1)

        return forecast_y