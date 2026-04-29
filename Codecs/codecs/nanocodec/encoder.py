import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(torch.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=24):
        super().__init__()

        self.initial = nn.Conv1d(in_channels, base_channels, 7, padding=3)

        self.layers = nn.ModuleList()
        channels = base_channels

        # Paper stride pattern for 12.5 FPS
        strides = [2, 3, 6, 7, 7]

        for stride in strides:
            block = nn.Sequential(
                ResidualBlock(channels, 1),
                ResidualBlock(channels, 3),
                ResidualBlock(channels, 5),
                nn.Conv1d(channels, channels * 2, kernel_size=4, stride=stride, padding=1)
            )
            self.layers.append(block)
            channels *= 2

        self.out = nn.Conv1d(channels, 32, 3, padding=1)

    def forward(self, x):
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


if __name__ == "__main__":
    x = torch.randn(1, 1, 16000)
    model = Encoder()
    y = model(x)

    print("Input:", x.shape)
    print("Encoded:", y.shape)
