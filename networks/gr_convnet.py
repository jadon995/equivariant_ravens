import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, include_batch_normal=False):
        super(ResidualBlock, self).__init__()
        self.include_batch_normal = include_batch_normal

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation_fn = nn.Mish()

    def forward(self, x_in):
        x = self.conv1(x_in)
        if self.include_batch_normal:
            x = self.bn1(x)
        x = self.activation_fn(x)

        x = self.conv2(x)
        if self.include_batch_normal:
            x = self.bn2(x)
        # x = self.activation_fn(x + x_in)
        return x + x_in

class GenerativeResNet(nn.Module):
    def __init__(self, in_type, outdim, include_batch_normal=False, cutoff_early=False):
        super(GenerativeResNet, self).__init__()
        self.in_type = in_type
        self.outdim = outdim
        self.include_batch_normal = include_batch_normal
        # self.include_batch_normal = True

        self.conv1 = nn.Conv2d(self.in_type, 32, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.res1 = ResidualBlock(128, 128, include_batch_normal=self.include_batch_normal)
        self.res2 = ResidualBlock(128, 128, include_batch_normal=self.include_batch_normal)
        self.res3 = ResidualBlock(128, 128, include_batch_normal=self.include_batch_normal)
        self.res4 = ResidualBlock(128, 128, include_batch_normal=self.include_batch_normal)
        self.res5 = ResidualBlock(128, 128, include_batch_normal=self.include_batch_normal)

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(32, outdim, kernel_size=2)

        self.dropout_pos = nn.Dropout(p=0.1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

        self.activation_fn = nn.Mish()

    def forward(self, x_in):
        if self.include_batch_normal:
            x = self.activation_fn(self.bn1(self.conv1(x_in)))
            x = self.activation_fn(self.bn2(self.conv2(x)))
            x = self.activation_fn(self.bn3(self.conv3(x)))
        else:
            x = self.activation_fn(self.conv1(x_in))
            x = self.activation_fn(self.conv2(x))
            x = self.activation_fn(self.conv3(x))
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        if self.include_batch_normal:
            x = self.activation_fn(self.bn4(self.conv4(x)))
            x = self.activation_fn(self.bn5(self.conv5(x)))
        else:
            x = self.activation_fn(self.conv4(x))
            x = self.activation_fn(self.conv5(x))
        
        x = self.conv6(x)
        # print(x.shape)
        
        pos_output = self.pos_output(self.dropout_pos(x))
        # pos_output = self.pos_output(x)
        # print(pos_output.shape)

        
        return pos_output