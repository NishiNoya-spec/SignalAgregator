import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


# TODO ResnetBlock


class TorsionUNet(nn.Module):

    def __init__(self, in_channels, n_class):
        super().__init__()

        start_channels = 64
        self.dconv_down1 = double_conv(in_channels, start_channels)
        self.dconv_down2 = double_conv(
            start_channels,
            start_channels * 2,
        )
        self.dconv_down3 = double_conv(
            start_channels * 2,
            start_channels * 4,
        )
        self.dconv_down4 = double_conv(
            start_channels * 4,
            start_channels * 8,
        )
        self.dconv_down5 = double_conv(
            start_channels * 8,
            start_channels * 16,
        )
        self.dconv_down6 = double_conv(
            start_channels * 16,
            start_channels * 32,
        )

        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.dconv_up5 = double_conv(
            start_channels * 16 + start_channels * 32, start_channels * 16
        )
        self.dconv_up4 = double_conv(
            start_channels * 8 + start_channels * 16, start_channels * 8
        )
        self.dconv_up3 = double_conv(
            start_channels * 4 + start_channels * 8, start_channels * 4
        )
        self.dconv_up2 = double_conv(
            start_channels * 2 + start_channels * 4, start_channels * 2
        )
        self.dconv_up1 = double_conv(
            start_channels * 2 + start_channels, start_channels
        )

        self.conv_last = nn.Conv1d(
            start_channels, n_class, kernel_size=3, padding=1
        )

    def forward(self, x):
        # print(x.min(), x.max())
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)    # avgpooling  попробовать

        x = self.dconv_down6(x)
        # avgpooling + Linear (abs_max)
        x = self.upsample(x)
        x = torch.cat([x, conv5], dim=1)

        x = self.dconv_up5(x)
        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        # out = x.sigmoid()
        return x    # torch.clip(x, 0, 1)


if __name__ == "__main__":
    torsion_rnn = TorsionUNet(236, 1)
    print(torsion_rnn)
