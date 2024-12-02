from torch import nn
from torchvision.models.resnet import BasicBlock

# encoder from https://github.com/jc639/pytorch-handwritingCTC/blob/master/model.py


def downsample(chan_in, chan_out, stride, pad=0):
    return nn.Sequential(
        nn.Conv2d(
            chan_in, chan_out, kernel_size=1, stride=stride, bias=False, padding=pad
        ),
        nn.BatchNorm2d(chan_out),
    )


# create a residual network, modify the downsampling as input is rectangular
# this one uses avg pooling to control the output sequence length instead of having to
# resort to any value fixing shenanigans!
# still basically assumes that everything got resized to the same dimensions without any padding.
class ResNetEncoder(nn.Module):

    def __init__(self, chan_in: int = 3, time_step: int = 96, zero_init_residual=False):
        super().__init__()

        self.chan_in = chan_in
        self.conv1 = nn.Conv2d(
            chan_in, 64, kernel_size=7, stride=2, padding=2, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(*[BasicBlock(64, 64) for i in range(0, 3)])
        self.layer2 = nn.Sequential(
            *[
                (
                    BasicBlock(64, 128, stride=2, downsample=downsample(64, 128, 2))
                    if i == 0
                    else BasicBlock(128, 128)
                )
                for i in range(0, 4)
            ]
        )
        self.layer3 = nn.Sequential(
            *[
                (
                    BasicBlock(
                        128, 256, stride=(1, 2), downsample=downsample(128, 256, (1, 2))
                    )
                    if i == 0
                    else BasicBlock(256, 256)
                )
                for i in range(0, 6)
            ]
        )
        self.layer4 = nn.Sequential(
            *[
                (
                    BasicBlock(
                        256, 512, stride=(1, 2), downsample=downsample(256, 512, (1, 2))
                    )
                    if i == 0
                    else BasicBlock(512, 512)
                )
                for i in range(0, 3)
            ]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(time_step, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init_constant_(m.bn2.weight, 0)

    @property
    def output_hsize(self):
        return 512

    def forward(self, xb):

        out = self.maxpool(self.bn1(self.relu(self.conv1(xb))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        return out.squeeze(dim=3).transpose(1, 2)
