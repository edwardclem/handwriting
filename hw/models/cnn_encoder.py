from typing import Tuple

from torch import nn
from torchvision.models.resnet import BasicBlock

# encoder inspired by https://github.com/jc639/pytorch-handwritingCTC/blob/master/model.py


# performs downsampling on residual path.
def downsample(chan_in, chan_out, stride, pad=0):
    return nn.Sequential(
        nn.Conv2d(
            chan_in, chan_out, kernel_size=1, stride=stride, bias=False, padding=pad
        ),
        nn.BatchNorm2d(chan_out),
    )


# makes a set of ResNet blocks - handles downsampling in the case that input planes differs from planes.
def make_layer(
    input_planes: int,
    planes: int,
    n_layers: int,
    stride: Tuple[int, int] = (1, 2),
    max_pool: bool = False,
) -> nn.Sequential:

    layers = []

    if input_planes != planes:
        layers.append(
            BasicBlock(
                input_planes,
                planes,
                stride=stride,
                downsample=downsample(input_planes, planes, stride),
            )
        )
        n_layers -= 1  # decrement number of required layers

    for _ in range(n_layers):
        layers.append(BasicBlock(planes, planes))

    if max_pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)


# feature encoder. Flattening moved elsewhere for eaiser debug.
class ResNetEncoder(nn.Module):
    def __init__(
        self,
        chan_in: int = 1,
        intermediate_stride: Tuple[int, int] = (1, 2),  # more aggressive over width
        intermediate_max_pool: bool = False,
        zero_init_residual=False,
    ):
        super().__init__()

        self.chan_in = chan_in
        self.intermediate_stride = intermediate_stride
        self.intermediate_max_pool = intermediate_max_pool
        # initial 7x7 convolution
        self.conv1 = nn.Conv2d(
            chan_in, 64, kernel_size=7, stride=2, padding=2, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # project from 64 -> 128 -> 256 -> 512 planes

        self.encoder = nn.Sequential(
            *[
                make_layer(
                    input_planes=64,
                    planes=64,
                    n_layers=3,
                    stride=1,
                    max_pool=self.intermediate_max_pool,
                ),
                make_layer(
                    input_planes=64,
                    planes=128,
                    n_layers=4,
                    stride=2,
                    max_pool=self.intermediate_max_pool,
                ),
                make_layer(
                    input_planes=128,
                    planes=256,
                    n_layers=6,
                    stride=self.intermediate_stride,
                    max_pool=self.intermediate_max_pool,
                ),
                make_layer(
                    input_planes=256,
                    planes=512,
                    n_layers=3,
                    stride=self.intermediate_stride,
                    max_pool=self.intermediate_max_pool,
                ),
            ]
        )

        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(time_step, 1))

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
        # input: (batch, channel, height, width)
        initial_conv = self.maxpool(self.bn1(self.relu(self.conv1(xb))))
        # output: (batch, 512, featuremap_height, featuremap_width)
        features = self.encoder(initial_conv)
        return features
