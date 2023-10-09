import torch
import torch.nn as nn


class CBR2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        norm="bnorm",
        relu=0.0,
    ):
        super().__init__()
        layers = []
        layers += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        ]  # Conv layer
        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]  # BatchNorm layer
            elif norm == "inorm":  # instance normalization
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None:
            layers += [nn.ReLU() if relu == 0.0 else nn.LeakyReLU(relu)]  # ReLU

        self.cbr = nn.Sequential(
            *layers
        )  # Convolution, BatchNorm, ReLU layer 통합해 리턴 위함
        # 이때, *은 iterable한 자료구조 내부의 데이터들을 unpacking해주는 역할.

    def forward(self, x):
        return self.cbr(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        norm="bnorm",
        relu=0.0,
    ):
        super().__init__()

        layers = []

        # 1st CBR2d
        layers += [
            CBR2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                norm=norm,
                relu=relu,
            )
        ]

        # 2nd CBR2d
        # 두 번째 블록의 경우, ReLU 층 없음
        layers += [
            CBR2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                norm=norm,
                relu=None,
            )
        ]

        self.resblk = nn.Sequential(*layers)  # resblock 하나에 해당

    def forward(self, x):
        return x + self.resblk(x)  # input과 output 더해주는 형태 (element-wise sum)


# High resolution image-> low resolution image
# Low resolution image with (rx*rx) channel
# High resolution image에서 downsampling factor에 대응이 되는 채널을 갖는 low resolution image. subpixel image 생성


class PixelUnshuffle(nn.Module):
    def __init__(self, ry=2, rx=2):  # 각 축 방향에 따른 sampling ratio
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)  # 원본 이미지 shape 받아옴

        x = x.reshape(
            B, C, H // ry, ry, W // rx, rx
        )  # 총 6개 axis 생성. 각 축마다 어떻게 정의되어있는지 ㅗ학인하기

        x = x.permute(0, 1, 3, 5, 2, 4)  # 순서 바꿔줌. 배치, 채널, 비율, downsacled 순

        x = x.reshape(
            B, C * (ry * rx), H // ry, W // rx
        )  # 전 단계에서의 1,3,5번 차원이 채널 방향으로 resize됨

        return x


# low resolution image with (ry*rx) channel -> High resolution image
# downsampling되어있는 이미지에서 high-resolution image로 변환


class PixelShuffle(nn.Module):  # PixelShuffle의 역순
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)  # 배치, 채널, 비율, 해상도 순
        x = x.reshape(B, C // (ry * rx), H * ry, W * rx)

        return x
