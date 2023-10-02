import os
import numpy as np

import torch
import torch.nn as nn

from layer import *

# 네트워크 구현
# Image regression task에서는 residual learning technique 많이 사용
class UNet(nn.Module):  # UNet class에 nn.Module 클래스 상속
    def __init__(self, nch, nker, norm="bnorm", learning_type="plain"):
        super(UNet, self).__init__()
        self.learning_type = learning_type

        # Contracting path
        self.enc1_1 = CBR2d(
            in_channels=nch, out_channels=1 * nker, norm=norm
        )  # 첫 번째 스테이지 첫 번째 레이어
        self.enc1_2 = CBR2d(in_channels=1 * nker, out_channels=1 * nker, norm=norm)

        self.pool1 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc2_1 = CBR2d(in_channels=1 * nker, out_channels=2 * nker, norm=norm)
        self.enc2_2 = CBR2d(in_channels=2 * nker, out_channels=2 * nker, norm=norm)

        self.pool2 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc3_1 = CBR2d(in_channels=2 * nker, out_channels=4 * nker, norm=norm)
        self.enc3_2 = CBR2d(in_channels=4 * nker, out_channels=4 * nker, norm=norm)

        self.pool3 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc4_1 = CBR2d(in_channels=4 * nker, out_channels=8 * nker, norm=norm)
        self.enc4_2 = CBR2d(in_channels=8 * nker, out_channels=8 * nker, norm=norm)

        self.pool4 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc5_1 = CBR2d(in_channels=8 * nker, out_channels=16 * nker, norm=norm)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=16 * nker, out_channels=8 * nker, norm=norm)

        self.unpool4 = nn.ConvTranspose2d(  # 논문에서는 concat 시 encoder의 output과 decoder의 output 해상도가 맞지 않기 때문에 crop해줘 concat했다 했으나, ConvTranspose2d 함수 내 padding값 조정해줌으로써 사이즈 맞춰 Crop하지 않아도 되도록 함
            in_channels=8 * nker,
            out_channels=8 * nker,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )  # pooling대신  upconv 적용

        # 인코더와 대칭 구조임에 유의
        self.dec4_2 = CBR2d(
            in_channels=2 * 8 * nker, out_channels=8 * nker, norm=norm
        )  # 논문에서와 같이 encoder 파트의 output을 concat해 진행해주기 때문에 input channel size 1024가 됨
        self.dec4_1 = CBR2d(in_channels=8 * nker, out_channels=4 * nker, norm=norm)

        self.unpool3 = nn.ConvTranspose2d(
            in_channels=4 * nker,
            out_channels=4 * nker,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.dec3_2 = CBR2d(in_channels=2 * 4 * nker, out_channels=4 * nker, norm=norm)
        self.dec3_1 = CBR2d(in_channels=4 * nker, out_channels=2 * nker, norm=norm)

        self.unpool2 = nn.ConvTranspose2d(
            in_channels=2 * nker,
            out_channels=2 * nker,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.dec2_2 = CBR2d(in_channels=2 * 2 * nker, out_channels=2 * nker, norm=norm)
        self.dec2_1 = CBR2d(in_channels=2 * nker, out_channels=1 * nker, norm=norm)

        self.unpool1 = nn.ConvTranspose2d(
            in_channels=1 * nker,
            out_channels=1 * nker,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.dec1_2 = CBR2d(in_channels=2 * 1 * nker, out_channels=1 * nker, norm=norm)
        self.dec1_1 = CBR2d(in_channels=1 * nker, out_channels=1 * nker, norm=norm)

        # output 생성 위함.
        self.fc = nn.Conv2d(
            in_channels=1 * nker,
            out_channels=nch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )  # 변수명만 fc일 뿐, 사실 상 fc layer는 사용하지 않음
        # 1x1 Convolution연산 통해 해상도 유지한 채 채널 수만 변환

    def forward(self, x):  # 각 레이어 연결. 여기서 x는 input image를 의미
        # Encoder part
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        # Decoder part
        # 이때, Encoder part의 output들이 Concat되는 부분들에 유의
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat(
            (unpool4, enc4_2), dim=1
        )  # 논문에서 언급된 것과 같이 결과들을 concat해줌. 이때, dimension=1은 채널 방향을 의미함. 0은 batch, 2는 height, 3은 width
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        if self.learning_type == "plain":
            x = self.fc(dec1_1)
        elif self.learning_type == "residual":
            x = (
                self.fc(dec1_1) + x
            )  # output. residual learning 적용. "입력과 출력의 차이"를 얻게 되는 것.

        # 즉, "입력과 출력의 차이"를 최소화하도록 학습하게 되는 것
        # 연산량 증가 최소화하면서 gradient vanishing 문제 완화

        # Denoising의 경우 noise만을 학습하며, Super-resolution의 경우 high frequency만을 학습하고, inpainting의 경우 sampling되지 않은 부분만을 학습하도록 하는training technique이 residual learning

        return x


class Hourglass(nn.Module):  # UNet class에 nn.Module 클래스 상속
    def __init__(self, nch, nker, norm="bnorm", learning_type="plain"):
        super(UNet, self).__init__()
        self.learning_type = learning_type

        # Contracting path
        self.enc1_1 = CBR2d(
            in_channels=nch, out_channels=1 * nker, norm=norm
        )  # 첫 번째 스테이지 첫 번째 레이어
        self.enc1_2 = CBR2d(in_channels=1 * nker, out_channels=1 * nker, norm=norm)

        self.pool1 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc2_1 = CBR2d(in_channels=1 * nker, out_channels=2 * nker, norm=norm)
        self.enc2_2 = CBR2d(in_channels=2 * nker, out_channels=2 * nker, norm=norm)

        self.pool2 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc3_1 = CBR2d(in_channels=2 * nker, out_channels=4 * nker, norm=norm)
        self.enc3_2 = CBR2d(in_channels=4 * nker, out_channels=4 * nker, norm=norm)

        self.pool3 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc4_1 = CBR2d(in_channels=4 * nker, out_channels=8 * nker, norm=norm)
        self.enc4_2 = CBR2d(in_channels=8 * nker, out_channels=8 * nker, norm=norm)

        self.pool4 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc5_1 = CBR2d(in_channels=8 * nker, out_channels=16 * nker, norm=norm)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=16 * nker, out_channels=8 * nker, norm=norm)

        self.unpool4 = nn.ConvTranspose2d(  # 논문에서는 concat 시 encoder의 output과 decoder의 output 해상도가 맞지 않기 때문에 crop해줘 concat했다 했으나, ConvTranspose2d 함수 내 padding값 조정해줌으로써 사이즈 맞춰 Crop하지 않아도 되도록 함
            in_channels=8 * nker,
            out_channels=8 * nker,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )  # pooling대신  upconv 적용

        # 인코더와 대칭 구조임에 유의
        self.dec4_2 = CBR2d(
            in_channels=1 * 8 * nker, out_channels=8 * nker, norm=norm
        )  # 논문에서와 같이 encoder 파트의 output을 concat해 진행해주기 때문에 input channel size 1024가 됨
        self.dec4_1 = CBR2d(in_channels=8 * nker, out_channels=4 * nker, norm=norm)

        self.unpool3 = nn.ConvTranspose2d(
            in_channels=4 * nker,
            out_channels=4 * nker,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.dec3_2 = CBR2d(in_channels=1 * 4 * nker, out_channels=4 * nker, norm=norm)
        self.dec3_1 = CBR2d(in_channels=4 * nker, out_channels=2 * nker, norm=norm)

        self.unpool2 = nn.ConvTranspose2d(
            in_channels=2 * nker,
            out_channels=2 * nker,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.dec2_2 = CBR2d(in_channels=1 * 2 * nker, out_channels=2 * nker, norm=norm)
        self.dec2_1 = CBR2d(in_channels=2 * nker, out_channels=1 * nker, norm=norm)

        self.unpool1 = nn.ConvTranspose2d(
            in_channels=1 * nker,
            out_channels=1 * nker,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.dec1_2 = CBR2d(in_channels=1 * 1 * nker, out_channels=1 * nker, norm=norm)
        self.dec1_1 = CBR2d(in_channels=1 * nker, out_channels=1 * nker, norm=norm)

        # output 생성 위함.
        self.fc = nn.Conv2d(
            in_channels=1 * nker,
            out_channels=nch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )  # 변수명만 fc일 뿐, 사실 상 fc layer는 사용하지 않음
        # 1x1 Convolution연산 통해 해상도 유지한 채 채널 수만 변환

    def forward(self, x):  # 각 레이어 연결. 여기서 x는 input image를 의미
        # Encoder part
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        # Decoder part
        # 이때, Encoder part의 output들이 Concat되는 부분들에 유의
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        # cat4 = torch.cat(
        #     (unpool4, enc4_2), dim=1
        # )  # 논문에서 언급된 것과 같이 결과들을 concat해줌. 이때, dimension=1은 채널 방향을 의미함. 0은 batch, 2는 height, 3은 width
        cat4 = unpool4
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        # cat3 = torch.cat((unpool3, enc3_2), dim=1)
        cat3 = unpool3
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        # cat2 = torch.cat((unpool2, enc2_2), dim=1)
        cat2 = unpool2
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        # cat1 = torch.cat((unpool1, enc1_2), dim=1)
        cat1 = unpool1
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        if self.learning_type == "plain":
            x = self.fc(dec1_1)
        elif self.learning_type == "residual":
            x = (
                self.fc(dec1_1) + x
            )  # output. residual learning 적용. "입력과 출력의 차이"를 얻게 되는 것.

        # 즉, "입력과 출력의 차이"를 최소화하도록 학습하게 되는 것
        # 연산량 증가 최소화하면서 gradient vanishing 문제 완화

        # Denoising의 경우 noise만을 학습하며, Super-resolution의 경우 high frequency만을 학습하고, inpainting의 경우 sampling되지 않은 부분만을 학습하도록 하는training technique이 residual learning

        return x
