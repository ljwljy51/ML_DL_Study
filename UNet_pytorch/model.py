import os
import numpy as np

import torch
import torch.nn as nn


# 네트워크 구현
class UNet(nn.Module):  # UNet class에 nn.Module 클래스 상속
    def __init__(self):
        super(UNet, self).__init__()

        # Convolution, Batch Normalization, ReLU 적용 레이어
        def CBR2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        ):
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
            layers += [nn.BatchNorm2d(num_features=out_channels)]  # BatchNorm layer
            layers += [nn.ReLU()]  # ReLU

            cbr = nn.Sequential(*layers)  # Convolution, BatchNorm, ReLU layer 통합해 리턴 위함
            # 이때, *은 iterable한 자료구조 내부의 데이터들을 unpacking해주는 역할.
            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)  # 첫 번째 스테이지 첫 번째 레이어
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)  # maxpooling layer

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(  # 논문에서는 concat 시 encoder의 output과 decoder의 output 해상도가 맞지 않기 때문에 crop해줘 concat했다 했으나, ConvTranspose2d 함수 내 padding값 조정해줌으로써 사이즈 맞춰 Crop하지 않아도 되도록 함
            in_channels=512,
            out_channels=512,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )  # pooling대신  upconv 적용

        # 인코더와 대칭 구조임에 유의
        self.dec4_2 = CBR2d(
            in_channels=2 * 512, out_channels=512
        )  # 논문에서와 같이 encoder 파트의 output을 concat해 진행해주기 때문에 input channel size 1024가 됨
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=256,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # output 생성 위함.
        self.fc = nn.Conv2d(
            in_channels=64,
            out_channels=1,
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

        x = self.fc(dec1_1)  # output

        return x
