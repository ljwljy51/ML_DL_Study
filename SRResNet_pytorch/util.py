import os
import numpy as np
from scipy.stats import poisson
from skimage.transform import radon, iradon, rescale, resize

import torch
import torch.nn as nn


# 네트워크 저장
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):  # ckpt 담을 디렉토리 없는 경우
        os.makedirs(ckpt_dir)  # 디렉토리 생성

    torch.save(
        {"net": net.state_dict(), "optim": optim.state_dict()},
        "%s/model_epoch%d.pth" % (ckpt_dir, epoch),
    )  # 마지막 인자는 path임. 즉, 해당 경로에 저장하라는 것


# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):  # 불러올 것 없는 경우
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)  # 저장된 백업본 파일명들 로드
    ckpt_lst.sort(
        key=lambda f: int("".join(filter(str.isdigit, f)))
    )  # 파일명 각각에서 숫자부분만 추출해서 숫자 기준으로 정렬

    dict_model = torch.load("%s/%s" % (ckpt_dir, ckpt_lst[-1]))  # 가장 최근 모델 로드 (정렬했으므로)

    net.load_state_dict(dict_model["net"])  # 모델 로드
    optim.load_state_dict(dict_model["optim"])  # 옵티마이저 로드
    epoch = int(ckpt_lst[-1].split("epoch")[1].split(".pth")[0])  # 파일명 활용해 epoch 숫자만 추출

    return net, optim, epoch


# sampling하기
def add_sampling(img, type="random", opts=None):
    sz = img.shape

    if type == "uniform":
        ## 1-1. Inpainting: Uniform sampling
        ds_y = opts[0].astype(np.int)  # 각 축으로 몇 배 sampling할 것인가
        ds_x = opts[1].astype(np.int)

        msk = np.zeros(sz)
        msk[::ds_y, ::ds_x, :] = 1  # 마스크 생성

        dst = img * msk  # 마스크와 이미지를 곱해줌. 샘플링 마스크 적용

    elif type == "random":
        ## 1-2. Inpainting: random sampling
        rnd = np.random.rand(sz[0], sz[1], sz[2])  # 채널 방향에서도 random하게 selection됨
        # rnd = np.random.rand(sz[0], sz[1], 1)  # 채널 방향에 대해서는 Random sampling 적용하지 않고싶은 경우
        prob = opts[0]  # 샘플링 비율

        msk = (rnd > prob).astype(np.float)
        # msk = np.tile(
        #    msk, (1, 1, sz[2])
        # )  # 채널방향으로 동일하게 masking하고 싶은 경우. tile함수를 통해 채널 방향으로 반복해 쌓아줌

        dst = img * msk

    elif type == "gaussian":
        ## 1-3. Inpainting: Gaussian sampling
        # Gaussian distribution 구현

        x0 = opts[0]  # 평균값. center를 기준으로 할 것이기에 0으로 설정
        y0 = opts[1]
        sgmx = opts[2]  # 표준편차
        sgmy = opts[3]

        a = opts[4]  # amplitude

        ly = np.linspace(-1, 1, sz[0])  # 구간 시작점, 구간 끝점, 구간 내 숫자 개수
        lx = np.linspace(-1, 1, sz[1])

        x, y = np.meshgrid(lx, ly)  # 격자 그리드 생성
        # 수식 통해 구현. 2D gaussian distribution
        gaus = a * np.exp(
            -((x - x0) ** 2 / (2 * sgmx**2) + (y - y0) ** 2 / (2 * sgmy**2))
        )
        gaus = np.tile(
            gaus[:, :, np.newaxis], (1, 1, sz[2])
        )  # tile함수를 통해 채널 방향으로 반복해 쌓아줌. 이때 gaus가 2차원이었기에 새 축 하나 추가함

        # gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))  # 채널방향으로 동일하게 하고 싶은 경우

        rnd = np.random.rand(sz[0], sz[1], sz[2])
        # rnd = np.random.rand(sz[0], sz[1], 1)  # 채널방향으로 동일하게 하고 싶은 경우
        msk = (rnd < gaus).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))  # 채널 방향으로 동일하게 하고 싶은 경우
        dst = img * msk
    return dst


# Noise 추가
def add_noise(img, type="random", opts=None):
    sz = img.shape

    if type == "random":
        sgm = opts[0]  # 값이 높아질 수록 이미지가 더 noisy해짐
        noise = (
            sgm / 255.0 * np.random.randn(sz[0], sz[1], sz[2])
        )  # img도 시각화 시 normalize하므로, sigma값에도 normalize 적용해줌
        dst = img + noise

    elif type == "poisson":
        # 주로 CT image 상에서 사용
        # 에너지가 높은 곳에서 noise가 많이 발생함
        dst = (
            poisson.rvs(255.0 * img) / 255.0
        )  # 샘플 생성. rvs는 푸아송 Noise를 생성함. 이때, normalize되어있던 이미지를 scale up 시켜줌
        # poisson noise의 경우, floating point로 값이 추가되는 것이 아닌 integer scale로 노이즈가 추가되기 때문에 scale up된 이미지에 noise 적용 뒤 다시 normalize해줌
        noise = dst - img
    return dst


# Blurring 추가하기
def add_blur(img, type="bilinear", opts=None):
    # ----------------
    # order options
    # ----------------
    # 0: Nearest-neighbor
    # 1: Bi-linear (default)
    # 2: Bi-quadratic
    # 3: Bi-cubic
    # 4: Bi-quartic
    # 5: Bi-quintic

    if type == "nearest":
        order = 0
    elif type == "bilinear":
        order = 1
    elif type == "biquadratic":
        order = 2
    elif type == "bicubic":
        order = 3
    elif type == "biquartic":
        order = 4
    elif type == "biquintic":
        order = 5

    sz = img.shape
    # dw = opts[0]  # downsamplnig ratio

    # downsampling된 이미지를 다시 upsampling해 리턴해줄 것인지 여부 (2번째 argumnet)
    if len(opts) == 1:  # downsampling된 이미지 리턴
        keepdim = True
    else:  # downsammpling된 것 다시 upsampling해 리턴
        keepdim = opts[1]

    # dst=rescale(img, scale=(dw, dw, 1), order=order)  # 채널 방향으로는 downsampling하지 않음
    # rescale은 sampling ratio를 인자로 받고, resize는 실제 해상도(output shape)를 인자로 받음
    # rescale의 경우, output size를 보장할 수 없음
    dst = resize(img, output_shape=(sz[0] // opts[0], sz[1] // opts[0], sz[2]), order=order)

    if keepdim:
        dst = resize(dst, output_shape=(sz[0], sz[1], sz[2]), order=order)

    return dst
