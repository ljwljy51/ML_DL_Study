import os
import numpy as np

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
