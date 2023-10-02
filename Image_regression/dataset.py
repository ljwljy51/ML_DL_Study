import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from util import *

# 데이터로더 구현
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir, transform=None, task=None, opts=None
    ):  # 데이터가 들어있는 디렉토리 경로와 Transform 요소들을 인자로 받음
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts

        lst_data = os.listdir(self.data_dir)  # 데이터가 있는 폴더 내의 모든 파일 이름 받아옴
        lst_data = [f for f in lst_data if f.endswith("jpg") | f.endswith("png")]

        lst_data.sort()  # 정렬 후 저장. Optional

        self.lst_data = lst_data

    def __len__(self):  # 데이터 수 알기 위함
        return len(self.lst_data)

    def __getitem__(self, index):  # 인덱스에 해당하는 파일 리턴
        # label = np.load(
        #     os.path.join(self.data_dir, self.lst_label[index])
        # )  # 데이터가 numpy형식으로 저장되어있기 때문에 np.load 사용
        # input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        sz = img.shape

        if sz[0] > sz[1]:  # 가로 세로 비율 대충 맞춰주기 위함
            img = img.transpose((1, 0, 2))

        # 0~255값을 갖는 데이터를 0~1의 값을 갖도록 Normalize
        if img.dtype == np.uint8:
            img = img / 255.0

        # pytorch에서 모든 input은 3개의 채널을 가져야 함
        # 이를 위해 채널 axis가 부족하다면, 임의의 채널 생성해 삽입
        if img.ndim == 2:  # 채널 수가 2인 경우
            img = img[:, :, np.newaxis]  # np.newaxis 활용해 새 dimension생성

        label = img  # task에 따라 이미지를 label로 설정

        if self.task == "denoising":
            input = add_noise(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == "inpainting":
            input = add_sampling(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == "super_resolution":
            input = add_blur(img, type=self.opts[0], opts=self.opts[1])

        # 딕셔너리 형태로 값 리턴 위함
        data = {"input": input, "label": label}

        # transform 함수 지정되어있는 경우. 전달 받은 경우
        if self.transform:
            data = self.transform(data)  # transform 적용

        return data


##Transform 구현(transforms 활용해도 되지만..)


class ToTensor(object):
    def __call__(self, data):  # 데이터를 입력받아 Tensor로 변환
        label, input = data["label"], data["input"]

        # numpy는  y, x, channel 순이고, tensor는 channel, y, x 순서임
        label = label.transpose((2, 0, 1)).astype(np.float32)  # 차원 순서 변경 및 자료형 변환
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {
            "label": torch.from_numpy(label),
            "input": torch.from_numpy(input),
        }  # from_numpy함수 통해 텐서로 변환.딕셔너리 형태로 반환

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):  # 1 channel 가정.
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data["label"], data["input"]

        input = (input - self.mean) / self.std
        label = (label - self.mean) / self.std
        data = {"label": label, "input": input}  # input에만 normalization 적용

        return data


class RandomFlip(object):
    def __call__(self, data):
        label, input = data["label"], data["input"]

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)  # input과 label 모두에 flip 적용해야함. 좌우반전

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)  # 좌우반전

        data = {"label": label, "input": input}

        return data


class RandomCrop(object):
    def __init__(self, shape):  # crop할 shape을 파라미터로 받음
        self.shape = shape

    def __call__(self, data):
        # label과 data에 동일하게 crop적용 위함
        input, label = data["input"], data["label"]

        h, w = input.shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]  # 세로축 방향으로 arange값 생성 위함
        id_x = np.arange(left, left + new_w, 1)

        input = input[id_y, id_x]
        label = label[id_y, id_x]

        data = {"input": input, "label": label}

        return data
