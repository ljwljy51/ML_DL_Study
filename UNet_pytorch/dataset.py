import os
import numpy as np

import torch
import torch.nn as nn


# 데이터로더 구현
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir, transform=None
    ):  # 데이터가 들어있는 디렉토리 경로와 Transform 요소들을 인자로 받음
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)  # 데이터가 있는 폴더 내의 모든 파일 이름 받아옴

        lst_label = [f for f in lst_data if f.startswith("label")]  # prefix 통해 label뽑음
        lst_input = [f for f in lst_data if f.startswith("input")]

        lst_label.sort()  # 정렬 후 저장. Optional
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):  # 데이터 수 알기 위함
        return len(self.lst_label)

    def __getitem__(self, index):  # 인덱스에 해당하는 파일 리턴
        label = np.load(
            os.path.join(self.data_dir, self.lst_label[index])
        )  # 데이터가 numpy형식으로 저장되어있기 때문에 np.load 사용
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 0~255값을 갖는 데이터를 0~1의 값을 갖도록 Normalize
        label = label / 255.0
        input = input / 255.0

        # pytorch에서 모든 input은 3개의 채널을 가져야 함
        # 이를 위해 채널 axis가 부족하다면, 임의의 채널 생성해 삽입
        if label.ndim == 2:  # 채널 수가 2인 경우
            label = label[:, :, np.newaxis]  # np.newaxis 활용해 새 dimension생성
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

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
