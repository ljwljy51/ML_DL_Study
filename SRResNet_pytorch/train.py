"""
-------------------------------------
denoising
-------------------------------------
python  train.py \
        --mode train \
        --network unet \
        --learning_type residual \
        --task denoising \
        --opts random 30.0

-------------------------------------
inpainting
-------------------------------------
python  train.py \
        --mode train \
        --network unet \
        --learning_type residual \
        --task inpainting \
        --opts uniform 0.5

-------------------------------------
python  train.py \
        --mode train \
        --network unet \
        --learning_type residual \
        --task inpainting \
        --opts random 0.5

-------------------------------------
super_resolution
-------------------------------------
python  train.py \
        --mode train \
        --network unet \
        --learning_type residual \
        --task super_resolution \
        --opts bilinear 4.0
"""

# 필요 라이브러리 추가
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *  # dataset.py 내의 모든 요소  import
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

# Parser 생성
parser = argparse.ArgumentParser(
    description="Train the UNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument(
    "--train_continue", default="off", type=str, dest="train_continue"
)  # 네트워크 처음부터 학습시킬건지, 이미 학습된 것 로드해 추가학습할건지 유무


parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=8, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=300, type=int, dest="num_epoch")

parser.add_argument(
    "--data_dir", default="./data/BSR/BSD500/data/images", type=str, dest="data_dir"
)
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument(
    "--task",
    default="super_resolution",
    choices=["denoising", "inpainting", "super_resolution"],
    type=str,
    dest="task",
)
parser.add_argument(
    "--opts", nargs="+", default=["bilinear", 4, 0], dest="opts"
)  # nargs='+'통해 배열 형태의 여러 argument를 받을 수 있음

parser.add_argument("--ny", default=320, type=int, dest="ny")  # 입력 이미지 사이즈 관련
parser.add_argument("--nx", default=480, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument(
    "--nker", default=64, type=int, dest="nker"
)  # Unet의 default 채널 사이즈 변경 위함

parser.add_argument(
    "--network",
    default="resnet",
    choices=["unet", "hourglass", "resnet", "srresnet"],
    type=str,
    dest="network",
)  # 여러 네트워크 선택할 수 있도록 하기 위함
parser.add_argument(
    "--learning_type",
    default="plain",
    choices=["plain", "residual"],
    type=str,
    dest="learning_type",
)

# Argument parsing
args = parser.parse_args()

# 트레이닝 파라미터 설정
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

# 경로 설정
data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

task = args.task
opts = [
    args.opts[0],
    np.asarray(args.opts[1:]).astype(np.float),
]  # 항상 옵션 첫 번째 argumnet는 type임

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network
learning_type = args.learning_type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("mode: %s" % mode)
print("train continue: %s" % train_continue)

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)

print("task: %s" % task)
print("opts: %s" % opts)

print("network: %s" % network)
print("learning type: %s" % learning_type)

print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)

print("device: %s" % device)

# 디렉토리 생성
result_dir_train = os.path.join(result_dir, "train")
result_dir_val = os.path.join(result_dir, "val")
result_dir_test = os.path.join(result_dir, "test")

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, "png"))
    os.makedirs(os.path.join(result_dir_val, "png"))

    os.makedirs(os.path.join(result_dir_test, "png"))
    os.makedirs(os.path.join(result_dir_test, "numpy"))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 네트워크 학습 준비
if mode == "train":  # train mode인 경우
    transform_train = transforms.Compose(
        [
            RandomCrop(shape=(ny, nx)),
            Normalization(mean=0.5, std=0.5),
            RandomFlip(),
        ]
    )
    transform_val = transforms.Compose(
        [RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5)]
    )

    dataset_train = Dataset(
        data_dir=os.path.join(data_dir, "train"),
        transform=transform_train,
        task=task,
        opts=opts,
    )  # 데이터셋 객체 생성
    loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=0
    )

    dataset_val = Dataset(
        data_dir=os.path.join(data_dir, "val"),
        transform=transform_val,
        task=task,
        opts=opts,
    )
    loader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 부수적 변수 설정
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:  # eval mode인 경우
    transform_test = transforms.Compose(
        [RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5)]
    )

    dataset_test = Dataset(
        data_dir=os.path.join(data_dir, "test"),
        transform=transform_test,
        task=task,
        opts=opts,
    )  # 데이터셋 객체 생성
    loader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=0
    )

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)


# 네트워크 생성
if network == "unet":
    net = UNet(
        in_channels=nch,
        out_channels=nch,
        nker=nker,
        norm="bnorm",
        learning_type=learning_type,
    ).to(device)
elif network == "Hourglass":
    net = Hourglass(
        in_channels=nch,
        out_channels=nch,
        nker=nker,
        norm="bnorm",
        learning_type=learning_type,
    ).to(device)
elif network == "resnet":
    net = ResNet(
        in_channels=nch,
        out_channels=nch,
        nker=nker,
        learning_type=learning_type,
    ).to(device)

elif network == "srresnet":
    net = SRResNet(
        in_channels=nch,
        out_channels=nch,
        nker=nker,
        learning_type=learning_type
    ).to(device)

# 손실함수 정의
# fn_loss = nn.BCEWithLogitsLoss().to(
#     device
# )  # Binary Cross Entropy loss. Sigmoid layer+BCE Loss의 조합. 즉, output이 0~1사이의 값이 아니어도 됨.
# 그냥 BCE를 사용할 경우, output이 0~1 값을 벗어날 때 오류 발생
# Segmentation task에서 주로 사용하는 Loss

# Regression/Restoration에서는 L2 loss / L1 loss 주로 사용
fn_loss = nn.MSELoss().to(device)


# optimizer 설정

# Optimizer 설정
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 부수적 function 설정
fn_tonumpy = (
    lambda x: x.to("cpu").detach().numpy().transpose(0, 2, 3, 1)
)  # tensor에서 numpy로 변환 위함. 이때, device 변경 및 detach를 통해 gradient 전파를 막아줌. 차원 수도 바꿔줌
# 이때, 0번 차원은 배치당 instance 수를 의미
fn_denorm = lambda x, mean, std: (x * std) + mean  # denormalization위함
# fn_class = lambda x: 1.0 * (x > 0.5)  # output이 0,1 중 하나의 값을 갖도록 하기 위함
# regression에서는 fn_class 사용 안함


# Tensorboard 사용 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, "train"))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, "val"))


# 네트워크 학습
st_epoch = 0

# train mode
if mode == "train":
    if train_continue == "on":  # 이미 학습된 것 불러와 추가학습하고싶은 경우
        net, optim, st_epoch = load(
            ckpt_dir=ckpt_dir, net=net, optim=optim
        )  # 저장된 모델 있다면 로드

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()  # train mode로 변환
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):  # index가 1부터 시작되도록 함
            # forward pass
            label = data["label"].to(device)
            input = data["input"].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()  # Pytorch에서는 gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문에 zerograd해줌

            loss = fn_loss(output, label)  # loss 계산
            loss.backward()  # Gradient 계산

            optim.step()  # 가중치 갱신

            # loss 계산 및 출력
            loss_arr += [loss.item()]

            print(
                "TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f"
                % (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr))
            )

            if batch % 10 == 0:
                # Tensorboard 저장. 전부  numpy배열로 변환.
                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                input = fn_tonumpy(
                    fn_denorm(input, mean=0.5, std=0.5)
                )  # denormalization
                output = fn_tonumpy(
                    fn_denorm(output, mean=0.5, std=0.5)
                )  # 각각 처리 다르게 해주는 것에 유의

                # writer_train.add_image(
                #     "label",
                #     label,
                #     num_batch_train * (epoch - 1) + batch,
                #     dataformats="NHWC",
                # )  # numpy 배열이므로 마지막 차원이 채널. 세 번째 인자는 global step value. number of batches seen by the graph.
                # writer_train.add_image(
                #     "input",
                #     input,
                #     num_batch_train * (epoch - 1) + batch,
                #     dataformats="NHWC",
                # )
                # writer_train.add_image(
                #     "output",
                #     output,
                #     num_batch_train * (epoch - 1) + batch,
                #     dataformats="NHWC",
                # )
                # matplotlib으로 이미지 저장할 때는 데이터의 range를 0~1 사이로 clipping해줘야 원활히 작동함
                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                id = num_batch_train * (epoch - 1) + batch

                plt.imsave(
                    os.path.join(result_dir_train, "png", "%04d_label.png" % id),
                    label[0],
                )  # 첫 번째 배치 안의 이미지들만을 저장
                plt.imsave(
                    os.path.join(result_dir_train, "png", "%04d_input.png" % id),
                    input[0],
                )
                plt.imsave(
                    os.path.join(result_dir_train, "png", "%04d_output.png" % id),
                    output[0],
                )

        # tensorboard에 loss 저장
        writer_train.add_scalar("loss", np.mean(loss_arr), epoch)

        # validation step에서는 backprop하지 않음
        with torch.no_grad():
            net.eval()  # validation step임을 명시. 드롭아웃이나 batchnorm등에서 다르게 작용하게 하기 위함
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data["label"].to(device)
                input = data["input"].to(device)

                output = net(input)

                # 손실함수 계산
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print(
                    "VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f"
                    % (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr))
                )

                if batch % 20 == 0:
                    # Tensorboard 저장
                    label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                    output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                    input = np.clip(input, a_min=0, a_max=1)
                    output = np.clip(output, a_min=0, a_max=1)

                    id = num_batch_val * (epoch - 1) + batch

                    plt.imsave(
                        os.path.join(result_dir_val, "png", "%04d_label.png" % id),
                        label[0],
                    )  # 첫 번째 배치 안의 이미지들만을 저장
                    plt.imsave(
                        os.path.join(result_dir_val, "png", "%04d_input.png" % id),
                        input[0],
                    )
                    plt.imsave(
                        os.path.join(result_dir_val, "png", "%04d_output.png" % id),
                        output[0],
                    )

                    # writer_val.add_image(
                    #     "label",
                    #     label,
                    #     num_batch_val * (epoch - 1) + batch,
                    #     dataformats="NHWC",
                    # )
                    # writer_val.add_image(
                    #     "input",
                    #     input,
                    #     num_batch_val * (epoch - 1) + batch,
                    #     dataformats="NHWC",
                    # )
                    # writer_val.add_image(
                    #     "output",
                    #     output,
                    #     num_batch_val * (epoch - 1) + batch,
                    #     dataformats="NHWC",
                    # )
        print("-----------------------------------------------------")
        # tensorboard에 Loss 저장
        writer_val.add_scalar("loss", np.mean(loss_arr), epoch)

        # 모델 저장
        if epoch % 10 == 0:  # epoch 특정 횟수마다 저장하고싶으면
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()  # 생성했던 두 개의 writer close시켜줌
    writer_val.close()

# test mode
else:
    net, optim, st_epoch = load(
        ckpt_dir=ckpt_dir, net=net, optim=optim
    )  # eval 모드인 경우, 항상 저장된 모델 불러옴
    with torch.no_grad():
        net.eval()  # validation step임을 명시. 드롭아웃이나 batchnorm등에서 다르게 작용하게 하기 위함
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data["label"].to(device)
            input = data["input"].to(device)

            output = net(input)

            # 손실함수 계산
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print(
                "TEST:  BATCH %04d / %04d | LOSS %.4f"
                % (batch, num_batch_test, np.mean(loss_arr))
            )

            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            # 출력된 output numpy 데이터로 저장

            # 각 슬라이스 따로따로 저장
            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                label_ = label[j]
                input_ = input[j]
                output_ = output[j]

                # numpy 저장
                np.save(
                    os.path.join(result_dir, "test", "numpy", "%04d_label.npy" % id),
                    label_,
                )
                np.save(
                    os.path.join(result_dir, "test", "numpy", "%04d_input.npy" % id),
                    input_,
                )
                np.save(
                    os.path.join(result_dir, "test", "numpy", "%04d_output.npy" % id),
                    output_,
                )

                # clipping 후 이미지 저장
                label_ = np.clip(label_, a_min=0, a_max=1)
                input_ = np.clip(input_, a_min=0, a_max=1)
                output_ = np.clip(output_, a_min=0, a_max=1)

                # png 저장
                plt.imsave(
                    os.path.join(result_dir, "test", "png", "%04d_label.png" % id),
                    label_,
                    cmap="gray",
                )
                plt.imsave(
                    os.path.join(result_dir, "test", "png", "%04d_input.png" % id),
                    input_,
                    cmap="gray",
                )
                plt.imsave(
                    os.path.join(result_dir, "test", "png", "%04d_output.png" % id),
                    output_,
                    cmap="gray",
                )

    # 전체 test set에 대한 평균 loss 출력 위함
    print(
        "AVERAGE TEST:  BATCH %04d / %04d | LOSS %.4f"
        % (batch, num_batch_test, np.mean(loss_arr))
    )
