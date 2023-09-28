# numpy 이미지 -> png 변환
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from dataset import *


parser = argparse.ArgumentParser(
    description="Display data_numpy format",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--data_dir", default="./data", type=str, dest="data_dir")
parser.add_argument("--sub_dir", default="train", type=str, dest="sub_dir")

args = parser.parse_args()

data_dir = args.data_dir
sub_dir = args.sub_dir
##################################################################################

# Transform 테스트 위함
# 데이터로더 확인

transform = transforms.Compose(
    [Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()]
)  # transforms.Compose함수 통해 여러 transform들을 묶어둘 수 있음. 순서 중요
dataset_train = Dataset(data_dir=os.path.join(data_dir, sub_dir), transform=transform)

for i in range(len(dataset_train)):
    data = dataset_train.__getitem__(i)
    input = data["input"]
    label = data["label"]

    plt.imsave(
        os.path.join(data_dir, "png", "input_%03d.png" % i),
        input.squeeze(),
        cmap="gray",
    )
    plt.imsave(
        os.path.join(data_dir, "png", "label_%03d.png" % i),
        label.squeeze(),
        cmap="gray",
    )
##################################################################################
