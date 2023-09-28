# 필요한 패키지 import
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 데이터 불러오기
parser = argparse.ArgumentParser(
    description="Processing data_tif",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--dir_data", default="./data", type=str, dest="dir_data")

args = parser.parse_args()

dir_data = args.dir_data

name_label = "train-labels.tif"  # label, input 파일 이름
name_input = "train-volume.tif"

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size  # label 이미지의 사이즈 알기 위함
nframe = img_label.n_frames  # 레이블의 프레임 수

##
nframe_train = 24
nframe_val = 3
nframe_test = 3
# 24, 3, 3개의 프레임을 각각 train, val, test data로 사용

# 각 데이터셋이 속해있는 디렉토리 경로 변수 지정
dir_save_train = os.path.join(dir_data, "train")
dir_save_val = os.path.join(dir_data, "val")
dir_save_test = os.path.join(dir_data, "test")

# train, validation, test 데이터 디렉토리 생성 코드
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

##데이터 저장
# 랜덤하게 데이터 분배하기 위한 코드
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# trainset 저장

offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, "label_%03d.npy" % i), label_)
    np.save(os.path.join(dir_save_train, "input_%03d.npy" % i), input_)


# validation set 저장

offset_nframe = nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, "label_%03d.npy" % i), label_)
    np.save(os.path.join(dir_save_val, "input_%03d.npy" % i), input_)

# test set 저장
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, "label_%03d.npy" % i), label_)
    np.save(os.path.join(dir_save_test, "input_%03d.npy" % i), input_)

# label은 input의 segmentation map. 흰 부분이 1, 검은 부분이 0
