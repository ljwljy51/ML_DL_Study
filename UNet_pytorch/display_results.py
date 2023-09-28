import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(
    description="Display result_numpy format",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--result_dir", default="./results/numpy", type=str, dest="result_dir"
)
args = parser.parse_args()

result_dir = args.result_dir

# numpy 형식 데이터 로드 위함
lst_data = os.listdir(result_dir)

# 파일 각각 분리
lst_label = [f for f in lst_data if f.startswith("label")]
lst_input = [f for f in lst_data if f.startswith("input")]
lst_output = [f for f in lst_data if f.startswith("output")]

lst_label.sort()
lst_input.sort()
lst_output.sort()

id = 0  # n번째 슬라이스 불러오는 경우

label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

# 시각화
plt.subplot(131)
plt.imshow(input, cmap="gray")
plt.title("input")

plt.subplot(132)
plt.imshow(output, cmap="gray")
plt.title("output")

plt.subplot(133)
plt.imshow(label, cmap="gray")
plt.title("label")
plt.show()
