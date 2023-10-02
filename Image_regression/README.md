# UNet_pytorch
reference
: https://www.youtube.com/@hanyoseob/videos  
</br>



---
## Dataset
- BSD 50 dataset 사용
  - https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- 구체적 경로는 run_image_regression.ipynb 참고
---

## How to use
run_image_regression.ipynb 파일 참조

## About files
### dataset.py
- 데이터셋 로드 및 transform 관련 

### display_data.py
- 데이터셋 시각화 위함. numpy형태의 데이터를 png로 변환해 확인할 수 있도록 함

### display_results.py
- test 결과로 생성된 result 디렉토리 내 파일들을 통해 테스트셋에 대한 결과를 시각화할 수 있음

### model.py
- UNet 모델, Autoencoder 모델 구현 (pytorch)

### train.py
- run_image_regression.ipynb를 통해 사용법 참고. train과 test가 가능하도록 함
- 사용 예시
```
!python "/content/drive/MyDrive/UNet_pytorch_regression/train.py" \
--mode "train" \
--num_epoch 40 \
--data_dir "/content/drive/MyDrive/UNet_pytorch_regression/data/BSR/BSDS500/data/images" \
--ckpt_dir "/content/drive/MyDrive/UNet_pytorch_regression/checkpoint/super_resolution/residual" \
--log_dir "/content/drive/MyDrive/UNet_pytorch_regression/log/super_resolution/residual" \
--result_dir "/content/drive/MyDrive/UNet_pytorch_regression/result/super_resolution/residual" \
--network "unet" \
--task "super_resolution" \
--opts "bilinear" 4.0 \
--learning_type "residual"
```

### util.py
- 모델 저장 및 로드 관련 함수
- Artifact 이미지 생성 관련 함수

### result_example
- 결과 예시
