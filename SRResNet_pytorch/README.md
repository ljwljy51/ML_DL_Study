# SRResNet_pytorch
reference
: https://www.youtube.com/@hanyoseob/videos  
</br>



---
## Dataset
- BSD 50 dataset 사용
  - https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- 구체적 경로는 run_train_super_resolution.ipynb 참고
---

## How to use
run_train_super_resolution.ipynb 파일 참조

## About files
### dataset.py
- 데이터셋 로드 및 transform 관련 

### display_data.py (사용 x)
- 데이터셋 시각화 위함. numpy형태의 데이터를 png로 변환해 확인할 수 있도록 함

### display_results.py (사용 x)
- test 결과로 생성된 result 디렉토리 내 파일들을 통해 테스트셋에 대한 결과를 시각화할 수 있음

### layer.py
- 모델 구현에 필요한 자주 사용되는 기본 layer 구현
- CBR block, ResBlock, PixelShuffler, PixelUnshuffler 

### model.py
- UNet 모델, Autoencoder 모델, ResNet 모델, SRResNet 모델 구현 (pytorch)

### train.py
- run_train_super_resolution.ipynb를 통해 사용법 참고. train과 test가 가능하도록 함
- 사용 예시
```
!python "/content/drive/MyDrive/SRResNet/train.py" \
--mode "train" \
--train_continue "on" \
--batch_size 4 \
--data_dir "/content/drive/MyDrive/SRResNet/datasets/BSR/BSDS500/data/images" \
--ckpt_dir "/content/drive/MyDrive/SRResNet/checkpoint/srresnet/super_resolution" \
--log_dir "/content/drive/MyDrive/SRResNet/log/srresnet/super_resolution" \
--result_dir "/content/drive/MyDrive/SRResNet/result/srresnet/super_resolution" \
--network "srresnet" \
--task "super_resolution" \
--opts "bilinear" 4.0 0 \
--learning_type "residual"
```

### util.py
- 모델 저장 및 로드 관련 함수
- Artifact 이미지 생성 관련 함수

### result_example
- 결과 예시
