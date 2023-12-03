# 1. RoBERTa & RegNetY Ensemble  
reference : https://dacon.io/competitions/official/235978/codeshare/7083?page=1&dtype=recent
- Tensorflow 기반 코드
  - Tensorflow에 대한 감을 익히기 위해 공개된 코드를 참고해 공부
  - 공개된 코드를 그대로 쓸 경우, 버전 오류가 발생해 일부 수정  
  - 텍스트 데이터를 최대한 활용하고, 이미지 데이터는 보조로 사용
  - GPU의 한계로 인해 작동 확인까지 마친 상태

- csv 파일 다루는 방법

- train data의 이미지, 텍스트 데이터를 증강해 validation set 생성  

- 텍스트데이터에 대한 전처리 및 데이터 증강  
  - 증강 기법 적용해 텍스트 데이터에 대한 class imbalance 문제 해결
  - 명사 앞에 형용사 붙여 증강
  - 동사 앞에 부사 붙여 증강
  - 유의어 사용해 증강

- 이미지 데이터 증강
  - 원본 이미지를 변형해 validation set으로 사용

- 각 증강 데이터를 별도의 csv파일로 저장  

- 이미지 데이터 전처리
  - 패딩 적용 및 종횡비 유지 부분에 유의

- RoBERTa 기반 텍스트 데이터의 각 증강 기법 별 모델 결과, RegNetY 기반 이미지 데이터의 모델 설정 별 결과를 사용해 voting 방식으로 Ensemble 학습기법 사용

## RoBERTa
  - klue/roberta-large 모델 사용
  - 텍스트 데이터를 입력으로 받아 분류 task를 수행하기 위함
  - 텍스트 증강 기법 별 모델 결과 활용

## RegNetY
  - imagenet으로 pretrain된 RegNetY120 모델 사용
  - 몇 개의 레이어에 대해 transfer learning 진행할 것인지에 차이를 둬 여러 모델 학습