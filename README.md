# BoxDetection-Yolov5

# 사용방법
## 설치방법 1
https://github.com/KAI-Devv/BoxDetection-Yolov5.git<br>
cd BoxDetection-Yolov5 <br>
pip install -r requirements.txt <br>

## 설치방법 2 (Docker를 활용한 설치)
아래 링크를 다운로드 하여 도커 설치를 따라해주시면 됩니다.
1) 철도 선로 데이터셋 : https://github.com/KAI-Devv/BoxDetection-Yolov5/files/10401838/_yolo_225.docx
2) 전차선 / 애자 데이터셋 : https://github.com/KAI-Devv/BoxDetection-Yolov5/files/10401839/_yolo_226.docx <br>

## 데이터 전처리
데이터 전처리 자동화 스크립트를 통해 YOLOv5에서 학습 가능한 라벨렝 데이터 포맷으로 변환하고,
Training:Validation:Test 데이터 비율을 8:1:1로 랜덤하게 분배하여 저장합니다.
탐지 객체 이름과 id 정보가 포함된 데이터셋 별 메타데이터 파일은 해당 파일을 참고 바랍니다.
1) 철도 선로 데이터셋 : preprocess/railway_metadata.json<br>
2) 전차선 / 애자 데이터셋 : preprocess/catenary_metadata.json><br><br>

아래 스크립트를 통해 전처리를 수행합니다.<br>
```
python preprocess/preprocess.py {데이터셋 폴더 경로} {메타데이터 파일 경로}<br><br>
```
해당 스크립트가 수행되면 {데이터셋 폴더 이름}에 '_data'라는 suffix를 가진 폴더가 생성되고. Training:Validation:Test 데이터를 분리합니다. <br>

아래 스크립트를 통해 두번째 전처리를 수행합니다.<br>
```
python preprocess/preprocess2.py {데이터셋 폴더 경로}_data {메타데이터 파일 경로}<br><br>
```
이는 yolov5 포맷에 맞는 라벨링 데이터 .txt 파일을 이미지별로 생성하고, 클래스 정보를 포함한 내용을 {데이터셋 폴더 이름}_data/data.yaml 파일에 저장됩니다.<br>

예시)
1) 철도 선로 데이터셋 <br>
    ```
    python preprocess/preprocess.py ../../../dataset/railway preprocess/railway_metadata.json <br>
    python preprocess/preprocess2.py ../../../dataset/railway_data preprocess/railway_metadata.json
    ```
2) 전차선 / 애자 데이터셋 <br>
    ```
    python preprocess/preprocess.py ../../../dataset/catenary preprocess/catenary_metadata.json <br>
    python preprocess/preprocess2.py ../../../dataset/catenary_data preprocess/catenary_metadata.json
    ```

## 학습
```
python -m torch.distributed.run --nproc_per_node {GPU 개수} --master_port 1 train.py --data {preprocess2.py를 통해 획득된 data.yaml 파일의 경로} --img {이미지 width값} --weights {모델 파일 경로} --device {GPU index list} --batch-size {batch 수} --epochs {epoch 수}
```
예시)
```
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data ../../../datasets/railway/railway_data/data.yaml --weights railway.pt --device 0,1,2,3 --batch-size 64 --epochs 100
```
## 유효성 검증
```
python validate-railway-dataset.py --data {preprocess2.py를 통해 획득된 data.yaml 파일의 상위 폴더} --data_type {railway or catenary} --weights {모델 파일 경로} --task test
```
예시)
```
python validate-railway-dataset.py --data ../../../dataset/railway_data --data_type railway --weights railway.pt --task test
```

## 테스트 및 적용
To write the results to the new file, add progress bar status, save both concatenated and resulting video in desired path's, use newdet.py. Otherwise detect.py <br>
```
python detect.py(newdet.py) --weights {모델 파일 경로} --source {적용 파일 경로 (파일 또는 폴더)} --data {preprocess2.py를 통해 획득된 data.yaml 파일의 경로}
```
<br><br><br>

# 모델 정보
## 모델 Description
YOLOv5 모델 <br>

## 모델 아키텍쳐
New CSP-Darknet53 백본, SPPF 및 New CSP-PAN Neck, YOLOv3 Head로 구성되어 있으며 도식은 아래와 같습니다 (yolov5l 모델 기준) <br><br>
<img width="850" src="https://user-images.githubusercontent.com/41655836/212057352-7f7fbf47-d18f-4dc5-b00b-a4dda2d37764.png"></a>

## 모델 입력값
모델 입력값으로 (B, H, W, 3)의 Tensor를 사용합니다. B는 Batch Size, H는 높이, W는 넓이값입니다. <br>

## 모델 출력값
모델 출력값으로 (6, 1) Shape의 바운딩박스, Confidence, Class 번호가 리스트형태로 출력됩니다. <br>
각 텐서값은 순서대로 (min_x, min_y, max_x, max_y, confidence, class_num)를 정의합니다.

## 모델 TASK
객체 탐지 <br>

## Training Dataset
학습 데이터는 이미지 파일과 이미지 파일에서 감지할 객체의 라벨링 데이터입니다.
이미지 파일은 .jpg 또는 .png 파일을 사용합니다. 라벨링 데이터는 .txt 파일 형태이며, 카테고리 id와 normalized된 center x, y 및 width, height값으로 구성됩니다.
preprocess.py 및 preprocess2.py 스크립트를 이용하여 철도 선로 및 전차선/애자 데이터 셋을 해당 포맷으로 자동으로 변환시켜 주어야 합니다.

## Configurations
학습을 위해 batch size는 64, SGD optimizer, BCE + CIoU Loss, learning rate 0.01을 사용하였습니다.

## Evaluation
mAP (0.75 IoU 기준)
1) 철도 선로 데이터셋 : 0.8741
2) 전차선 / 애자 데이터셋 : 0.8622

## Reference
YOLOv5모델을 사용하였으며, 학습데이터 적용 및 유효성 검증 수행을 위한 코드의 커스터마이징 작업이 수행되었습니다. <br>
관련 링크: https://github.com/ultralytics/yolov5

## License
SPDX-FileCopyrightText: © 2022 Seunghwa Jeong <<sh.jeong@kaistudio.co.kr>> <br>
SPDX-License-Identifier: GPL-3.0 <br>

YOLOv5에서 명시된 GPL-3.0 License를 따릅니다. <br>
(GPL-3.0 Licence 전문: https://github.com/ultralytics/yolov5/blob/master/LICENSE)