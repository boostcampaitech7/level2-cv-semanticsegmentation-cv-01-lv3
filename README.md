## Hand Bone Image Segmentation

뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

질병 진단의 목적으로 뼈의 형태나 위치가 변형되거나 부러지거나 골절 등이 있을 경우, 그 부위에서 발생하는 문제를 정확하게 파악하여 적절한 치료를 시행할 수 있습니다.

수술 계획을 세우는데 도움이 됩니다. 의사들은 뼈 구조를 분석하여 어떤 종류의 수술이 필요한지, 어떤 종류의 재료가 사용될 수 있는지 등을 결정할 수 있습니다.

의료장비 제작에 필요한 정보를 제공합니다. 예를 들어, 인공 관절이나 치아 임플란트를 제작할 때 뼈 구조를 분석하여 적절한 크기와 모양을 결정할 수 있습니다.

의료 교육에서도 활용될 수 있습니다. 의사들은 병태 및 부상에 대한 이해를 높이고 수술 계획을 개발하는 데 필요한 기술을 연습할 수 있습니다.

![image](https://github.com/user-attachments/assets/f7ee7a87-b032-4c5e-b391-438d08b79fe9)


여러분에 의해 만들어진 우수한 성능의 모델은 질병 진단, 수술 계획, 의료 장비 제작, 의료 교육 등에 사용될 수 있을 것으로 기대됩니다. 🌎


## Data

Input : 
hand bone x-ray 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. segmentation annotation은 json file로 제공됩니다.

Output :
모델은 각 클래스(29개)에 대한 확률 맵을 갖는 멀티채널 예측을 수행하고, 이를 기반으로 각 픽셀을 해당 클래스에 할당합니다.

Submission :
최종적으로 예측된 결과를 Run-Length Encoding(RLE) 형식으로 변환하여 csv 파일로 제출합니다.

## Usage

### Installation

1. Clone the repository & download data:
   ```
   git clone https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-01-lv3.git
   cd level2-cv-semanticsegmentation-cv-01-lv3.git
   ```

### Training

```
python train.py --data_dir=./data --batch_size=8 learning_rate=1e-6 --max_epoch=50
```

hyperparameter는 원하는 만큼 수정할 수 있습니다.

### Inference


```
python inference.py --data_dir=./data --batch_size=8 
```


### Project Structure

```
** project structure **

level2-cv-semanticsegmentation-cv-01-lv3/
│
├── data/
│   ├── train/
│   │    ├── DCM / ID ###
│   │    └── output_json / ID ###
│   │
│   └── test /
│        └── DCM / ID ###
├──utils/
│   ├── dataset.py
│   ├── method.py
│   ├── augmentation.py
│   ├── handrotation.py
│   ├── hard_voting.py
│   ├── soft_voting.py
│   ├── trainer.py
│   └── visualization.py
│
├── inference.py
├── train.py
├── requirements.txt
└── README.md
```

   
