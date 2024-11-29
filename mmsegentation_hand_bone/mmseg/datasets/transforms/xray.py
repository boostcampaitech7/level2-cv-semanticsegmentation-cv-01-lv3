import os
import json

import numpy as np
import cv2

from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

@TRANSFORMS.register_module()
class LoadXRayAnnotations(BaseTransform):
    def transform(self, result):
        # 라벨 파일 경로 가져오기
        label_path = result["seg_map_path"]

        # 이미지 크기 설정 
        # 원본 이미지(2048x2048)를 3/4 크기로 줄임
        # 메모리 사용량과 학습 속도를 고려하여 1536x1536으로 설정
        image_size = (2048, 2048)

        # 라벨 텐서 초기화 (높이, 너비, 클래스 수)
        # 각 클래스별 마스크를 담을 3차원 배열 생성
        label_shape = image_size + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # JSON 형식의 라벨 파일 읽기
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # 각 클래스별 어노테이션 처리
        for ann in annotations:
            # 클래스 이름과 인덱스 가져오기
            c = ann["label"]
            class_ind = CLASS2IND[c]
            # 폴리곤 포인트 좌표 배열로 변환
            points = np.array(ann["points"])

            # 폴리곤을 마스크로 변환
            # 해당 클래스의 영역을 1로 채움
            class_label = np.zeros(image_size, dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        # 결과 딕셔너리에 생성된 마스크 저장
        result["gt_seg_map"] = label

        return result

@TRANSFORMS.register_module()
class TransposeAnnotations(BaseTransform):
    def transform(self, result):
        result["gt_seg_map"] = np.transpose(result["gt_seg_map"], (2, 0, 1))

        return result

