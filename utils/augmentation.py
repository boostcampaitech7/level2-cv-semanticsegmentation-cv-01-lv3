import cv2
import numpy as np
import sys
import os
from albumentations.core.transforms_interface import BasicTransform

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import CLASSES, CLASS2IND
from utils.preprocessing import get_bone_center_x, get_lunate_bottom_point

# 제외할 ID 목록을 상수로 정의
EXCLUDE_IDS = {294, 466, 277, 391, 548, 310, 319, 321, 281, 285, 276, 308, 306, 486, 279, 313, 
              282, 296, 405, 309, 289, 293, 303, 291, 361, 314, 290, 274, 302, 362, 362, 288, 
              317, 278, 298, 315, 295, 284, 312, 424, 301, 437, 316, 470, 311, 299, 286, 275, 
              292, 297, 305, 469, 408, 304, 76, 283}

class HandRotationAugmentation(BasicTransform):
    """
    손 이미지 회전 augmentation
    - 오른손: 시계방향 회전 (-45도)
    - 왼손: 반시계방향 회전 (45도)
    - Radius와 Ulna는 고정
    - 나머지 부분은 Lunate 기준으로 회전
    """
    def __init__(self, rotation_range=15, always_apply=False, p=0.5):
        # p 값을 먼저 저장
        self.rotation_range = rotation_range
        # 부모 클래스 초기화
        super().__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def determine_hand_side(self, mask):
        """radius와 ulna의 상대적 위치로 왼손/오른손 판단"""
        radius_x = get_bone_center_x(mask, 'Radius')
        ulna_x = get_bone_center_x(mask, 'Ulna')
        return 'R' if radius_x < ulna_x else 'L'

    def get_rotation_angle(self, hand_side):
        """손의 방향에 따른 회전 각도 결정"""
        base_angle = -45 if hand_side == 'R' else 45  # 오른손은 시계방향, 왼손은 반시계방향
        random_variation = np.random.uniform(-self.rotation_range, self.rotation_range)
        return base_angle + random_variation

    def apply(self, image, **params):
        try:
            mask = params.get('mask')
            if mask is None:
                return image

            height, width = image.shape[:2]
            hand_side = self.determine_hand_side(mask)
            angle = self.get_rotation_angle(hand_side)
            center_x, center_y = get_lunate_bottom_point(mask)

            # 고정할 부분과 회전할 부분 분리
            fixed_classes = ['Radius', 'Ulna']
            fixed_indices = [CLASS2IND[c] for c in fixed_classes]

            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            rotated_area = cv2.warpAffine(image, rotation_matrix, (width, height))

            # 최종 이미지 합성
            final_image = np.where(
                np.any(mask[:, :, fixed_indices] > 0, axis=2)[..., None],
                image,
                rotated_area
            )
            return final_image

        except Exception as e:
            print(f"Image augmentation failed: {e}")
            return image

    def apply_to_mask(self, mask, **params):
        try:
            height, width = mask.shape[:2]
            hand_side = self.determine_hand_side(mask)
            angle = self.get_rotation_angle(hand_side)
            center_x, center_y = get_lunate_bottom_point(mask)

            fixed_classes = ['Radius', 'Ulna']
            fixed_indices = [CLASS2IND[c] for c in fixed_classes]

            rotated_mask = np.zeros_like(mask)
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

            for i in range(len(CLASSES)):
                if i in fixed_indices:
                    rotated_mask[:, :, i] = mask[:, :, i]
                else:
                    rotated_mask[:, :, i] = cv2.warpAffine(
                        mask[:, :, i],
                        rotation_matrix,
                        (width, height),
                        flags=cv2.INTER_NEAREST
                    )
            return rotated_mask

        except Exception as e:
            print(f"Mask augmentation failed: {e}")
            return mask

    def get_transform_init_args_names(self):
        return ("rotation_range",)