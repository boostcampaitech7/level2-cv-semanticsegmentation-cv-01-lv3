import cv2
import numpy as np
import json
import sys
import os
import argparse

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import CLASSES, CLASS2IND


def load_mask_from_json(json_path, height, width):
    """
    JSON 파일에서 마스크를 로드하는 함수
    """
    with open(json_path, 'r') as f:
        ann = json.load(f)
    
    # 클래스 수만큼의 채널을 가진 마스크 초기화
    mask = np.zeros((height, width, len(CLASSES)), dtype=np.uint8)
    
    # annotations가 리스트인 경우
    if isinstance(ann.get('annotations', []), list):
        for annotation in ann['annotations']:
            # points를 numpy 배열로 변환
            points = np.array(annotation['points'], dtype=np.int32)
            
            # points가 비어있지 않은지 확인
            if len(points) > 0:
                # label을 사용하여 클래스 인덱스 찾기
                class_name = annotation['label']
                class_idx = CLASS2IND[class_name]  # CLASS2IND를 사용하여 정확한 인덱스 매핑
                
                # 해당 클래스의 마스크 채널을 가져와서 contiguous array로 만듦
                class_mask = np.ascontiguousarray(mask[:, :, class_idx])
                # points를 [N,1,2] 형태로 재구성
                points = points.reshape((-1, 1, 2))
                # fillPoly 적용
                cv2.fillPoly(class_mask, [points], 1)
                # 결과를 다시 마스크에 저장
                mask[:, :, class_idx] = class_mask
    
    return mask

def get_lunate_bottom_point(mask):
    """Lunate의 최하단 지점을 찾는 함수"""
    lunate_idx = CLASS2IND['Lunate']
    lunate_mask = mask[:, :, lunate_idx]
    
    y_coords, x_coords = np.where(lunate_mask > 0)
    if len(y_coords) == 0:
        raise ValueError("Lunate mask is empty")
    
    bottom_y = np.max(y_coords)
    bottom_x = int(np.mean(x_coords[y_coords == bottom_y]))
    
    return (float(bottom_x), float(bottom_y))

def rotate_hand_selective(image, json_path, angle):
    height, width = image.shape[:2]
    
    # 마스크 로드
    mask = load_mask_from_json(json_path, height, width)
    
    # Lunate의 최하단 지점 찾기 (원본 위치)
    original_center_x, original_center_y = get_lunate_bottom_point(mask)
    print(f"Original center (Lunate bottom): ({original_center_x}, {original_center_y})")
    
    # 1단계: 회전할 클래스와 고정할 클래스 분리
    fixed_classes = ['Radius', 'Ulna']
    fixed_indices = [CLASS2IND[c] for c in fixed_classes]
    
    # 고정된 클래스의 마스크와 이미지
    fixed_mask = np.zeros((height, width), dtype=np.uint8)
    for idx in fixed_indices:
        fixed_mask = np.logical_or(fixed_mask, mask[:, :, idx] > 0)
    fixed_area = cv2.bitwise_and(image, image, mask=fixed_mask.astype(np.uint8))
    
    # 회전할 클래스의 마스크와 이미지
    rotate_mask = np.zeros((height, width), dtype=np.uint8)
    rotate_area = np.zeros_like(image)
    rotated_mask_temp = np.zeros_like(mask)
    
    # 회전할 클래스들만 분리
    for i in range(mask.shape[2]):
        if i not in fixed_indices:
            rotate_mask = np.logical_or(rotate_mask, mask[:, :, i] > 0)
    rotate_area = cv2.bitwise_and(image, image, mask=rotate_mask.astype(np.uint8))
    
    # 2단계: 회전할 부분만 회전 (원본 크기 유지)
    rotation_matrix = cv2.getRotationMatrix2D((original_center_x, original_center_y), angle, 1.0)
    
    # 회전 적용 (원본 크기로)
    rotated_area = cv2.warpAffine(rotate_area, rotation_matrix, (width, height))
    
    # 회전할 클래스들의 마스크 회전
    for i in range(len(CLASSES)):
        if i not in fixed_indices:
            class_mask = mask[:, :, i]
            rotated_class = cv2.warpAffine(class_mask, rotation_matrix, (width, height),
                                         flags=cv2.INTER_NEAREST)
            rotated_mask_temp[:, :, i] = rotated_class
    
    # 3단계: 회전된 Lunate의 최하단 지점 찾기
    rotated_lunate = rotated_mask_temp[:, :, CLASS2IND['Lunate']]
    y_coords, x_coords = np.where(rotated_lunate > 0)
    if len(y_coords) > 0:
        rotated_center_x = int(np.mean(x_coords[y_coords == np.max(y_coords)]))
        rotated_center_y = np.max(y_coords)
        
        # 이동 거리 계산
        dx = original_center_x - rotated_center_x
        dy = original_center_y - rotated_center_y
        
        # 이동 행렬 생성
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # 회전된 부분을 원래 위치로 이동
        rotated_area = cv2.warpAffine(rotated_area, translation_matrix, (width, height))
        for i in range(len(CLASSES)):
            if i not in fixed_indices:
                rotated_mask_temp[:, :, i] = cv2.warpAffine(rotated_mask_temp[:, :, i],
                                                          translation_matrix,
                                                          (width, height),
                                                          flags=cv2.INTER_NEAREST)
    
    # 최종 이미지 합성
    final_image = fixed_area.copy()
    final_mask = np.zeros_like(mask)
    
    # 고정된 클래스 복사
    for idx in fixed_indices:
        final_mask[:, :, idx] = mask[:, :, idx]
    
    # 회전된 클래스 합성
    non_zero_mask = rotated_area.sum(axis=2) > 0
    final_image[non_zero_mask] = rotated_area[non_zero_mask]
    
    # 회전된 마스크 합성
    for i in range(len(CLASSES)):
        if i not in fixed_indices:
            final_mask[:, :, i] = rotated_mask_temp[:, :, i]
    
    return final_image, final_mask

def get_bone_center_x(mask, bone_name):
    """특정 뼈의 중심 x 좌표를 찾는 함수"""
    bone_idx = CLASS2IND[bone_name]
    bone_mask = mask[:, :, bone_idx]
    y_coords, x_coords = np.where(bone_mask > 0)
    if len(x_coords) == 0:
        raise ValueError(f"{bone_name} mask is empty")
    return np.mean(x_coords)

def determine_hand_side(mask):
    """
    radius와 ulna의 상대적 위치로 왼손/오른손 판단
    radius의 x좌표가 ulna의 x좌표보다 작으면 오른손, 크면 왼손
    """
    radius_x = get_bone_center_x(mask, 'Radius')
    ulna_x = get_bone_center_x(mask, 'Ulna')
    
    print(f"Radius center x: {radius_x}, Ulna center x: {ulna_x}")
    return 'R' if radius_x < ulna_x else 'L'

def get_rotation_angle(image_path, mask):
    """
    손의 방향을 판단하여 회전 각도 결정
    R(오른손): 시계방향 45도
    L(왼손): 반시계방향 45도
    """
    try:
        # 먼저 파일명으로 시도
        if image_path.endswith('_R.png'):
            hand_side = 'R'
        elif image_path.endswith('_L.png'):
            hand_side = 'L'
        else:
            # 파일명으로 판단할 수 없는 경우 radius/ulna 위치로 판단
            hand_side = determine_hand_side(mask)
            print(f"Determined hand side from bone positions: {hand_side}")
    except Exception as e:
        print(f"Error in filename detection: {e}")
        # 에러 발생시 radius/ulna 위치로 판단
        hand_side = determine_hand_side(mask)
        print(f"Determined hand side from bone positions: {hand_side}")
    
    # 회전 각도 결정
    angle = -45 if hand_side == 'R' else 45
    print(f"Rotation angle for {hand_side} hand: {angle} degrees")
    return angle

# 테스트 코드
if __name__ == "__main__":
    # ArgumentParser 설정
    parser = argparse.ArgumentParser(description='Hand image rotation preprocessing')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image file')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to the input JSON annotation file')
    
    args = parser.parse_args()
    
    # 이미지 로드
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {args.image_path}")
    
    # 마스크 로드
    mask = load_mask_from_json(args.json_path, image.shape[0], image.shape[1])
    
    # 회전 각도 결정
    angle = get_rotation_angle(args.image_path, mask)
    
    # 회전 테스트
    rotated_image, rotated_mask = rotate_hand_selective(image, args.json_path, angle)
