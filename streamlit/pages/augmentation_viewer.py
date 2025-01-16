import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import cv2
from utils.data_loader import DataLoader
from utils.mask_generator import MaskGenerator
from utils.visualization import Visualizer
from utils.augmentation import Augmentations

st.title("증강 효과 뷰어 🔄")

def main():
    # 데이터 로더 초기화
    data_loader = DataLoader("../data/")
    aug = Augmentations()
    
    # 이미지 파일 목록 가져오기
    image_files = data_loader.get_image_list()
    image_pairs = data_loader.get_image_pairs(image_files)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 이미지 쌍 선택
        selected_pair = st.selectbox("이미지 쌍 선택", list(image_pairs.keys()))
    
    with col2:
        # 증강 방법 선택
        transform_type = st.selectbox("증강 방법 선택", list(aug.transforms.keys()))
    
    if selected_pair:
        # 원본 이미지와 마스크
        image = data_loader.load_image(image_pairs[selected_pair]['L'])
        image = np.array(image)
        
        # Grayscale 이미지를 RGB로 변환
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        json_path = data_loader.get_json_path(image_pairs[selected_pair]['L'])
        
        if os.path.exists(json_path):
            json_data = data_loader.load_json(json_path)
            mask = MaskGenerator.create_mask_from_json(json_data, np.array(image).shape)
            rgb_mask = MaskGenerator.mask_to_rgb(mask)
            
            # 원본 이미지와 마스크 표시
            st.subheader("원본 이미지와 마스크")
            Visualizer.display_image_and_mask(image, rgb_mask)
            
            if st.button("증강 적용"):
                # 이미지와 마스크에 동일한 증강 적용
                augmented_image = aug.apply_transform(np.array(image), transform_type)
                augmented_mask = aug.apply_transform(mask, transform_type)
                rgb_augmented_mask = MaskGenerator.mask_to_rgb(augmented_mask)
                
                # 증강된 이미지와 마스크 표시
                st.subheader("증강된 이미지와 마스크")
                Visualizer.display_image_and_mask(augmented_image, rgb_mask)
                
                # 증강된 이미지 다운로드 버튼
                if st.button("증강된 결과 다운로드"):
                    # 이미지와 마스크를 나란히 저장
                    combined_image = np.hstack((augmented_image, rgb_mask))
                    st.download_button(
                        label="결과 저장",
                        data=combined_image.tobytes(),
                        file_name=f"augmented_{transform_type}.png",
                        mime="image/png"
                    )
        else:
            st.error("선택한 이미지의 어노테이션 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()