import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import cv2
from utils.data_loader import DataLoader
from utils.mask_generator import MaskGenerator
from utils.visualization import Visualizer

st.title("Inference 결과 뷰어 📊")

def main():
    # 데이터 로더 초기화
    data_loader = DataLoader("../data/", mode='test')
    
    # CSV 파일 선택
    csv_files = [f for f in os.listdir("../") if f.endswith('_output.csv')]
    selected_csv = st.selectbox("CSV 파일 선택", csv_files)
    csv_path = os.path.join("..", selected_csv)
    
    # 이미지 파일 목록 가져오기
    image_files = data_loader.get_image_list()
    
    # 이미지 쌍 찾기
    image_pairs = data_loader.get_image_pairs(image_files)
    
    # 이미지 쌍 선택
    selected_pair = st.selectbox("이미지 쌍 선택", list(image_pairs.keys()))
    
    if selected_pair and selected_csv:
        # Left, Right 이미지를 나란히 표시하기 위한 컬럼 생성
        col1, col2 = st.columns(2)
        
        mask_generator = MaskGenerator()
        
        # Left 이미지 처리
        with col1:
            st.subheader("Left Image")
            image_l = cv2.imread(os.path.join(data_loader.images_dir, image_pairs[selected_pair]['L']))
            image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
            
            try:
                mask_l = mask_generator.load_and_process_masks(data_loader, csv_path, 
                                                             image_pairs[selected_pair]['L'].split('/')[-1], 
                                                             image_l.shape)
                rgb_mask_l = mask_generator.mask_to_rgb(mask_l)
                Visualizer.display_image_and_mask(image_l, rgb_mask_l)
                
                # 마스크 정보 표시
                st.write("Left 이미지 마스크 정보:")
                st.write(f"마스크 shape: {mask_l.shape}")
                st.write(f"유니크한 클래스: {np.unique(mask_l)}")
                st.write(f"마스크의 최소/최대값: {mask_l.min()}, {mask_l.max()}")
            except Exception as e:
                st.error(f"Left 이미지 마스크 처리 중 오류 발생: {str(e)}")
                st.write("RLE 인코딩된 원본 데이터:")
                mask_data = mask_generator.get_mask_data(csv_path, image_pairs[selected_pair]['L'].split('/')[-1])
                st.write(mask_data)
                st.image(image_l, caption="원본 이미지만 표시", use_column_width=True)
        
        # Right 이미지 처리
        with col2:
            st.subheader("Right Image")
            image_r = cv2.imread(os.path.join(data_loader.images_dir, image_pairs[selected_pair]['R']))
            image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
            
            try:
                mask_r = mask_generator.load_and_process_masks(data_loader, csv_path, 
                                                             image_pairs[selected_pair]['R'].split('/')[-1], 
                                                             image_r.shape)
                rgb_mask_r = mask_generator.mask_to_rgb(mask_r)
                Visualizer.display_image_and_mask(image_r, rgb_mask_r)
                
                # 마스크 정보 표시
                st.write("Right 이미지 마스크 정보:")
                st.write(f"마스크 shape: {mask_r.shape}")
                st.write(f"유니크한 클래스: {np.unique(mask_r)}")
                st.write(f"마스크의 최소/최대값: {mask_r.min()}, {mask_r.max()}")
            except Exception as e:
                st.error(f"Right 이미지 마스크 처리 중 오류 발생: {str(e)}")
                st.write("RLE 인코딩된 원본 데이터:")
                mask_data = mask_generator.get_mask_data(csv_path, image_pairs[selected_pair]['R'].split('/')[-1])
                st.write(mask_data)
                st.image(image_r, caption="원본 이미지만 표시", use_column_width=True)
    
    # if selected_image and selected_csv:
    #     # 이미지 로드
    #     image = cv2.imread(os.path.join(data_loader.images_dir, selected_image))
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    #     mask_generator = MaskGenerator()
    #     # 마스크 생성
    #     mask = mask_generator.load_and_process_masks(data_loader, csv_path, selected_image, image.shape)
        
    #     # RGB 마스크 생성
    #     rgb_mask = mask_generator.mask_to_rgb(mask)
        
    #     # 이미지와 마스크 시각화
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.image(image, caption="원본 이미지", use_column_width=True)
    #     with col2:
    #         st.image(rgb_mask, caption="Segmentation 결과", use_column_width=True)
        
    #     # Blend된 결과 표시
    #     alpha = 0.7
    #     blended = cv2.addWeighted(image, alpha, rgb_mask, 1-alpha, 0)
    #     st.image(blended, caption="Blended 결과", use_column_width=True)

if __name__ == "__main__":
    main()