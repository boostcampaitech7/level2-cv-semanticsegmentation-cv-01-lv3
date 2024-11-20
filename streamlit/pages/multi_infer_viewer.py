import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import cv2
from utils.data_loader import DataLoader
from utils.mask_generator import MaskGenerator
from utils.visualization import Visualizer

st.set_page_config(layout="wide")  # wide 옵션 추가
st.title("다중 Inference 결과 비교 뷰어 📊")

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# 상수 정의
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

def get_distinct_colors(n):
    """뚜렷하게 구분되는 색상 목록 반환"""
    distinct_colors = [
        (255, 0, 0),    # 빨강
        (0, 255, 0),    # 초록
        (0, 0, 255),    # 파랑
        (255, 255, 0),  # 노랑
        (255, 0, 255),  # 마젠타
        (0, 255, 255),  # 시안
        (128, 0, 0),    # 진한 빨강
        (0, 128, 0),    # 진한 초록
        (0, 0, 128),    # 진한 파랑
    ]
    return distinct_colors[:n]

def overlay_multiple_masks(image, masks, colors=None, alpha=0.5, beta=0.1):
    """여러 마스크를 하나의 이미지에 오버레이 (각각 다른 색상으로)"""
    result = image.copy()
    if colors is None:
        colors = [get_distinct_colors(len(masks))[idx] 
                 for idx in range(len(masks))]
    
    # 모든 마스크의 색상을 합칠 배열 초기화
    colored_overlay = np.zeros_like(image, dtype=np.float32)
    
    for mask, color in zip(masks, colors):
        # 외곽선 추출
        contours = cv2.findContours(mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)[0]
        
        # 외곽선은 진하게
        cv2.drawContours(result, contours, -1, color, 2)
        
        # 내부 색상을 colored_overlay에 추가
        temp_mask = np.zeros_like(image, dtype=np.float32)
        cv2.fillPoly(temp_mask, contours, color)
        colored_overlay += temp_mask / 255.0  # 색상값을 0~1 범위로 정규화
    
    # 전체 오버레이를 0~255 범위로 클리핑
    colored_overlay = np.clip(colored_overlay, 0, 255).astype(np.uint8)
    
    # alpha를 사용하여 원본 이미지와 마스크를 블렌딩
    result = cv2.addWeighted(result, alpha, colored_overlay, beta, 0)
    
    return result

def main():
    # 데이터 로더 초기화
    data_loader = DataLoader("../data/", mode='test')
    
    # prediction 폴더 내의 여러 CSV 파일 선택
    prediction_dir = os.path.join("../", "prediction")
    csv_files = [f for f in os.listdir(prediction_dir) if f.endswith('.csv')]
    selected_csvs = st.multiselect("비교할 CSV 파일들 선택", csv_files)
    
    # 이미지 쌍 선택
    image_files = data_loader.get_image_list()
    image_pairs = data_loader.get_image_pairs(image_files)
    selected_pair = st.selectbox("이미지 쌍 선택", list(image_pairs.keys()))
    
    # 시각화 모드 선택 부분 수정
    view_mode = st.radio("시각화 모드 선택", 
                        ["마스크 중첩 모드", "나란히 비교 모드", "클래스별 비교 모드"])
    
    
    if selected_pair and selected_csvs:
        mask_generator = MaskGenerator()
        # CSV 파일별 고유 색상 생성
        csv_colors = {csv: get_distinct_colors(len(selected_csvs))[idx] 
                     for idx, csv in enumerate(selected_csvs)}
    
    if view_mode == "클래스별 비교 모드":
        st.subheader("클래스별 마스크 비교")
        
        # 클래스 선택
        selected_class = st.selectbox("클래스 선택", CLASSES)
    
        # Left 이미지
        image_l = cv2.imread(os.path.join(data_loader.images_dir, 
                                        image_pairs[selected_pair]['L']))
        image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
        
        # Right 이미지
        image_r = cv2.imread(os.path.join(data_loader.images_dir, 
                                        image_pairs[selected_pair]['R']))
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        # 클래스별 비교 모드 부분 수정
        with col1:
            st.write(f"Left Image - {selected_class}")
            masks_l = []
            for csv_file in selected_csvs:
                try:
                    mask = mask_generator.load_and_process_masks_by_class(
                        data_loader,
                        os.path.join(prediction_dir, csv_file),
                        image_pairs[selected_pair]['L'].split('/')[-1],
                        image_l.shape,
                        selected_class  # 클래스 이름 직접 전달
                    )
                    masks_l.append(mask)
                except Exception as e:
                    st.error(f"마스크 처리 중 오류 발생: {str(e)}")
            
            if masks_l:
                result_l = overlay_multiple_masks(image_l, masks_l, 
                                            list(csv_colors.values()), 
                                            beta=0.4)
                st.image(result_l, use_container_width=True)
                
                # 범례 표시
                st.write("📋 범례")
                for csv, color in csv_colors.items():
                    st.markdown(
                        f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                        f'<div style="width: 25px; height: 25px; background-color: rgb{color}; '
                        f'margin-right: 10px; border: 1px solid black;"></div>'
                        f'<span style="font-size: 16px;">{csv}</span></div>',
                        unsafe_allow_html=True
                    )
        
        # 클래스별 비교 모드 부분 수정
        with col2:
            st.write(f"Right Image - {selected_class}")
            masks_r = []
            for csv_file in selected_csvs:
                try:
                    mask = mask_generator.load_and_process_masks_by_class(
                        data_loader,
                        os.path.join(prediction_dir, csv_file),
                        image_pairs[selected_pair]['R'].split('/')[-1],
                        image_r.shape,
                        selected_class  # 클래스 이름 직접 전달
                    )
                    masks_r.append(mask)
                except Exception as e:
                    st.error(f"마스크 처리 중 오류 발생: {str(e)}")
            
            if masks_r:
                result_r = overlay_multiple_masks(image_r, masks_r, 
                                            list(csv_colors.values()), 
                                            beta=0.4)
                st.image(result_r, use_container_width=True)
                
                # 범례 표시
                st.write("📋 범례")
                for csv, color in csv_colors.items():
                    st.markdown(
                        f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                        f'<div style="width: 25px; height: 25px; background-color: rgb{color}; '
                        f'margin-right: 10px; border: 1px solid black;"></div>'
                        f'<span style="font-size: 16px;">{csv}</span></div>',
                        unsafe_allow_html=True
                    )
        
       
    elif view_mode == "마스크 중첩 모드":
        # 전체 화면 너비 사용
        st.subheader("마스크 중첩 비교")
        
        # Left 이미지
        image_l = cv2.imread(os.path.join(data_loader.images_dir, 
                                        image_pairs[selected_pair]['L']))
        image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
        
        # Right 이미지
        image_r = cv2.imread(os.path.join(data_loader.images_dir, 
                                        image_pairs[selected_pair]['R']))
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
    
            # 뚜렷한 색상 생성
        distinct_colors = get_distinct_colors(len(selected_csvs))
        csv_colors = {csv: color for csv, color in zip(selected_csvs, distinct_colors)}
        
        # Left 이미지 처리
        col1, col2 = st.columns([3, 1])  # 3:1 비율로 컬럼 분할
        
        with col1:
            st.write("Left Image")
            masks_l = []
            for csv_file in selected_csvs:
                try:
                    mask = mask_generator.load_and_process_masks(
                        data_loader,
                        os.path.join(prediction_dir, csv_file),
                        image_pairs[selected_pair]['L'].split('/')[-1],
                        image_l.shape
                    )
                    masks_l.append(mask)
                except Exception as e:
                    st.error(f"마스크 처리 중 오류 발생: {str(e)}")
            
            if masks_l:
                result_l = overlay_multiple_masks(image_l, masks_l, 
                                            list(csv_colors.values()), 
                                            beta=0.4)  # 투명도 조정
                st.image(result_l, use_container_width=True)
        
        with col2:
            # 범례를 더 눈에 띄게 표시
            st.write("📋 범례")
            for csv, color in csv_colors.items():
                st.markdown(
                    f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                    f'<div style="width: 25px; height: 25px; background-color: rgb{color}; '
                    f'margin-right: 10px; border: 1px solid black;"></div>'
                    f'<span style="font-size: 16px;">{csv}</span></div>',
                    unsafe_allow_html=True
                )
        
        # Left 이미지 처리
        col1, col2 = st.columns([3, 1])  # 3:1 비율로 컬럼 분할
        
        with col1:
            st.write("Right Image")
            # Right 이미지도 동일한 방식으로 처리
            masks_r = []
            for csv_file in selected_csvs:
                try:
                    mask = mask_generator.load_and_process_masks(
                        data_loader,
                        os.path.join(prediction_dir, csv_file),
                        image_pairs[selected_pair]['R'].split('/')[-1],
                        image_r.shape
                    )
                    masks_r.append(mask)
                except Exception as e:
                    st.error(f"마스크 처리 중 오류 발생: {str(e)}")
            
            if masks_r:
                result_r = overlay_multiple_masks(image_r, masks_r, 
                                            list(csv_colors.values()), 
                                            beta=0.4)
                st.image(result_r, use_container_width=True)
                
        with col2:
            # 범례를 더 눈에 띄게 표시
            st.write("📋 범례")
            for csv, color in csv_colors.items():
                st.markdown(
                    f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                    f'<div style="width: 25px; height: 25px; background-color: rgb{color}; '
                    f'margin-right: 10px; border: 1px solid black;"></div>'
                    f'<span style="font-size: 16px;">{csv}</span></div>',
                    unsafe_allow_html=True
                )
        
    else:  # 나란히 비교 모드
        num_cols = len(selected_csvs) + 1
        cols = st.columns(num_cols)
            
        # Left 이미지 세트
        st.subheader("Left Image Set")
        with cols[0]:
            st.write("Original")
            image_l = cv2.imread(os.path.join(data_loader.images_dir, 
                                            image_pairs[selected_pair]['L']))
            image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
            st.image(image_l, use_container_width=True)
        
        # 각 CSV 파일별 마스크
        for idx, csv_file in enumerate(selected_csvs, 1):
            with cols[idx]:
                st.write(f"Mask: {csv_file}")
                try:
                    mask_l = mask_generator.load_and_process_masks(
                        data_loader,
                        os.path.join(prediction_dir, csv_file),
                        image_pairs[selected_pair]['L'].split('/')[-1],
                        image_l.shape
                    )
                    # 원본 이미지에 마스크 오버레이
                    result = overlay_multiple_masks(
                        image_l, [mask_l], 
                        [csv_colors[csv_file]], 
                        beta=0.5
                    )
                    st.image(result, use_container_width=True)
                except Exception as e:
                    st.error(f"마스크 처리 중 오류 발생: {str(e)}")
        
        # Right 이미지 세트 (Left와 동일한 로직)
        st.subheader("Right Image Set")
        with cols[0]:
            st.write("Original")
            image_r = cv2.imread(os.path.join(data_loader.images_dir, 
                                            image_pairs[selected_pair]['R']))
            image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
            st.image(image_r, use_container_width=True)
        
        for idx, csv_file in enumerate(selected_csvs, 1):
            with cols[idx]:
                st.write(f"Mask: {csv_file}")
                try:
                    mask_r = mask_generator.load_and_process_masks(
                        data_loader,
                        os.path.join(prediction_dir, csv_file),
                        image_pairs[selected_pair]['R'].split('/')[-1],
                        image_r.shape
                    )
                    result = overlay_multiple_masks(
                        image_r, [mask_r], 
                        [csv_colors[csv_file]], 
                        beta=0.5
                    )
                    st.image(result, use_container_width=True)
                except Exception as e:
                    st.error(f"마스크 처리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()