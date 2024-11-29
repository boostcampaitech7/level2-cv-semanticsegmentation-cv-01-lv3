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

def overlay_multiple_masks_from_rle(image, data_loaders, csv_paths, image_name, image_shape, alpha=0.9, beta=0.9, csv_colors=None, selected_class=None):
    """
    여러 CSV의 RLE 인코딩된 마스크들을 하나의 이미지에 오버레이하는 함수
    selected_class: 특정 클래스만 표시하고 싶을 때 사용
    """
    result = image.copy()
    colored_overlay = np.zeros_like(image, dtype=np.float32)
    
    # 각 CSV 파일별로 처리
    for csv_path, csv_color in zip(csv_paths, csv_colors):
        df = data_loaders.load_inference_csv(csv_path)
        image_masks = df[df['image_name'] == image_name]
        
        # 특정 클래스만 필터링
        if selected_class is not None:
            image_masks = image_masks[image_masks['class'] == selected_class]
        
        # 각 클래스별로 마스크 생성
        for _, row in image_masks.iterrows():
            rle = row['rle']
            
            # RLE 디코딩하여 마스크 생성
            mask = MaskGenerator.decode_rle_to_mask(rle, image_shape[0], image_shape[1])
            
            # 마스크에 대한 윤곽선 처리
            contours = cv2.findContours(mask.astype(np.uint8), 
                                      cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)[0]
            
            # 외곽선 그리기
            cv2.drawContours(result, contours, -1, csv_color, 2)
            
            # 내부 색상 처리
            temp_mask = np.zeros_like(image, dtype=np.float32)
            cv2.fillPoly(temp_mask, contours, csv_color)
            
            # 마스크 누적 (투명도 조절을 위해 가중치 적용)
            colored_overlay += temp_mask * (beta / len(csv_paths))
    
    # 전체 오버레이를 0~255 범위로 클리핑
    colored_overlay = np.clip(colored_overlay, 0, 255).astype(np.uint8)
    
    # 최종 블렌딩
    result = cv2.addWeighted(result, alpha, colored_overlay, 1.0, 0)
    
    return result

def main():
    # 데이터 로더 초기화
    data_loader = DataLoader("../data/", mode='test')
    
    # 사이드바에 설정 요소들 배치
    with st.sidebar:
        st.header("설정")
        
        # prediction 폴더 내의 여러 CSV 파일 선택
        prediction_dir = os.path.join("../", "prediction")
        csv_files = [f for f in os.listdir(prediction_dir) if f.endswith('.csv')]
        selected_csvs = st.multiselect("비교할 CSV 파일들 선택", csv_files)
        
        # 이미지 쌍 선택
        image_files = data_loader.get_image_list()
        image_pairs = data_loader.get_image_pairs(image_files)
        selected_pair = st.selectbox("이미지 쌍 선택", list(image_pairs.keys()))
        
        # 시각화 모드 선택
        view_mode = st.radio("시각화 모드 선택", 
                            ["마스크 중첩 모드", "나란히 비교 모드", "클래스별 비교 모드"])
    
    # 메인 영역에 결과 표시
    if selected_pair and selected_csvs:
        mask_generator = MaskGenerator()
        # CSV 파일별 고유 색상 생성
        csv_colors = {csv: get_distinct_colors(len(selected_csvs))[idx] 
                     for idx, csv in enumerate(selected_csvs)}
        
        if view_mode == "클래스별 비교 모드":
            st.subheader("클래스별 마스크 비교")
            # 클래스 선택도 사이드바로 이동
            with st.sidebar:
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
                            selected_class  # 클래스 이름 직접 전달,
                        )
                        masks_l.append(mask)
                    except Exception as e:
                        st.error(f"마스크 처리 중 오류 발생: {str(e)}")
                
                if masks_l:
                    result_l = overlay_multiple_masks_from_rle(
                        image=image_l,
                        data_loaders=data_loader,
                        csv_paths=[os.path.join(prediction_dir, csv) for csv in selected_csvs],
                        image_name=image_pairs[selected_pair]['L'].split('/')[-1],
                        image_shape=image_l.shape,
                        alpha=0.7,
                        beta=0.3,
                        csv_colors=[csv_colors[csv] for csv in selected_csvs],
                        selected_class=selected_class  # 선택된 클래스 전달
                    )
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
                    result_r = overlay_multiple_masks_from_rle(
                        image=image_r,
                        data_loaders=data_loader,
                        csv_paths=[os.path.join(prediction_dir, csv) for csv in selected_csvs],
                        image_name=image_pairs[selected_pair]['R'].split('/')[-1],
                        image_shape=image_r.shape,
                        alpha=0.7,
                        beta=0.3,
                        csv_colors=[csv_colors[csv] for csv in selected_csvs],
                        selected_class=selected_class  # 선택된 클래스 전달
                    )
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
            
        # 마스크 중첩 모드 부분
        elif view_mode == "마스크 중첩 모드":
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
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("Left Image")
                try:
                    result_l = overlay_multiple_masks_from_rle(
                        image=image_l,
                        data_loaders=data_loader,
                        csv_paths=[os.path.join(prediction_dir, csv) for csv in selected_csvs],
                        image_name=image_pairs[selected_pair]['L'].split('/')[-1],
                        image_shape=image_l.shape,
                        alpha=0.7,
                        beta=0.3,
                        csv_colors=[csv_colors[csv] for csv in selected_csvs]
                    )
                    st.image(result_l, use_container_width=True)
                except Exception as e:
                    st.error(f"마스크 처리 중 오류 발생: {str(e)}")

            with col2:
                st.write("📋 범례")
                for csv, color in csv_colors.items():
                    st.markdown(
                        f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                        f'<div style="width: 25px; height: 25px; background-color: rgb{color}; '
                        f'margin-right: 10px; border: 1px solid black;"></div>'
                        f'<span style="font-size: 16px;">{csv}</span></div>',
                        unsafe_allow_html=True
                    )

            # Right 이미지 처리
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("Right Image")
                try:
                    result_r = overlay_multiple_masks_from_rle(
                        image=image_r,
                        data_loaders=data_loader,
                        csv_paths=[os.path.join(prediction_dir, csv) for csv in selected_csvs],
                        image_name=image_pairs[selected_pair]['R'].split('/')[-1],
                        image_shape=image_r.shape,
                        alpha=0.7,
                        beta=0.3,
                        csv_colors=[csv_colors[csv] for csv in selected_csvs]
                    )
                    st.image(result_r, use_container_width=True)
                except Exception as e:
                    st.error(f"마스크 처리 중 오류 발생: {str(e)}")
            with col2:
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
            with cols[0]:
                st.write("Original")
                image_l = cv2.imread(os.path.join(data_loader.images_dir, 
                                                image_pairs[selected_pair]['L']))
                image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
                st.image(image_l, use_container_width=True)
            
            # 각 CSV 파일별 마스크 (Left)
            for idx, csv_file in enumerate(selected_csvs, 1):
                with cols[idx]:
                    st.write(f"Mask: {csv_file}")
                    try:
                        result = overlay_multiple_masks_from_rle(
                            image=image_l,
                            data_loaders=data_loader,
                            csv_paths=[os.path.join(prediction_dir, csv_file)],
                            image_name=image_pairs[selected_pair]['L'].split('/')[-1],
                            image_shape=image_l.shape,
                            alpha=0.7,
                            beta=0.3,
                            csv_colors=[csv_colors[csv_file]]
                        )
                        st.image(result, use_container_width=True)
                    except Exception as e:
                        st.error(f"마스크 처리 중 오류 발생: {str(e)}")
            
            # Right 이미지 세트 (Left와 동일한 로직)
            with cols[0]:
                st.write("Original")
                image_r = cv2.imread(os.path.join(data_loader.images_dir, 
                                                image_pairs[selected_pair]['R']))
                image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
                st.image(image_r, use_container_width=True)
            
            # 각 CSV 파일별 마스크 (Right)
            for idx, csv_file in enumerate(selected_csvs, 1):
                with cols[idx]:
                    st.write(f"Mask: {csv_file}")
                    try:
                        result = overlay_multiple_masks_from_rle(
                            image=image_r,
                            data_loaders=data_loader,
                            csv_paths=[os.path.join(prediction_dir, csv_file)],
                            image_name=image_pairs[selected_pair]['R'].split('/')[-1],
                            image_shape=image_r.shape,
                            alpha=0.7,
                            beta=0.3,
                            csv_colors=[csv_colors[csv_file]]
                        )
                        st.image(result, use_container_width=True)
                    except Exception as e:
                        st.error(f"마스크 처리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()