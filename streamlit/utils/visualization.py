import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from utils.mask_generator import MaskGenerator
# 색상 리스트
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# PALETTE를 0-1 범위의 RGB로 정규화
colors = np.array(PALETTE) / 255.0

# 커스텀 컬러맵 생성
custom_cmap = ListedColormap(colors)

class Visualizer:
    @staticmethod
    def display_image_and_mask(image, mask):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("원본 이미지")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("세그멘테이션 마스크")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(mask, cmap='bone')
            ax.axis('off')
            st.pyplot(fig)