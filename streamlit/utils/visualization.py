import streamlit as st
import matplotlib.pyplot as plt

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
    
    @staticmethod
    def display_cloud_point(mask):
        col1 = st.columns(1)

        with col1:
            st.subheader("포인트 클라우드")
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.imshow(mask, cmap='bone')
            ax.axis('off')
            st.pyplot(fig)

        