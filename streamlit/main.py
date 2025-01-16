import streamlit as st
st.set_page_config(
    page_title="세그멘테이션 데이터 뷰어",
    page_icon="🔍",
    layout="wide"
)
def main():
    st.title("세그멘테이션 데이터 뷰어 🔍")
    
    st.markdown("""
    ### 주요 기능
    - 데이터 뷰어: 이미지와 세그멘테이션 마스크 시각화
    - JSON 어노테이션 데이터 확인
    
    왼쪽 사이드바에서 원하는 페이지로 이동하실 수 있습니다.
    """)

if __name__ == "__main__":
    main() 