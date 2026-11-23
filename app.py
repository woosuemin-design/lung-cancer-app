import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="폐암 위험 및 합병증 예측 플랫폼", layout="centered")


def get_model_path():
    if "__file__" in globals():
        p = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "lung_cancer_models.pkl"
        )
        if os.path.exists(p):
            return p
    return "lung_cancer_models.pkl"


@st.cache_resource
def load_models():
    path = get_model_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            "lung_cancer_models.pkl 파일이 없습니다. 먼저 'python train_model.py'를 실행해 주세요."
        )
    return joblib.load(path)


st.title("비소세포폐암 위험도 및 합병증 예측 플랫폼")
st.caption(
    "※ 학습 데이터 기반 프로토타입 결과이며, 실제 의학적 진단을 대체하지 않습니다."
)

try:
    models = load_models()
except Exception as e:
    st.error(str(e))
    st.stop()

age = st.number_input("나이(시퀀싱_보고_나이)", min_value=18, max_value=100, value=66)
gender = st.selectbox("성별", ["Female", "Male"])
smoking = st.selectbox("흡연_여부", ["True", "False"])

symptom_options = [
    "지속적인 기침",
    "호흡곤란",
    "만성 가래 및 기침",
    "객혈",
    "원인 불명의 체중 감소",
    "식욕 부진",
    "극심한 암성 피로",
]
selected_symptoms = st.multiselect("증상 선택", symptom_options)
custom_symptom = st.text_input("기타 증상(선택)")

symptoms_all = selected_symptoms + (
    [custom_symptom.strip()] if custom_symptom.strip() else []
)
symptom_text = ", ".join(symptoms_all)

if st.button("예측 실행"):
    input_df = pd.DataFrame([
        {
            "시퀀싱_보고_나이": age,
            "성별": gender,
            "흡연_여부": smoking,
            "증상": symptom_text,
        }
    ])

    cancer_pred = models["cancer"].predict(input_df)[0]
    risk_pred = float(models["risk"].predict(input_df)[0])
    comp_pred = models["complication"].predict(input_df)[0]
    meta_pred = models["metastasis"].predict(input_df)[0]

    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("폐암_발생_여부", str(cancer_pred))
    col2.metric("폐암_위험확률", f"{risk_pred:.1f}%")

    st.write("**예상_합병증:**", comp_pred)
    st.write("**전이_부위:**", meta_pred)
