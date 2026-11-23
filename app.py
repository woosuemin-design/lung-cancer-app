import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

st.set_page_config(page_title="폐암 위험 및 합병증 예측 플랫폼", layout="centered")


def get_base_dir():
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


def find_csv_file(base_dir):
    candidates = [
        os.path.join(base_dir, "orange_input_patients.csv"),
        os.path.join(base_dir, "orange_input_patients.csv.csv"),
        "orange_input_patients.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def train_and_save_models(csv_path, model_path):
    try:
        raw_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw_df = pd.read_csv(csv_path, encoding="cp949")

    first_cell = str(raw_df.iloc[0, 0]).strip().lower()
    if first_cell in ["continuous", "discrete", "string", "class"]:
        df = raw_df.iloc[1:].copy().reset_index(drop=True)
    else:
        df = raw_df.copy().reset_index(drop=True)

    df["시퀀싱_보고_나이"] = pd.to_numeric(
        df["시퀀싱_보고_나이"], errors="coerce"
    )
    df["폐암_위험확률"] = pd.to_numeric(
        df["폐암_위험확률"], errors="coerce"
    )

    age_median = df["시퀀싱_보고_나이"].median()
    df["시퀀싱_보고_나이"] = df["시퀀싱_보고_나이"].fillna(
        66.0 if pd.isna(age_median) else age_median
    )

    risk_median = df["폐암_위험확률"].median()
    df["폐암_위험확률"] = df["폐암_위험확률"].fillna(
        86.0 if pd.isna(risk_median) else risk_median
    )

    num_cols = ["시퀀싱_보고_나이"]
    cat_cols = ["성별", "흡연_여부"]
    text_col = "증상"

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str)
    df[text_col] = df[text_col].fillna("").astype(str)

    X = df[num_cols + cat_cols + [text_col]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("text", TfidfVectorizer(), text_col),
        ]
    )

    clf_cancer = Pipeline([
        ("prep", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            ),
        ),
    ])
    clf_cancer.fit(X, df["폐암_발생_여부"].fillna("Unknown").astype(str))

    reg_risk = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])
    reg_risk.fit(X, df["폐암_위험확률"])

    clf_comp = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    clf_comp.fit(
        X, df["예상_합병증"].fillna("특이 합병증 징후 낮음").astype(str)
    )

    clf_meta = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    clf_meta.fit(X, df["전이_부위"].fillna("없음").astype(str))

    bundle = {
        "cancer": clf_cancer,
        "risk": reg_risk,
        "complication": clf_comp,
        "metastasis": clf_meta,
    }
    try:
        joblib.dump(bundle, model_path)
    except Exception:
        pass
    return bundle


@st.cache_resource
def load_or_build_models():
    base_dir = get_base_dir()
    model_path = os.path.join(base_dir, "lung_cancer_models.pkl")

    if os.path.exists(model_path):
        return joblib.load(model_path)

    csv_path = find_csv_file(base_dir)
    if csv_path is None:
        raise FileNotFoundError(
            "lung_cancer_models.pkl과 orange_input_patients.csv 파일이 모두 없습니다. GitHub 저장소에 orange_input_patients.csv를 올려주세요."
        )

    return train_and_save_models(csv_path, model_path)


st.title("비소세포폐암 위험도 및 합병증 예측 플랫폼")
st.caption(
    "※ 학습 데이터 기반 프로토타입 결과이며, 실제 의학적 진단을 대체하지 않습니다."
)

try:
    with st.spinner("모델을 준비하는 중입니다..."):
        models = load_or_build_models()
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
