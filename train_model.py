import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def find_csv_file():
    search_dirs = [os.getcwd()]
    if "__file__" in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in search_dirs:
            search_dirs.insert(0, script_dir)

    names = [
        "orange_input_patients.csv",
        "orange_input_patients.csv.csv",
        "patient_info.csv",
        "patient_info.csv.csv",
    ]

    for d in search_dirs:
        for name in names:
            full_path = os.path.join(d, name)
            if os.path.exists(full_path):
                return full_path, d

    raise FileNotFoundError(
        "CSV 파일을 찾지 못했습니다. 'orange_input_patients.csv'와 'train_model.py'를 같은 폴더에 넣어주세요."
    )


def main():
    csv_path, target_dir = find_csv_file()
    print(f"데이터 불러오는 중: {csv_path}")

    try:
        raw_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw_df = pd.read_csv(csv_path, encoding="cp949")

    # Orange 메타데이터 행(continuous, discrete, string, class) 제거
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

    print("1/4 폐암 발생 여부 모델 학습 중...")
    clf_cancer = Pipeline([
        ("prep", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=150, class_weight="balanced", random_state=42
            ),
        ),
    ])
    clf_cancer.fit(X, df["폐암_발생_여부"].fillna("Unknown").astype(str))

    print("2/4 폐암 위험확률 모델 학습 중...")
    reg_risk = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=150, random_state=42)),
    ])
    reg_risk.fit(X, df["폐암_위험확률"])

    print("3/4 예상 합병증 모델 학습 중...")
    clf_comp = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=150, random_state=42)),
    ])
    clf_comp.fit(
        X, df["예상_합병증"].fillna("특이 합병증 징후 낮음").astype(str)
    )

    print("4/4 예상 전이 부위 모델 학습 중...")
    clf_meta = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=150, random_state=42)),
    ])
    clf_meta.fit(X, df["전이_부위"].fillna("없음").astype(str))

    model_path = os.path.join(target_dir, "lung_cancer_models.pkl")
    bundle = {
        "cancer": clf_cancer,
        "risk": reg_risk,
        "complication": clf_comp,
        "metastasis": clf_meta,
    }
    joblib.dump(bundle, model_path)
    print(f"완료! 저장된 파일: {model_path}")


if __name__ == "__main__":
    main()
