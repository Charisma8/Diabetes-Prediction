from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes_prediction_dataset.csv"
ARTIFACT_PATH = BASE_DIR / "model" / "model_bundle.pkl"

TARGET_COLUMN = "diabetes"
NUMERIC_FEATURES = ["age", "bmi", "hba1c_level", "blood_glucose_level"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES

FEATURE_GROUPS = {
    "lifestyle": ["smoking_history", "bmi"],
    "clinical": ["hba1c_level", "blood_glucose_level", "hypertension", "heart_disease"],
    "personal": ["age", "gender"],
}

SIMULATION_RULES = {
    "weight_reset": {"bmi": 23.5},
    "glucose_control": {"hba1c_level": 5.4, "blood_glucose_level": 100},
    "smoking_cessation": {"smoking_history": "never"},
}

FEATURE_LABELS = {
    "gender": "Gender",
    "age": "Age",
    "hypertension": "Hypertension",
    "heart_disease": "Heart disease",
    "smoking_history": "Smoking history",
    "bmi": "BMI",
    "hba1c_level": "HbA1c",
    "blood_glucose_level": "Blood glucose",
}

SMOKING_OPTIONS = ["never", "former", "not current", "current", "ever", "no_info"]
GENDER_OPTIONS = ["Female", "Male", "Other"]


@dataclass
class PredictionArtifacts:
    bundle: dict[str, Any]

    @property
    def pipeline(self):
        return self.bundle["pipeline"]

    @property
    def feature_names(self) -> list[str]:
        return self.bundle["feature_names"]

    @property
    def transformed_feature_names(self) -> list[str]:
        return self.bundle["transformed_feature_names"]


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"HbA1c_level": "hba1c_level"})
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    df["smoking_history"] = (
        df["smoking_history"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"no info": "no_info"})
    )
    df["gender"] = df["gender"].astype(str).str.strip().str.title()
    df["hypertension"] = df["hypertension"].astype(int)
    df["heart_disease"] = df["heart_disease"].astype(int)
    df = df.dropna(subset=[TARGET_COLUMN]).drop_duplicates()
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _build_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_candidate_pipelines() -> dict[str, Pipeline]:
    common_preprocessor = build_preprocessor()
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", common_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=220,
                        max_depth=10,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def compute_metrics(y_true: pd.Series, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "f1": round(float(f1_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions)), 4),
        "recall": round(float(recall_score(y_true, predictions)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
    }


def train_and_save_model() -> dict[str, Any]:
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    evaluations: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, pipeline in build_candidate_pipelines().items():
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        evaluations[name] = compute_metrics(y_test, probabilities, predictions)
        fitted_models[name] = pipeline

    deployment_candidates = {
        name: metrics
        for name, metrics in evaluations.items()
        if name != "Logistic Regression"
    }
    best_model_name = max(
        deployment_candidates,
        key=lambda candidate: (
            deployment_candidates[candidate]["roc_auc"],
            deployment_candidates[candidate]["pr_auc"],
        ),
    )
    best_pipeline = fitted_models[best_model_name]

    preprocessor = best_pipeline.named_steps["preprocessor"]
    transformed_feature_names = list(preprocessor.get_feature_names_out())
    defaults = {
        "gender": X["gender"].mode().iat[0],
        "age": round(float(X["age"].median()), 1),
        "hypertension": int(X["hypertension"].mode().iat[0]),
        "heart_disease": int(X["heart_disease"].mode().iat[0]),
        "smoking_history": X["smoking_history"].mode().iat[0],
        "bmi": round(float(X["bmi"].median()), 2),
        "hba1c_level": round(float(X["hba1c_level"].median()), 1),
        "blood_glucose_level": round(float(X["blood_glucose_level"].median()), 0),
    }

    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "feature_names": FEATURE_COLUMNS,
        "transformed_feature_names": transformed_feature_names,
        "metrics": evaluations,
        "defaults": defaults,
        "class_rate": round(float(y.mean()), 4),
        "dataset_size": int(df.shape[0]),
        "background_sample": X_train.sample(min(len(X_train), 256), random_state=42),
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH)
    return bundle


def load_artifacts() -> PredictionArtifacts:
    if not ARTIFACT_PATH.exists():
        train_and_save_model()
    bundle = joblib.load(ARTIFACT_PATH)
    return PredictionArtifacts(bundle=bundle)


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "gender": str(payload.get("gender", "")).strip().title() or "Female",
        "age": float(payload.get("age", 0)),
        "hypertension": int(payload.get("hypertension", 0)),
        "heart_disease": int(payload.get("heart_disease", 0)),
        "smoking_history": str(payload.get("smoking_history", "")).strip().lower().replace("no info", "no_info"),
        "bmi": float(payload.get("bmi", 0)),
        "hba1c_level": float(payload.get("hba1c_level", 0)),
        "blood_glucose_level": float(payload.get("blood_glucose_level", 0)),
    }
    if normalized["smoking_history"] not in SMOKING_OPTIONS:
        normalized["smoking_history"] = "no_info"
    if normalized["gender"] not in GENDER_OPTIONS:
        normalized["gender"] = "Other"
    return normalized


def prepare_input_frame(payload: dict[str, Any]) -> pd.DataFrame:
    normalized = normalize_payload(payload)
    return pd.DataFrame([normalized], columns=FEATURE_COLUMNS)


def risk_level_from_score(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Elevated"
    if score < 75:
        return "High"
    return "Very High"


def transform_feature_name(transformed_name: str) -> str:
    if transformed_name.startswith("numeric__"):
        return transformed_name.split("__", 1)[1]
    if transformed_name.startswith("binary__"):
        return transformed_name.split("__", 1)[1]
    if transformed_name.startswith("categorical__smoking_history_"):
        return "smoking_history"
    if transformed_name.startswith("categorical__gender_"):
        return "gender"
    return transformed_name


def build_feature_groups(impacts: dict[str, float]) -> dict[str, float]:
    totals = {}
    impact_sum = sum(abs(value) for value in impacts.values()) or 1.0
    for group_name, features in FEATURE_GROUPS.items():
        group_total = sum(abs(impacts.get(feature, 0.0)) for feature in features)
        totals[group_name] = round(group_total / impact_sum * 100, 1)
    return totals


def build_recommendations(row: dict[str, Any], top_driver_names: list[str]) -> list[str]:
    recommendations: list[str] = []
    if row["bmi"] >= 30:
        recommendations.append("Prioritize weight management because your BMI is currently in a higher-risk range.")
    if row["hba1c_level"] >= 5.7:
        recommendations.append("Track long-term blood sugar closely; your HbA1c is above the healthy baseline.")
    if row["blood_glucose_level"] >= 140:
        recommendations.append("A glucose-focused intervention could have the biggest short-term impact on your risk.")
    if row["smoking_history"] in {"current", "ever"}:
        recommendations.append("Smoking history is contributing to risk; cessation support would materially improve the outlook.")
    if row["hypertension"] == 1:
        recommendations.append("Blood pressure management matters here because hypertension is adding to overall risk.")
    if not recommendations:
        recommendations.append("Your profile is comparatively stable, so maintaining exercise, sleep, and nutrition habits is the main priority.")

    driver_summary = ", ".join(FEATURE_LABELS.get(feature, feature) for feature in top_driver_names[:3])
    recommendations.append(f"Most influential factors right now: {driver_summary}.")
    return recommendations[:4]


def _apply_scenario(base_row: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    updated = dict(base_row)
    for key, target_value in changes.items():
        if key in {"bmi", "hba1c_level", "blood_glucose_level"}:
            updated[key] = min(updated[key], target_value)
        else:
            updated[key] = target_value
    return updated


def build_simulation_scenarios(base_row: dict[str, Any], probability_fn) -> list[dict[str, Any]]:
    scenarios = []
    baseline_risk = round(probability_fn(base_row) * 100, 1)

    scenario_catalog = [
        ("Weight reset", "Bringing BMI closer to a healthy target.", SIMULATION_RULES["weight_reset"]),
        ("Glucose control", "Improving glucose and HbA1c control.", SIMULATION_RULES["glucose_control"]),
        ("Smoking cessation", "Moving to a non-smoking history state.", SIMULATION_RULES["smoking_cessation"]),
    ]

    for title, summary, changes in scenario_catalog:
        updated = _apply_scenario(base_row, changes)
        if updated == base_row:
            continue

        new_risk = round(probability_fn(updated) * 100, 1)
        scenarios.append(
            {
                "title": title,
                "summary": summary,
                "new_risk_percent": new_risk,
                "risk_drop": round(max(baseline_risk - new_risk, 0), 1),
                "changes": [
                    {
                        "feature": FEATURE_LABELS[key],
                        "original": base_row[key],
                        "new": updated[key],
                    }
                    for key in changes
                    if updated[key] != base_row[key]
                ],
            }
        )

    scenarios.sort(key=lambda item: item["risk_drop"], reverse=True)
    return scenarios
