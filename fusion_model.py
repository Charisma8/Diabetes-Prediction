import pandas as pd
import numpy as np
import joblib
import warnings
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

LIFESTYLE_FEATURES = ['BMI', 'Age', 'SkinThickness']
CLINICAL_FEATURES = ['Glucose', 'BloodPressure', 'Insulin']
GENETIC_FEATURES = ['DiabetesPedigreeFunction', 'Pregnancies']
ALL_FEATURES = LIFESTYLE_FEATURES + CLINICAL_FEATURES + GENETIC_FEATURES
TARGET = 'Outcome'

df = pd.read_csv('data/diabetes.csv')

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)

X = df[ALL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Before SMOTE:", Counter(y_train))

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE:", Counter(y_train_res))

scaler = StandardScaler()
X_train_res_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_res),
    columns=ALL_FEATURES
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=ALL_FEATURES
)

lifestyle_model = RandomForestClassifier(n_estimators=100, random_state=42)
clinical_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
genetic_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

lifestyle_model.fit(X_train_res_scaled[LIFESTYLE_FEATURES], y_train_res)
clinical_model.fit(X_train_res_scaled[CLINICAL_FEATURES], y_train_res)
genetic_model.fit(X_train_res_scaled[GENETIC_FEATURES], y_train_res)

lifestyle_train_probs = lifestyle_model.predict_proba(X_train_res_scaled[LIFESTYLE_FEATURES])[:, 1]
clinical_train_probs = clinical_model.predict_proba(X_train_res_scaled[CLINICAL_FEATURES])[:, 1]
genetic_train_probs = genetic_model.predict_proba(X_train_res_scaled[GENETIC_FEATURES])[:, 1]

lifestyle_test_probs = lifestyle_model.predict_proba(X_test_scaled[LIFESTYLE_FEATURES])[:, 1]
clinical_test_probs = clinical_model.predict_proba(X_test_scaled[CLINICAL_FEATURES])[:, 1]
genetic_test_probs = genetic_model.predict_proba(X_test_scaled[GENETIC_FEATURES])[:, 1]

print("Lifestyle AUC:", round(roc_auc_score(y_test, lifestyle_test_probs), 3))
print("Clinical AUC:", round(roc_auc_score(y_test, clinical_test_probs), 3))
print("Genetic AUC:", round(roc_auc_score(y_test, genetic_test_probs), 3))

fusion_train = np.column_stack([
    lifestyle_train_probs,
    clinical_train_probs,
    genetic_train_probs
])

fusion_test = np.column_stack([
    lifestyle_test_probs,
    clinical_test_probs,
    genetic_test_probs
])

meta_model = LogisticRegression(random_state=42)
meta_model.fit(fusion_train, y_train_res)

final_probs = meta_model.predict_proba(fusion_test)[:, 1]
final_preds = meta_model.predict(fusion_test)

print("\nFusion Results")
print(classification_report(y_test, final_preds, target_names=['No Diabetes', 'Diabetes']))
print("Fusion AUC:", round(roc_auc_score(y_test, final_probs), 3))

joblib.dump(lifestyle_model, 'model/lifestyle_model.pkl')
joblib.dump(clinical_model, 'model/clinical_model.pkl')
joblib.dump(genetic_model, 'model/genetic_model.pkl')
joblib.dump(meta_model, 'model/meta_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')