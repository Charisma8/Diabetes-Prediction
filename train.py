import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# ── 1. Load data ──────────────────────────────────────────────────
df = pd.read_csv('data/diabetes.csv')

print("Columns found:", df.columns.tolist())
print("Shape:", df.shape)

# ── 2. Fix invalid zero values ────────────────────────────────────
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)

# ── 3. Split features and target ──────────────────────────────────
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print("Feature names:", X.columns.tolist())
print("Class balance:", y.value_counts().to_dict())

# ── 4. Train/test split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Apply SMOTE on training data only ──────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE - class balance:", dict(zip(*np.unique(y_train_res, return_counts=True))))

# ── 6. Scale features ─────────────────────────────────────────────
# fit_transform on train, only transform on test
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_res),
    columns=X.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns
)

# ── 7. Train XGBoost model ────────────────────────────────────────
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train_res)

# ── 8. Evaluate ───────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n── Model Evaluation ──────────────────────────────")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
print("AUC-ROC Score:", round(roc_auc_score(y_test, y_prob), 4))

# ── 9. Save model, scaler, and training data ──────────────────────
os.makedirs('model', exist_ok=True)

joblib.dump(model,  'model/diabetes_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Save training data for DiCE counterfactuals (used in /counterfactual endpoint)
X_train_scaled.to_csv('model/train_data.csv', index=False)
y_train_res.to_csv('model/train_labels.csv', index=False)

print("\nFiles saved:")
print("  model/diabetes_model.pkl")
print("  model/scaler.pkl")
print("  model/train_data.csv")
print("  model/train_labels.csv")
print("\nScaler trained on features:", scaler.feature_names_in_.tolist())
print("\nDone! Now restart app.py")