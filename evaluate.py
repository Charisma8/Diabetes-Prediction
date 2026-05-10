import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------- 1. Load data ----------
df = pd.read_csv("data/diabetes_data.csv", sep=";")

print("Original columns:")
print(df.columns.tolist())

# Clean column names
df.columns = df.columns.str.strip()

# Clean string values in all object columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

print("\nCleaned columns:")
print(df.columns.tolist())

print("\nSample rows:")
print(df.head())

# ---------- 2. Show unique values before mapping ----------
print("\nUnique values before mapping:")
for col in df.columns:
    if df[col].dtype == "object":
        print(f"{col}: {df[col].unique()}")

# ---------- 3. Encode target and features ----------
df["class"] = df["class"].replace({
    "Positive": 1, "Negative": 0,
    "positive": 1, "negative": 0
})

df["gender"] = df["gender"].replace({
    "Male": 1, "Female": 0,
    "male": 1, "female": 0
})

yes_no_cols = [
    "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia",
    "genital_thrush", "visual_blurring", "itching", "irritability",
    "delayed_healing", "partial_paresis", "muscle_stiffness", "alopecia", "obesity"
]

for col in yes_no_cols:
    df[col] = df[col].replace({
        "Yes": 1, "No": 0,
        "yes": 1, "no": 0
    })

# Convert age to numeric
df["age"] = pd.to_numeric(df["age"], errors="coerce")

print("\nMissing values after encoding:")
print(df.isnull().sum())

print("\nShape before dropna:", df.shape)

# Drop rows with missing values only if needed
df = df.dropna()

print("Shape after dropna:", df.shape)

if df.shape[0] == 0:
    raise ValueError("DataFrame is empty after cleaning. Check unique values printed above.")

print("\nClass counts after cleaning:")
print(df["class"].value_counts())

# ---------- 4. Split features and target ----------
X = df.drop("class", axis=1)
y = df["class"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# ---------- 5. Apply SMOTE ----------
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("\nAfter SMOTE:")
print("X_res shape:", X_res.shape)
print("y_res counts:")
print(pd.Series(y_res).value_counts())

# ---------- 6. Scale ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# ---------- 7. Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ---------- 8. Models ----------
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
}

results = {}

print("\n" + "=" * 60)
print("MODEL EVALUATION REPORT")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "ap": ap
    }

    print(f"\n{name}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")
    print(f"Avg Prec  : {ap:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# ---------- 9. Cross-validation ----------
print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y_res, cv=cv, scoring="roc_auc")
    print(f"\n{name}")
    print(f"AUC scores : {[round(s, 3) for s in cv_scores]}")
    print(f"Mean AUC   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ---------- 10. Plots ----------
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Diabetes Model Evaluation Dashboard", fontsize=16, fontweight="bold")
colors = ["#4C72B0", "#55A868", "#C44E52"]

ax = axes[0, 0]
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
for (name, res), color in zip(results.items(), colors):
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, res["y_prob"])
    ax.plot(rec_curve, prec_curve, color=color, lw=2, label=f"{name} (AP={res['ap']:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 2]
metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
x = np.arange(len(metric_names))
width = 0.25

for i, ((name, res), color) in enumerate(zip(results.items(), colors)):
    vals = [res["accuracy"], res["precision"], res["recall"], res["f1"], res["auc"]]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + width)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.1)
ax.set_title("Metric Comparison")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

for idx, ((name, res), color) in enumerate(zip(results.items(), colors)):
    ax = axes[1, idx]
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(
        cm, annot=True, fmt="d", ax=ax, cmap="Blues", cbar=False,
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"]
    )
    ax.set_title(f"Confusion Matrix\n{name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("evaluation_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved: evaluation_dashboard.png")

# ---------- 11. Best model ----------
best_name = max(results, key=lambda n: results[n]["auc"])
best = results[best_name]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)
print(f"Best model : {best_name}")
print(f"AUC-ROC    : {best['auc']:.4f}")
print(f"F1 Score   : {best['f1']:.4f}")
print(f"Recall     : {best['recall']:.4f}")
print(f"Accuracy   : {best['accuracy']:.4f}")

joblib.dump(best["model"], "model/diabetes_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("\nSaved: model/diabetes_model.pkl")
print("Saved: model/scaler.pkl")

# Feature Importance (for XGBoost)
xgb_model = results['XGBoost']['model']
importances = xgb_model.feature_importances_
features = X.columns.tolist()

plt.figure(figsize=(8, 5))
sorted_idx = np.argsort(importances)
plt.barh([features[i] for i in sorted_idx],
         [importances[i] for i in sorted_idx],
         color='#4C72B0')
plt.title('Feature Importance — XGBoost')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()