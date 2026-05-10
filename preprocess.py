import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv('data/diabetes.csv')

print("=" * 50)
print("STEP 1: RAW DATA")
print("=" * 50)
print(f"Shape: {df.shape}")          # rows x columns
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")


# ─────────────────────────────────────────
# 2. FIX IMPOSSIBLE ZERO VALUES
# ─────────────────────────────────────────
# In medical data, 0 is impossible for these columns
# A person can't have 0 glucose, 0 BMI, or 0 blood pressure
# These are actually missing values disguised as 0

cols_with_invalid_zeros = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

print("\n" + "=" * 50)
print("STEP 2: FIXING INVALID ZEROS")
print("=" * 50)

for col in cols_with_invalid_zeros:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} zeros found → replacing with NaN")

# Replace 0 with NaN so we can handle them properly
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

print(f"\nMissing values after replacement:\n{df.isnull().sum()}")


# ─────────────────────────────────────────
# 3. FILL MISSING VALUES
# ─────────────────────────────────────────
# Strategy: fill with MEDIAN (not mean) because median
# is not affected by extreme outliers in medical data

print("\n" + "=" * 50)
print("STEP 3: FILLING MISSING VALUES WITH MEDIAN")
print("=" * 50)

for col in cols_with_invalid_zeros:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"{col}: filled with median = {median_val:.2f}")

print(f"\nMissing values after filling:\n{df.isnull().sum()}")
print("\n✓ No more missing values!")


# ─────────────────────────────────────────
# 4. REMOVE OUTLIERS
# ─────────────────────────────────────────
# Using IQR method — removes extreme values that are
# statistically impossible (e.g., glucose of 900)

print("\n" + "=" * 50)
print("STEP 4: REMOVING OUTLIERS")
print("=" * 50)

df_before = df.shape[0]

for col in df.columns[:-1]:  # skip 'Outcome' column
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR   # using 3x IQR (less aggressive)
    upper = Q3 + 3 * IQR
    
    outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    if outliers > 0:
        print(f"{col}: removing {outliers} outliers (range: {lower:.1f} – {upper:.1f})")
    
    df = df[(df[col] >= lower) & (df[col] <= upper)]

df_after = df.shape[0]
print(f"\nRows before: {df_before}")
print(f"Rows after:  {df_after}")
print(f"Removed:     {df_before - df_after} rows")


# ─────────────────────────────────────────
# 5. CHECK CLASS BALANCE
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: CLASS BALANCE CHECK")
print("=" * 50)

class_counts = df['Outcome'].value_counts()
print(f"Non-diabetic (0): {class_counts[0]}")
print(f"Diabetic     (1): {class_counts[1]}")
print(f"Ratio: {class_counts[0]/class_counts[1]:.2f}:1")


# ─────────────────────────────────────────
# 6. SEPARATE FEATURES AND TARGET
# ─────────────────────────────────────────
X = df.drop('Outcome', axis=1)
y = df['Outcome']

feature_names = X.columns.tolist()
print(f"\nFeatures: {feature_names}")


# ─────────────────────────────────────────
# 7. APPLY SMOTE (fix class imbalance)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6: SMOTE — BALANCING CLASSES")
print("=" * 50)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

print(f"Before SMOTE: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"After  SMOTE: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
print("✓ Classes are now balanced!")


# ─────────────────────────────────────────
# 8. SCALE FEATURES
# ─────────────────────────────────────────
# StandardScaler → makes all features have mean=0, std=1
# This is REQUIRED for many ML models to work correctly

print("\n" + "=" * 50)
print("STEP 7: FEATURE SCALING")
print("=" * 50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

print("Before scaling (first row):")
print([round(v, 2) for v in X_resampled.iloc[0].values])

print("\nAfter scaling (first row):")
print([round(v, 2) for v in X_scaled[0]])
print("\n✓ All features now on same scale!")


# ─────────────────────────────────────────
# 9. SAVE CLEANED DATA + SCALER
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 8: SAVING FILES")
print("=" * 50)

os.makedirs('model', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Save cleaned (pre-SMOTE) dataframe for reference
df.to_csv('data/diabetes_cleaned.csv', index=False)

# Save scaler (needed later in Flask API)
joblib.dump(scaler, 'model/scaler.pkl')

# Save processed arrays for training
np.save('data/X_scaled.npy', X_scaled)
np.save('data/y.npy', y_resampled.values)

print("✓ data/diabetes_cleaned.csv")
print("✓ model/scaler.pkl")
print("✓ data/X_scaled.npy")
print("✓ data/y.npy")


# ─────────────────────────────────────────
# 10. QUICK VISUAL CHECK (optional)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 9: GENERATING CHARTS (close window to continue)")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Diabetes Dataset — Preprocessing Summary', fontsize=14)

# Chart 1: Class balance after SMOTE
axes[0, 0].bar(['Non-diabetic', 'Diabetic'],
               [np.sum(y_resampled == 0), np.sum(y_resampled == 1)],
               color=['#4CAF50', '#F44336'])
axes[0, 0].set_title('Class Balance (after SMOTE)')
axes[0, 0].set_ylabel('Count')

# Chart 2: Glucose distribution
axes[0, 1].hist(df['Glucose'], bins=30, color='#2196F3', edgecolor='white')
axes[0, 1].set_title('Glucose Distribution (cleaned)')
axes[0, 1].set_xlabel('Glucose Level')

# Chart 3: BMI distribution
axes[1, 0].hist(df['BMI'], bins=30, color='#FF9800', edgecolor='white')
axes[1, 0].set_title('BMI Distribution (cleaned)')
axes[1, 0].set_xlabel('BMI')

# Chart 4: Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=axes[1, 1], annot_kws={'size': 7})
axes[1, 1].set_title('Feature Correlation')

plt.tight_layout()
plt.savefig('data/preprocessing_charts.png', dpi=150)
plt.show()
print("✓ Chart saved to data/preprocessing_charts.png")

print("\n" + "=" * 50)
print("ALL DONE! Your data is clean and ready for training.")
print("=" * 50)