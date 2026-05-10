import joblib

# Load your saved scaler and model
scaler = joblib.load('model/scaler.pkl')
model = joblib.load('model/diabetes_model.pkl')

# Check what the scaler expects
print("=" * 50)
print("SCALER INFO")
print("=" * 50)
print(f"Number of features scaler expects: {scaler.n_features_in_}")

# Try to get feature names (if saved)
try:
    print(f"Feature names: {scaler.feature_names_in_}")
except AttributeError:
    print("Feature names not saved in scaler")

print()
print("=" * 50)
print("MODEL INFO")
print("=" * 50)
print(f"Number of features model expects: {model.n_features_in_}")

try:
    print(f"Feature names: {model.feature_names_in_}")
except AttributeError:
    print("Feature names not saved in model")