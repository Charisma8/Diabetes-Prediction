from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import shap
import traceback

app = Flask(__name__)
CORS(app)

model    = joblib.load('model/diabetes_model.pkl')
scaler   = joblib.load('model/scaler.pkl')

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

explainer = shap.TreeExplainer(model)


def to_python(val):
    if isinstance(val, (np.float32, np.float64, np.floating)):
        return float(val)
    if isinstance(val, (np.int32, np.int64, np.integer)):
        return int(val)
    return val


def input_to_scaled_df(data):
    raw_df    = pd.DataFrame(
        [[float(data[f]) for f in FEATURE_NAMES]],
        columns=FEATURE_NAMES
    )
    scaled    = scaler.transform(raw_df)
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_NAMES)
    return scaled_df


def get_risk_level(risk_percent):
    if risk_percent < 30:
        return "Low"
    elif risk_percent < 60:
        return "Moderate"
    else:
        return "High"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        for f in FEATURE_NAMES:
            try:
                float(data[f])
            except (ValueError, TypeError):
                return jsonify({"error": f"{f} must be a number"}), 400

        scaled_df = input_to_scaled_df(data)

        prob         = to_python(model.predict_proba(scaled_df)[0][1])
        risk_percent = round(prob * 100, 1)
        risk_level   = get_risk_level(risk_percent)

        shap_matrix = explainer.shap_values(scaled_df)
        shap_row    = shap_matrix[0] if hasattr(shap_matrix, 'ndim') and shap_matrix.ndim == 2 else shap_matrix[0]

        shap_list = [
            {
                "feature": FEATURE_NAMES[i],
                "value":   round(to_python(shap_row[i]), 3)
            }
            for i in range(len(FEATURE_NAMES))
        ]
        shap_list.sort(key=lambda x: abs(x["value"]), reverse=True)

        top_two     = shap_list[:2]
        direction   = "increasing" if risk_percent >= 50 else "reducing"
        explanation = (
            f"Your {top_two[0]['feature']} and {top_two[1]['feature']} "
            f"are the two biggest factors {direction} your diabetes risk."
        )

        return jsonify({
            "risk_percent": risk_percent,
            "risk_level":   risk_level,
            "shap_values":  shap_list,
            "explanation":  explanation
        })

    except Exception:
        return jsonify({
            "error":   "Internal server error",
            "details": traceback.format_exc()
        }), 500


@app.route('/whatif', methods=['POST'])
def whatif():
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        scaled_df    = input_to_scaled_df(data)
        prob         = to_python(model.predict_proba(scaled_df)[0][1])
        risk_percent = round(prob * 100, 1)

        return jsonify({
            "risk_percent": risk_percent,
            "risk_level":   get_risk_level(risk_percent)
        })

    except Exception:
        return jsonify({
            "error":   "Internal server error",
            "details": traceback.format_exc()
        }), 500


@app.route('/counterfactual', methods=['POST'])
def counterfactual():
    try:
        import dice_ml

        data = request.json

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        scaled_df    = input_to_scaled_df(data)
        prob         = to_python(model.predict_proba(scaled_df)[0][1])
        risk_percent = round(prob * 100, 1)

        train_X  = pd.read_csv('model/train_data.csv')
        train_y  = pd.read_csv('model/train_labels.csv').squeeze()
        train_df = train_X.copy()
        train_df['Outcome'] = train_y.values

        dice_data  = dice_ml.Data(
            dataframe           = train_df,
            continuous_features = FEATURE_NAMES,
            outcome_name        = 'Outcome'
        )
        dice_model = dice_ml.Model(model=model, backend='sklearn')
        dice_exp   = dice_ml.Dice(dice_data, dice_model, method='random')

        desired = 0 if prob >= 0.5 else 1
        cf = dice_exp.generate_counterfactuals(
            scaled_df,
            total_CFs        = 3,
            desired_class    = desired,
            features_to_vary = ['BMI', 'Glucose', 'Insulin', 'BloodPressure']
        )

        cf_df      = cf.cf_examples_list[0].final_cfs_df
        cf_records = []

        for _, row in cf_df.iterrows():
            changes  = []
            cf_input = pd.DataFrame([row[FEATURE_NAMES]], columns=FEATURE_NAMES)
            cf_prob  = to_python(model.predict_proba(cf_input)[0][1])
            cf_pct   = round(cf_prob * 100, 1)

            for feat in ['BMI', 'Glucose', 'Insulin', 'BloodPressure']:
                original = round(to_python(scaled_df[feat].values[0]), 2)
                new_val  = round(to_python(row[feat]), 2)
                delta    = round(new_val - original, 3)
                if abs(delta) > 0.01:
                    changes.append({
                        "feature":  feat,
                        "original": original,
                        "new":      new_val,
                        "delta":    delta
                    })

            cf_records.append({
                "new_risk_percent": cf_pct,
                "risk_drop":        round(risk_percent - cf_pct, 1),
                "changes":          changes
            })

        return jsonify({
            "current_risk":    risk_percent,
            "counterfactuals": cf_records
        })

    except Exception:
        return jsonify({
            "error":   "Internal server error",
            "details": traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)