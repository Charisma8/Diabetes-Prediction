from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import joblib, numpy as np, shap

app = Flask(__name__)
CORS(app)
api = Api(app,
    title='Diabetes Risk Prediction API',
    version='1.0',
    description='Predicts diabetes risk using lifestyle, clinical, and genetic data'
)

model_ml = joblib.load('model/diabetes_model.pkl')
scaler = joblib.load('model/scaler.pkl')
explainer = shap.TreeExplainer(model_ml)

FEATURE_NAMES = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                 'Insulin','BMI','DiabetesPedigreeFunction','Age']

# Define input/output models for the docs
input_model = api.model('PatientInput', {
    'Pregnancies':             fields.Float(required=True, description='Number of pregnancies', example=2),
    'Glucose':                 fields.Float(required=True, description='Plasma glucose concentration (mg/dL)', example=148),
    'BloodPressure':           fields.Float(required=True, description='Diastolic blood pressure (mm Hg)', example=72),
    'SkinThickness':           fields.Float(required=True, description='Triceps skinfold thickness (mm)', example=35),
    'Insulin':                 fields.Float(required=True, description='2-Hour serum insulin (mu U/ml)', example=0),
    'BMI':                     fields.Float(required=True, description='Body mass index', example=33.6),
    'DiabetesPedigreeFunction':fields.Float(required=True, description='Genetic risk score', example=0.627),
    'Age':                     fields.Float(required=True, description='Age in years', example=50),
})

shap_model = api.model('ShapValue', {
    'feature': fields.String(description='Feature name'),
    'value':   fields.Float(description='SHAP contribution value')
})

output_model = api.model('PredictionOutput', {
    'risk_percent': fields.Float(description='Diabetes risk as a percentage (0–100)'),
    'risk_level':   fields.String(description='Risk category: Low, Moderate, or High'),
    'shap_values':  fields.List(fields.Nested(shap_model), description='Feature contributions')
})

ns = api.namespace('api', description='Prediction endpoints')

@ns.route('/predict')
class Predict(Resource):
    @ns.expect(input_model)
    @ns.marshal_with(output_model)
    @ns.doc(description='Submit patient data and receive a diabetes risk score with explanations')
    def post(self):
        data = api.payload
        features = [float(data[f]) for f in FEATURE_NAMES]
        scaled = scaler.transform([features])
        prob = model_ml.predict_proba(scaled)[0][1]
        risk_percent = round(prob * 100, 1)
        shap_vals = explainer.shap_values(scaled)
        shap_list = sorted([
            {"feature": FEATURE_NAMES[i], "value": round(float(shap_vals[0][i]), 3)}
            for i in range(len(FEATURE_NAMES))
        ], key=lambda x: abs(x["value"]), reverse=True)
        level = "Low" if risk_percent < 30 else "Moderate" if risk_percent < 60 else "High"
        return {"risk_percent": risk_percent, "risk_level": level, "shap_values": shap_list}

if __name__ == '__main__':
    app.run(debug=True, port=5000)