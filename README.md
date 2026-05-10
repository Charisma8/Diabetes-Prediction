# Diabetes Risk Prediction Dashboard

This project turns a standard diabetes classifier into a stronger portfolio and academic demo by combining:

- model benchmarking across multiple algorithms
- explainable predictions with SHAP
- intervention simulation for actionable risk reduction
- a Flask API plus a polished React dashboard

## Current Scope

The working prototype uses the public `diabetes_prediction_dataset.csv` file in `backend/data/` and focuses on lifestyle and clinical predictors:

- gender
- age
- hypertension
- heart disease
- smoking history
- BMI
- HbA1c
- blood glucose

Important note: this version does **not** include real genomic markers. The codebase is structured so a hereditary or genomic module can be added later without rebuilding the UI or API.

## Project Structure

- `backend/train.py`: trains and saves the deployment bundle
- `backend/app.py`: Flask API with prediction, metadata, and simulation endpoints
- `backend/ml_pipeline.py`: shared preprocessing, model training, defaults, and scenario logic
- `frontend/`: React dashboard built with Vite and Recharts

## How To Run

### Backend

```powershell
cd backend
venv\Scripts\python.exe train.py
venv\Scripts\python.exe app.py
```

API endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /simulate`

### Frontend

```powershell
cd frontend
npm run dev
```

The frontend expects the backend at `http://127.0.0.1:5000` by default.

## Standout Features You Can Talk About

- Compared multiple models instead of shipping the first one that worked.
- Added SHAP-based local explanations to improve trust and interpretability.
- Built a "what-if" simulator so predictions become actionable.
- Separated lifestyle, clinical, and personal background influence in the dashboard.
- Designed the API for future hereditary or genomic data integration.

## Suggested Research Framing

Use this problem framing in your report:

> Explainable machine learning can improve early diabetes risk screening by combining lifestyle and clinical data into a personalized risk assessment workflow, while remaining extensible to hereditary or genomic inputs.

## Strong Next Upgrade

If you want to align even more closely with your original problem statement, the best next enhancement is:

1. add a real family-history or genomic-risk dataset
2. retrain a multimodal fusion model
3. compare `clinical only` vs `lifestyle + clinical + hereditary`
