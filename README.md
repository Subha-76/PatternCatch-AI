# ThinkPattern AI
### Behavioral Pattern Prediction Using Sequential Machine Learning Models

ThinkPattern AI is a Python project template for exploring how chronological activity records can be used to predict a future behavioral outcome. It includes a small illustrative dataset, preprocessing and sequence-building utilities, LSTM/GRU training code, evaluation helpers, prediction logic, and a Streamlit dashboard.

> **Status:** Source files and sample data only. The application has not been run and no model has been trained as part of creating this project package.

## Features

- Load and validate time-ordered behavioral activity records
- Prepare numerical features and target values
- Build sliding-window sequences for sequential models
- Train an LSTM or GRU regression model when you choose to run training
- Compare predictions using MAE and RMSE
- Display sample records and a basic dashboard
- Keep model artifacts local rather than committing them to Git

## Project structure

```text
ThinkPattern-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample_behavior.csv
├── docs/
│   └── project_overview.md
├── models/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── sequence_builder.py
    ├── train_model.py
    ├── predict_behavior.py
    └── evaluation.py
```

## Dataset

`data/sample_behavior.csv` is a **synthetic demonstration dataset**, not real user data. Each row represents one day's activity for an example anonymous participant.

Columns:
- `user_id`: synthetic participant identifier
- `timestamp`: date/time of the observation
- `study_minutes`: time spent studying that day
- `app_sessions`: number of app sessions that day
- `tasks_completed`: tasks completed that day
- `sleep_hours`: hours slept that day
- `next_day_study_minutes`: illustrative target value for the next day's study duration

The target column is included for teaching and prototyping. For a rigorous experiment, use a clearly defined real or ethically collected dataset and check how the target was generated. Do not interpret predictions as certain statements about a person's behavior.

## Setup (only when you decide to run it)

Requires Python 3.10 or newer.

1. Extract the ZIP file.
2. Open the `ThinkPattern-AI` folder in VS Code.
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate it:
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

The dashboard can show the sample data without a trained model. To train a model, use the training instructions below.

## Train a model (optional; not run during packaging)

For example, from the project root:
```bash
python -m src.train_model --model lstm --epochs 20
```

To use GRU instead:
```bash
python -m src.train_model --model gru --epochs 20
```

Training reads the CSV and writes a model file under `models/`. This may require additional memory and disk space depending on your environment. Model artifacts are ignored by Git by default.

## GitHub upload

1. Create a new repository on GitHub named `ThinkPattern-AI`.
2. Extract this ZIP and open its folder in VS Code.
3. Open the VS Code terminal in the project root.
4. Run:
   ```bash
   git init
   git add .
   git commit -m "Add ThinkPattern AI project"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/ThinkPattern-AI.git
   git push -u origin main
   ```
5. Replace `YOUR-USERNAME` with your GitHub username.

If Git is not installed, install Git first or use GitHub's web upload option. Do not commit personal data, API keys, virtual environments, or large trained model files.

## Evaluation

For time-dependent data, split observations chronologically rather than randomly. The included helpers report Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for a regression target. Results from the small synthetic dataset are only a demonstration and are not evidence of real-world predictive performance.

## Responsible use

- Obtain informed consent before collecting personal behavioral data.
- Minimize and anonymize collected data.
- Do not use predictions as diagnoses or as the sole basis for high-impact decisions.
- Explain uncertainty and provide users with a way to correct inaccurate records.
