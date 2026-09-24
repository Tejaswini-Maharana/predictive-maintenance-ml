# Predictive Maintenance Using Machine Learning

## Project Overview

This project develops a machine learning based Predictive Maintenance System using the AI4I 2020 Predictive Maintenance Dataset.

The system analyzes machine operating parameters and predicts whether a machine is likely to experience failure.

A Streamlit dashboard is also developed to provide an interactive web interface for entering sensor values and viewing machine condition predictions.

## Machine Learning Models

Three machine learning algorithms are implemented:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

The models are compared using:

- Accuracy
- Precision
- Recall
- F1 Score

The model with the highest F1 Score is selected as the final model.

## Input Parameters

The system uses the following machine parameters:

- Machine Type
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

## Output

The system provides:

- Machine Failure Prediction
- Failure Probability
- Maintenance Recommendation
- Estimated Maintenance Time
- Sensor Warnings

## Dataset

Dataset:

AI4I 2020 Predictive Maintenance Dataset

The dataset contains machine operating parameters and machine failure information.

## Project Structure

Predictive_Maintenance/

    ai4i2020.csv
    predictive_maintenance.py
    dashboard.py
    requirements.txt
    README.md
    .gitignore

    model/
        predictive_maintenance_model.pkl

    graphs/
        failure_distribution.png
        air_temperature_distribution.png
        process_temperature_distribution.png
        rotational_speed_distribution.png
        torque_distribution.png
        tool_wear_distribution.png
        speed_vs_torque.png
        tool_wear_vs_failure.png
        failure_types.png
        correlation_heatmap.png
        model_comparison.png
        confusion_matrix.png
        feature_importance.png

## Installation

Open the VS Code terminal and run:

pip install -r requirements.txt

## Train the Machine Learning Model

Run:

python predictive_maintenance.py

This will:

- Load the dataset
- Analyze the data
- Train machine learning models
- Compare model performance
- Generate graphs
- Save the best model

## Run the Dashboard

After training the model, run:

streamlit run dashboard.py

The Streamlit application will open in the browser.

## Maintenance Logic

The project uses sensor conditions and model failure probability to provide maintenance recommendations.

The estimated maintenance times are rule-based project estimates.

The AI4I dataset does not contain actual Remaining Useful Life (RUL) or maintenance-history information.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Pickle

## Project Applications

Predictive maintenance can be used to:

- Detect potential machine failures
- Monitor machine operating conditions
- Reduce unexpected downtime
- Support maintenance planning
- Analyze machine sensor parameters
- Improve machine reliability

## Author

Mechanical Engineering Final Year Project