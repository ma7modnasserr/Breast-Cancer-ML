# Breast Cancer Detection

```markdown
# Breast Cancer Detection Model and Deployment

This repository contains a machine learning model for breast cancer detection and a web application built with Streamlit to facilitate user interaction. The model predicts whether a tumor is benign or malignant based on various features of the tumor.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [How to Use](#how-to-use)
- [Requirements](#requirements)
- [Contributing](#contributing)

## Project Overview

The project involves:
- Data preprocessing and exploratory data analysis
- Training multiple machine learning models
- Evaluating the models based on accuracy and performance metrics
- Developing a Streamlit application for user input and predictions

## Dataset

The dataset used in this project is the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) which contains features computed from digitized images of breast cancer fine needle aspirate (FNA) samples.

- The dataset contains the following features:
  - `radius_mean`
  - `texture_mean`
  - `perimeter_mean`
  - `area_mean`
  - `smoothness_mean`
  - `compactness_mean`
  - `concavity_mean`
  - `concave points_mean`
  - `symmetry_mean`
  - `fractal_dimension_mean`
  - `radius_se`
  - `texture_se`
  - `perimeter_se`
  - `area_se`
  - `smoothness_se`
  - `compactness_se`
  - `concavity_se`
  - `concave points_se`
  - `symmetry_se`
  - `fractal_dimension_se`
  - `radius_worst`
  - `texture_worst`
  - `perimeter_worst`
  - `area_worst`
  - `smoothness_worst`
  - `compactness_worst`
  - `concavity_worst`
  - `concave points_worst`
  - `symmetry_worst`
  - `fractal_dimension_worst`
  - `diagnosis` (0 = benign, 1 = malignant)

## Model Training

The project involves the training of multiple models, including:
- Logistic Regression
- Support Vector Machine (SVM) with different kernels:
  - Linear
  - Gaussian (RBF)
  - Polynomial
  - Sigmoid

### Steps:
1. Load the dataset and preprocess the data.
2. Split the data into training and testing sets.
3. Standardize the features.
4. Train the models and evaluate their performance using accuracy and classification reports.
5. Save the trained models and the scaler using `joblib`.


## Model Evaluation

The model's performance is evaluated using metrics such as accuracy, confusion matrix, and classification report. Results for each model are compared, and the best-performing models are saved for deployment.

## Deployment

A Streamlit application is provided to allow users to interact with the model. Users can:
- Input tumor features manually
- Upload a CSV file containing multiple records for prediction

The deployment code is contained in `breactcancerapp.py`.

### Run the Application
To run the Streamlit application:
1. Install the required packages (see Requirements).
2. Navigate to the directory containing `breastcancerapp.py`.
3. Run the following command in your terminal:
   ```bash
   streamlit run breastcancerapp.py
   ```

## How to Use

1. **Manual Input**:
   - Select a model from the dropdown menu.
   - Enter the tumor features in the provided input fields.
   - Click on the "Predict" button to see the prediction result.

2. **CSV File Upload**:
   - Click on the "Upload a CSV file" button to upload a file with tumor features.
   - The application will predict the diagnosis for each row in the uploaded file and display the results.

## Requirements

Make sure you have the following packages installed:
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- joblib
- streamlit

You can install these using pip:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib streamlit
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.
