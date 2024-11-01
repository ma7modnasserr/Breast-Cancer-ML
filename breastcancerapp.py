import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and scaler
models = {
    'Logistic Regression': joblib.load(r'C:\Users\Mahmoud\Downloads\streamlit\logistic_regression_model.pkl'),
    'Linear SVM': joblib.load(r'C:\Users\Mahmoud\Downloads\streamlit\linear_svm_model.pkl'),
    'Gaussian SVM': joblib.load(r'C:\Users\Mahmoud\Downloads\streamlit\gaussian_svm_model.pkl'),
    'Polynomial SVM': joblib.load(r'C:\Users\Mahmoud\Downloads\streamlit\polynomial_svm_model.pkl'),
    'Sigmoid SVM': joblib.load(r'C:\Users\Mahmoud\Downloads\streamlit\sigmoid_svm_model.pkl')
}
scaler = joblib.load(r'C:\Users\Mahmoud\Downloads\streamlit\scaler.pkl')

# Load dataset to get feature names (modify the path to your dataset)
data = pd.read_csv(r'C:\Users\Mahmoud\Downloads\streamlit\breast cancer.csv')  
X = data.drop(columns=['diagnosis', 'id'])  # Adjust based on your dataset

# Streamlit App Title
st.title("Breast Cancer Detection App")
st.write("Enter the tumor's features to predict if it is benign or malignant.")

# Model selection
model_name = st.selectbox("Choose a model", list(models.keys()))
selected_model = models[model_name]

# Input fields for each feature
def user_input_features():
    feature_names = [col for col in X.columns]  # Use feature names from the dataset
    data = {name: st.number_input(f"Enter {name}:", value=0.0) for name in feature_names}
    return pd.DataFrame([data])

# CSV file uploader
uploaded_file = st.file_uploader("Upload a CSV file with features", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(input_df)

    # Check if the uploaded data has the correct columns
    if set(input_df.columns) == set(X.columns):
        input_scaled = scaler.transform(input_df)

        # Predictions for each row
        predictions = selected_model.predict(input_scaled)

        # Display predictions
        for i, prediction in enumerate(predictions):
            result = "Malignant" if prediction == 1 else "Benign"
            st.write(f"Prediction for row {i + 1}: The model predicts this tumor is **{result}**.")
    else:
        st.error("The uploaded file does not contain the correct feature names.")
else:
    input_df = user_input_features()
    input_scaled = scaler.transform(input_df)

    # Prediction for manual input
    if st.button("Predict"):
        prediction = selected_model.predict(input_scaled)
        if prediction[0] == 1:
            st.write("The model predicts this tumor is **Malignant**.")
        else:
            st.write("The model predicts this tumor is **Benign**.")
