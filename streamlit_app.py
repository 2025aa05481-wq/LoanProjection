import streamlit as st
import joblib
import os

# Set page configuration
st.set_page_config(layout='wide', page_title='Loan Prediction App')

st.title('Loan Prediction Application')
st.write('This application predicts loan approval status using various classification models.')

# Path to the directory containing saved models
MODEL_DIR = 'model/'

# Load all trained models
models = {}
model_filenames = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'K-Nearest Neighbor': 'knn_model.pkl',
    'Naive Bayes': 'naive_bayes_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    'XGBoost': 'xgboost_model.pkl'
}

for name, filename in model_filenames.items():
    model_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(model_path):
        models[name] = joblib.load(model_path)
        st.sidebar.success(f'{name} model loaded successfully!')
    else:
        st.sidebar.error(f'Error: {filename} not found in {MODEL_DIR}')

if not models:
    st.error('No models were loaded. Please ensure models are trained and saved in the \'model/\' directory.')
else:
    st.sidebar.header('Model Selection')
    selected_model_name = st.sidebar.selectbox('Choose a classification model:', list(models.keys()))
    st.sidebar.write(f'You selected: {selected_model_name}')

    st.subheader(f'Selected Model: {selected_model_name}')
    st.write('Further application logic and input fields will go here.')

# Placeholder for dataset upload (as per assignment requirements)
st.sidebar.subheader('Upload Test Data (CSV)')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.write("File uploaded successfully! (Processing logic to be added later)")

# Placeholder for evaluation metrics display
st.subheader('Model Evaluation Metrics (To be populated)')
st.write('This section will display accuracy, AUC, precision, recall, F1, and MCC scores.')

# Placeholder for confusion matrix/classification report
st.subheader('Confusion Matrix / Classification Report (To be populated)')
st.write('This section will show the confusion matrix or a classification report.')
