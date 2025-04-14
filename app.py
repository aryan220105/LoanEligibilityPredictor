#!/opt/homebrew/Caskroom/miniforge/base/envs/ml/bin/python3
import streamlit as st
import pandas as pd
import joblib
from utils import *
from pathlib import Path
import numpy as np 
import warnings
warnings.filterwarnings('ignore')

# Load the trained models
RFC_MODEL_PATH: Path = Path("./model/rfc.pkl").resolve()
NN_MODEL_PATH: Path = Path("./model/nn.pkl").resolve()

# Load models
rfc_model = joblib.load(RFC_MODEL_PATH)
nn_model = joblib.load(NN_MODEL_PATH)

def get_prediction_and_confidence(model, input_df, model_type='RFC'):
    """Get prediction and confidence based on model type"""
    if model_type == 'RFC':
        # For scikit-learn models that have predict_proba
        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][1]
    else:
        # For Keras/Sequential models
        pred_raw = model.predict(input_df)
        prediction = 1 if pred_raw[0][0] > 0.5 else 0
        confidence = pred_raw[0][0]  # Keras models output probability directly
    
    return prediction, confidence

def predict_loan_eligibility(input_data, model_choice='RFC'):
    """Preprocess input to make it compatible for model prediction and provide inference"""
    input_df = pd.DataFrame([input_data])    
    input_df = preprocess_data(input_df)
    input_df = add_features(input_df)
    
    # Select the appropriate model based on user's choice
    if model_choice == 'RFC':
        model = rfc_model
        prediction, confidence = get_prediction_and_confidence(model, input_df, 'RFC')
    else:
        model = nn_model
        prediction, confidence = get_prediction_and_confidence(model, input_df, 'NN')
    
    return 'Approved' if prediction == 1 else 'Rejected', confidence


def add_css_animations():
    """Custom CSS for animations and styling"""
    st.markdown("""
    <style>
    /* Approved animation */
    @keyframes approved {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    .approved {
        color: #28a745;
        font-size: 24px;
        font-weight: bold;
        animation: approved 1s ease-in-out;
    }

    /* Rejected animation */
    @keyframes rejected {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    .rejected {
        color: #dc3545;
        font-size: 24px;
        font-weight: bold;
        animation: rejected 1s ease-in-out;
    }

    /* Confidence bar */
    .confidence-bar {
        width: 100%;
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 10px;
    }

    .confidence-fill {
        height: 100%;
        background-color: #28a745;
        transition: width 1s ease-in-out;
    }
    
    /* Model selection card styles */
    .model-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: #6c757d;
    }
    
    .model-card.selected {
        border-color: #007bff;
        background-color: #f0f7ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit App UI
def main():
    st.title("Loan Eligibility Predictor")
    # st.set_page_config(page_title="Loan Eligibility Predictor")
    st.write("Enter your details below to check your loan eligibility.")
    add_css_animations()
    
    # Model selection section
    st.subheader("Select Prediction Model")
    model_choice = st.radio(
        "Choose a model for prediction:",
        ["Random Forest Classifier (RFC)", "Neural Network (NN)"],
        index=0,
        help="RFC models are generally more interpretable but NN models might capture complex patterns better."
    )
    
    # Converting radio selection to model key
    model_key = 'RFC' if 'Random Forest' in model_choice else 'NN'
    
    # Add a divider
    st.markdown("---")

    # two col form 
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Details")
        gender = st.selectbox("👩‍💼 Gender", ["Male", "Female"])
        married = st.selectbox("💍 Marital Status", ["Yes", "No"])
        dependents = st.selectbox("👨‍👩‍👧 Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])

    with col2:
        st.subheader("Financial Details")
        self_employed = st.selectbox("💼 Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("💰 Applicant Income", min_value=0, value=500000, step=1000)
        coapplicant_income = st.number_input("👫 Coapplicant Income", min_value=0, value=0, step=1000)
        loan_amount = st.number_input("🏦 Loan Amount", min_value=0, value=100000, step=1000)
        loan_amount_term = st.number_input("⏳ Loan Amount Term (in months)", min_value=0, value=300, step=10)

    property_area = st.selectbox("🏡 Property Area", ["Urban", "Rural", "Semiurban"])

    # On Submit click
    if st.button("Check Eligibility"):
        input_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Property_Area': property_area
        }

        # Pass the model choice to the prediction function
        with st.spinner(f"Analyzing your application with {model_choice}..."):
            status, confidence = predict_loan_eligibility(input_data, model_key)
        
        # Animated results
        if status == "Approved":
            st.markdown(f'<p class="approved">🎉 Loan Status: {status}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="rejected">❌ Loan Status: {status}</p>', unsafe_allow_html=True)

        # Show which model made the prediction
        st.markdown(f"*Prediction made using {model_choice}*")

        # Confidence bar
        st.write("Confidence of Approval:")
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"{confidence * 100:.2f}%")
        
        # Add option to compare with the other model
        st.markdown("---")
        if st.button("Compare with other model"):
            other_model_key = 'NN' if model_key == 'RFC' else 'RFC'
            other_model_name = "Neural Network" if other_model_key == 'NN' else "Random Forest Classifier"
            
            with st.spinner(f"Running comparison with {other_model_name}..."):
                other_status, other_confidence = predict_loan_eligibility(input_data, other_model_key)
            
            st.subheader(f"Comparison with {other_model_name}")
            if other_status == "Approved":
                st.markdown(f'<p class="approved">🎉 Loan Status: {other_status}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="rejected">❌ Loan Status: {other_status}</p>', unsafe_allow_html=True)
                
            st.write("Confidence of Approval:")
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {other_confidence * 100}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"{other_confidence * 100:.2f}%")
            
            # Display confidence difference
            confidence_diff = abs(confidence - other_confidence) * 100
            st.markdown(f"*The models differ by {confidence_diff:.2f}% in confidence*")

            # Model agreement indicator
            if status == other_status:
                st.success(f"✓ Both models agree on the loan status ({status})")
            else:
                st.warning("⚠️ The models disagree on the loan status. Consider reviewing the application more carefully.")

if __name__ == "__main__":
    main()