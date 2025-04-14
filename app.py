#!/opt/homebrew/Caskroom/miniforge/base/envs/ml/bin/python3
import streamlit as st
import pandas as pd
import joblib
from utils import *
from pathlib import Path
import numpy as np 
import warnings
warnings.filterwarnings('ignore')

RFC_MODEL_PATH: Path = Path("./model/rfc.pkl").resolve()
NN_MODEL_PATH: Path = Path("./model/nn.pkl").resolve()

# load models via joblib
rfc_model = joblib.load(RFC_MODEL_PATH)
nn_model = joblib.load(NN_MODEL_PATH)

def get_prediction_and_confidence(model, input_df, model_type='RFC'):
    """Get prediction and confidence based on model type"""
    if model_type == 'RFC':
        # for scikit-learn models which has predict_proba
        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][1]
    else:
        # for Keras/Sequential models which does not have predict_probe
        pred_raw = model.predict(input_df)
        prediction = 1 if pred_raw[0][0] > 0.5 else 0
        confidence = pred_raw[0][0]  # Keras models outputs probability directly
    
    return prediction, confidence

def predict_loan_eligibility(input_data, model_choice='RFC'):
    """Preprocess input to make it compatible for model prediction and provide inference"""
    input_df = pd.DataFrame([input_data])    
    input_df = preprocess_data(input_df)
    input_df = engineer_features(input_df)
    
    # select model based on user's choice
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

    /* Model comparison styles */
    .comparison-container {
        display: flex;
        gap: 20px;
        margin-top: 20px;
    }
    
    .model-result {
        flex: 1;
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    
    .comparison-header {
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# session state vars (to achieve reactive state)
def init_session_state():
    if 'input_data' not in st.session_state:
        st.session_state.input_data = None
    if 'show_comparison' not in st.session_state:
        st.session_state.show_comparison = False
    if 'primary_prediction' not in st.session_state:
        st.session_state.primary_prediction = None
    if 'primary_confidence' not in st.session_state:
        st.session_state.primary_confidence = None
    if 'primary_model' not in st.session_state:
        st.session_state.primary_model = None

def toggle_comparison():
    st.session_state.show_comparison = not st.session_state.show_comparison

# MAIN Streamlit App UI
def main():
    st.title("Loan Eligibility Predictor")
    add_css_animations()
    
    # Init session state
    init_session_state()
    
    st.subheader("Select Prediction Model")
    model_choice = st.radio(
        "Choose a model for prediction:",
        ["Random Forest Classifier (RFC)", "Neural Network (NN)"],
        index=0,
        help="RFC models are generally more interpretable but NN models might capture complex patterns better."
    )
    
    # Converting radio selection to model key
    model_key = 'RFC' if 'Random Forest' in model_choice else 'NN'
    
    st.markdown("---")

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

    # input data dict 
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

    check_eligibility = st.button("Check Eligibility")
    
    if check_eligibility:
        st.session_state.input_data = input_data
        st.session_state.primary_model = model_key
        st.session_state.show_comparison = False
        
        with st.spinner(f"Analyzing your application with {model_choice}..."):
            status, confidence = predict_loan_eligibility(input_data, model_key)
            st.session_state.primary_prediction = status
            st.session_state.primary_confidence = confidence
        
        display_prediction(status, confidence, model_choice)
        
        st.button("Compare with other model", on_click=toggle_comparison)
    
    if st.session_state.show_comparison and st.session_state.input_data is not None:
        display_model_comparison(st.session_state.input_data, st.session_state.primary_model)
    
    # if we have results already in session state but didn't just click the check button to basically update session
    elif not check_eligibility and st.session_state.primary_prediction is not None:
        # Re-display the last prediction
        display_prediction(
            st.session_state.primary_prediction, 
            st.session_state.primary_confidence, 
            "Random Forest Classifier (RFC)" if st.session_state.primary_model == "RFC" else "Neural Network (NN)"
        )
        
        # Add compare button
        st.button("Compare with other model", on_click=toggle_comparison)
        
        if st.session_state.show_comparison:
            display_model_comparison(st.session_state.input_data, st.session_state.primary_model)

def display_prediction(status, confidence, model_name):
    """Display a single prediction result"""
    if status == "Approved":
        st.markdown(f'<p class="approved">🎉 Loan Status: {status}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="rejected">❌ Loan Status: {status}</p>', unsafe_allow_html=True)

    st.markdown(f"*Prediction made using {model_name}*")

    # Confidence bar
    st.write("Confidence of Approval:")
    st.markdown(f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
    </div>
    """, unsafe_allow_html=True)
    st.write(f"{confidence * 100:.2f}%")
    
    st.markdown("---")

def display_model_comparison(input_data, primary_model_key):
    """Display comparison between RFC and NN models"""
    st.subheader("Model Comparison")
    
    rfc_status, rfc_confidence = predict_loan_eligibility(input_data, 'RFC')
    nn_status, nn_confidence = predict_loan_eligibility(input_data, 'NN')
    
    # 2 cols for side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Random Forest Classifier")
        if rfc_status == "Approved":
            st.markdown(f'<p class="approved">🎉 Status: {rfc_status}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="rejected">❌ Status: {rfc_status}</p>', unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {rfc_confidence * 100}%;"></div>
        </div>
        <p>Confidence: {rfc_confidence * 100:.2f}%</p>
        """, unsafe_allow_html=True)
        
        if primary_model_key == 'RFC':
            st.info("✓ Currently selected model")
    
    with col2:
        st.markdown("### Neural Network")
        if nn_status == "Approved":
            st.markdown(f'<p class="approved">🎉 Status: {nn_status}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="rejected">❌ Status: {nn_status}</p>', unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {nn_confidence * 100}%;"></div>
        </div>
        <p>Confidence: {nn_confidence * 100:.2f}%</p>
        """, unsafe_allow_html=True)
        
        if primary_model_key == 'NN':
            st.info("✓ Currently selected model")
    
    st.markdown("---")
    
    # confidence difference
    confidence_diff = abs(rfc_confidence - nn_confidence) * 100
    st.markdown(f"**Models confidence difference:** {confidence_diff:.2f}%")
    
    if rfc_status == nn_status:
        st.success(f"✓ Both models agree on the loan status ({rfc_status})")
    else:
        st.warning("⚠️ The models disagree on the loan status. Consider reviewing the application more carefully.")
        
        st.markdown("""
        ### Why might the models disagree?
        
        Models can disagree for several reasons:
        
        1. **Different features importance**: Each model weighs features differently
        2. **Training data sensitivity**: Different models may be sensitive to different patterns in the training data
        3. **Model complexity**: Neural networks might capture more complex relationships than Random Forests
        4. **Boundary cases**: Your application might be near the decision boundary for both models
        
        When models disagree, it's often beneficial to have a human review the case more carefully.
        """)

if __name__ == "__main__":
    main()