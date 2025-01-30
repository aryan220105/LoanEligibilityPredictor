#!/opt/homebrew/Caskroom/miniforge/base/envs/ml/bin/python3
import streamlit as st
import pandas as pd
import joblib
from utils import preprocess_data, add_features
from pathlib import Path

# Load the trained model
MODEL_PATH: Path = Path("./model/loanPredictorModel.pkl").resolve()
model = joblib.load(MODEL_PATH)
# print(MODEL_PATH)

def predict_loan_eligibility(input_data):
    """Preprocess input to make it compitable for model prediction and provide inference"""
    input_df = pd.DataFrame([input_data])    
    input_df = preprocess_data(input_df)
    input_df = add_features(input_df)
    
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][1]
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
    </style>
    """, unsafe_allow_html=True)

# Streamlit App UI
def main():
    st.title("Loan Eligibility Predictor")
    st.set_page_config(page_title="Loan Eligibility Predictor")
    st.write("Enter your details below to check your loan eligibility.")

    # Add some CSS animations and little bit of styling too 
    add_css_animations()

    # Arrange form into two columns
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
        loan_amount = st.number_input("🏦 Loan Amount", min_value=0, value=10000, step=1000)
        loan_amount_term = st.number_input("⏳ Loan Amount Term (in months)", min_value=0, value=300, step=10)

    property_area = st.selectbox("🏡 Property Area", ["Urban", "Rural", "Semiurban"])

    # On Submit
    if st.button("Check Eligibility"):
        # Prepare input data
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

        status, confidence = predict_loan_eligibility(input_data)

        # Animmated results
        if status == "Approved":
            st.markdown(f'<p class="approved">🎉 Loan Status: {status}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="rejected">❌ Loan Status: {status}</p>', unsafe_allow_html=True)

        # Confidence bar
        st.write("Confidence of Approval:")
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"{confidence:.2%}")

if __name__ == "__main__":
    main()