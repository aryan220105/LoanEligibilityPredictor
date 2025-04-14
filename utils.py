import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def df_NaN_Status(df: pd.DataFrame, normalize=False): 
    total = df.shape[0]
    for col in df.columns: 
        nan_count = df[col].isna().sum()
        if normalize: 
            print(f"{col:<20}\t{nan_count / total:.5f}")
        else: 
            print(f"{col:<20}\t{nan_count:>2}/{total}")
            
# Save the model 
import pickle
def save_model(model: any, fp: Path = 'model/loanPredictorModel.pkl') -> None: 
    with open(fp, 'wb') as f:
        pickle.dump(model, f)

# Load model        
def load_model(fp: Path) -> RandomForestClassifier: 
    with open(fp, 'rb') as f:
        model: RandomForestClassifier = pickle.load(f)
    return model

# Data processing
def impute_data(df, label_encoding: bool = False):
    """Handle missing numerical missing values via imputing and encode categorical values"""
    df = df.copy()
    
    # Handle NaN/missing values
    numerical_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Fill NaN-values of numerical columns with Imputer
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
    
    # Fill NaN-values of categorical columns with Imputer
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed']
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float) 
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get the imputed data and encode categorical values via one hot encoding"""
    df = impute_data(df)
    categorical_cols: list[str] = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    encoder = LabelEncoder() if LabelEncoder else OneHotEncoder()
    for col in categorical_cols: 
        df[col] = encoder.fit_transform(df[col])
    return df

# Data Engineering 
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create more sophisticated features to improve model performance.
    """
    df = df.copy()
    
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['DTI'] = df['EMI'] / (df['Total_Income'] + 1)  # add 1 to prevent division by zero
    df['Debt_to_Income'] = df['LoanAmount'] / (df['Total_Income'] + 1)
    df['Income_Per_Capita'] = df['Total_Income'] / (df['Dependents'].astype(float) + 1)
    df['Income_to_EMI_Ratio'] = df['Total_Income'] / (df['EMI'] + 1)  # add 1 to prevent division by zero
    
    # additional features
    df['Income_Stability'] = df['CoapplicantIncome'] / (df['ApplicantIncome'] + 1)
    df['Loan_Income_Ratio'] = df['LoanAmount'] / (df['ApplicantIncome'] + 1)
    df['Loan_Term_Monthly'] = df['Loan_Amount_Term'] / 12  # Convert to years
    df['Monthly_Income'] = df['Total_Income'] / 12
    df['LoanAmount_Log'] = np.log1p(df['LoanAmount'])
    df['ApplicantIncome_Log'] = np.log1p(df['ApplicantIncome'])
    
    # interactive features
    df['Income_Education_Interaction'] = df['ApplicantIncome'] * df['Education']
    df['Loan_Property_Interaction'] = df['LoanAmount'] * df['Property_Area']
    
    # log of income to better handle the skewed distribution
    df['Log_Total_Income'] = np.log1p(df['Total_Income'])
    
    # polynomial features for key relationships
    df['Income_Loan_Ratio_Squared'] = (df['Total_Income'] / (df['LoanAmount'] + 1)) ** 2
    
    return df

def get_train_test_data(train_data: pd.DataFrame, test_size_ratio: float = 0.1): 
    # Preprocess the training and tesstttting data
    X = preprocess_data(train_data.drop('Loan_Status', axis=1))
    y = LabelEncoder().fit_transform(train_data['Loan_Status'])
    
    X = engineer_features(X)
    return train_test_split(X, y, test_size=test_size_ratio)
    

# Sample Predictions
def predict_loan_eligibility(model, input_data):
    """Input sample data to get inference for model"""
    input_df = pd.DataFrame([input_data])
    input_df = preprocess_data(input_df)
    input_df = engineer_features(input_df)
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][1]
    return 'Y' if prediction == 1 else 'N', confidence

def conf_to_canvas(confidence: float) -> float:
    if confidence <= 40:
        return 0.25  
    elif confidence >= 60:
        return 0.98  
    normalized = (confidence - 40) / 20  
    dramatized = normalized ** 2 if confidence < 50 else 1 - (1 - normalized) ** 2
    return 0.25 + dramatized * (0.98 - 0.25)