import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

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
def save_model(model: RandomForestClassifier, fp: Path = 'model/loanPredictorModel.pkl') -> None: 
    with open(fp, 'wb') as f:
        pickle.dump(model, f)

# Load model        
def load_model(fp: Path) -> RandomForestClassifier: 
    with open(fp, 'rb') as f:
        model: RandomForestClassifier = pickle.load(f)
    return model

# Data processing
def preprocess_data(df):
    """Handle missing numerical missing values via imputing and encode categorical values"""
    df = df.copy()
    
    # Handle NaN/missing values
    numerical_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Fill NaN-values of numerical columns with Imputer
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
    
    # Fill NaN-values of categorical columns with Imputer
    # categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History']
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed']
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
        
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols + ['Property_Area']:
        df[col] = le.fit_transform(df[col])
    
    return df

# Data Engineering 
def add_features(df: pd.DataFrame) -> pd.DataFrame: 
    # Adding more features so my model can be fucking accurate
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Income_to_Loan'] = df['Total_Income'] / df['LoanAmount']
    df['Debt_to_Income'] = df['LoanAmount'] / df['Total_Income']
    return df

def get_train_test_data(train_data: pd.DataFrame, test_data: pd.DataFrame): 
    # Preprocess the training and tesstttting data
    X_train = preprocess_data(train_data.drop('Loan_Status', axis=1))
    y_train = LabelEncoder().fit_transform(train_data['Loan_Status'])
    X_test = preprocess_data(test_data)    
    
    X_train = add_features(X_train)
    X_test = add_features(X_test)
    
    return X_train, y_train, X_test

# Sample Predictions
def predict_loan_eligibility(model, input_data):
    """Input sample data to get inference for model"""
    input_df = pd.DataFrame([input_data])
    input_df = preprocess_data(input_df)
    input_df = add_features(input_df)
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][1]
    return 'Y' if prediction == 1 else 'N', confidence