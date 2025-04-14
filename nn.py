import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from typing import Tuple

def create_neural_network_model(input_shape: int) -> tf.keras.Model:
    KERNEL_REGULARIZER_PARAM: float = 1e-3
    model = Sequential([
        Dense(64, activation='relu', 
              input_shape=(input_shape,), 
              kernel_regularizer=l2(KERNEL_REGULARIZER_PARAM)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(48, activation='relu', kernel_regularizer=l2(KERNEL_REGULARIZER_PARAM)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu', kernel_regularizer=l2(KERNEL_REGULARIZER_PARAM)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001, decay=1e-6)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def prepare_data_nn(train_data: pd.DataFrame, test_data: pd.DataFrame
                   ) -> Tuple[np.ndarray, pd.Series, np.ndarray]:
    # Separate features and target from training data
    X_train = train_data.drop('Loan_Status', axis=1)
    y_train = train_data['Loan_Status']
    
    # Define categorical and numerical columns
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                        'Self_Employed', 'Property_Area']
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                      'Loan_Amount_Term']
    
    # Preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(test_data)
    
    # Encode target variable ('Y' as 1, others as 0)
    y_train_encoded = (y_train == 'Y').astype(int)
    
    return X_train_processed, y_train_encoded, X_test_processed

def train_neural_network(X_train: np.ndarray, y_train: pd.Series
                        ) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create model
    model = create_neural_network_model(input_shape=X_tr.shape[1])
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=50, 
        restore_best_weights=True,
        min_delta=0.001
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=10, 
        min_lr=1e-6
    )
    
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, X_val, y_val

def evaluate_model(model: tf.keras.Model, X_val: np.ndarray, y_val: pd.Series) -> None: 
    y_pred_proba = model.predict(X_val).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("Validation Metrics:")
    print(classification_report(y_val, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print(f"\nROC AUC Score: {roc_auc_score(y_val, y_pred_proba):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")

def nn_main(train_data: pd.DataFrame, test_data: pd.DataFrame
           ) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    X_train, y_train, X_test = prepare_data_nn(train_data, test_data)
    
    model, X_val, y_val = train_neural_network(X_train, y_train)
    
    evaluate_model(model, X_val, y_val)
    
    test_predictions_proba = model.predict(X_test).flatten()
    test_predictions = (test_predictions_proba > 0.5).astype(int)
    
    return model, test_predictions, test_predictions_proba

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train Neural Network for Loan Prediction")
    parser.add_argument('--train_path', type=str, default='train.csv',
                        help='Path to the training CSV file')
    parser.add_argument('--test_path', type=str, default='test.csv',
                        help='Path to the test CSV file')
    
    args = parser.parse_args()
    
    try:
        train_df = pd.read_csv(args.train_path)
        test_df = pd.read_csv(args.test_path)
    except Exception as e:
        raise FileNotFoundError("Incorrect file paths.") from e
    
    model, predictions, probabilities = nn_main(train_df, test_df)

