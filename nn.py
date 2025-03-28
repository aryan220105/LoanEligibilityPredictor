# neural_network.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def nn_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing pipeline specifically for neural network"""
    df = impute_data(df)
    
    # One-hot encode categorical features
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                       'Self_Employed', 'Property_Area']
    numerical_cols = [col for col in df.columns if col not in categorical_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])
    
    processed = preprocessor.fit_transform(df)
    return processed

# Get processed data
X_train_nn = nn_preprocess_data(train_data.drop('Loan_Status', axis=1))
X_test_nn = nn_preprocess_data(test_data)
y_train_nn = LabelEncoder().fit_transform(train_data['Loan_Status'])

# Neural Network Architecture
model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_nn.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_nn.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall()])

# Training with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model_nn.fit(
    X_train_nn, y_train_nn,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluation
nn_pred_proba = model_nn.predict(X_train_nn).flatten()
nn_pred = (nn_pred_proba > 0.5).astype(int)
print(f"Neural Network F1 Score: {f1_score(y_train_nn, nn_pred)}")
print(f"ROC-AUC Score: {roc_auc_score(y_train_nn, nn_pred_proba)}")