import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

# IMPORTANT: Using '../' assumes your 'train (1).csv' is one folder up from this script.
# If you moved the CSV into this folder, change it back to 'train (1).csv'.
df = pd.read_csv("C:/Users/HP/Dropbox/PC/Downloads/train.csv")

# Drop Loan_ID
df = df.drop('Loan_ID', axis=1)

# Target Variable: Convert 'Y'/'N' to 1/0
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Imputation:
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Feature Engineering:
df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanAmount_log'] = np.log(df['LoanAmount'] + 1)
df['Total_Income_log'] = np.log(df['Total_Income'] + 1)

# Drop original columns
df = df.drop(columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income'])

# One-Hot Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop('Loan_Status', axis=1)
y = df_encoded['Loan_Status']

# Save the column names for consistent input feature ordering in the Flask app
feature_cols = X.columns.tolist()
with open('model_features.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

# Standardize numerical features 
scaler = StandardScaler()
X[['LoanAmount_log', 'Total_Income_log']] = scaler.fit_transform(X[['LoanAmount_log', 'Total_Income_log']])

# Save the fitted scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train the Logistic Regression Model
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X, y)

# Save the trained model
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training and saving complete. Check for the three .pkl files.")