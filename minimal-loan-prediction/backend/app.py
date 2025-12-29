import flask
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from flask_cors import CORS # Required to allow your front-end to talk to the back-end

# --- Initialization ---
app = Flask(__name__)
# Enable CORS to allow the frontend (running on a different port/origin) to access the API
CORS(app)

# --- Load Trained Assets ---
try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        FEATURE_COLS = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model, features, and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Required model files (logistic_regression_model.pkl, model_features.pkl, scaler.pkl) not found.")
    print("Please ensure you have run the data preprocessing and training steps to create these files.")
    # Exit or handle the error gracefully for a production app
    model = None
    FEATURE_COLS = []
    scaler = None

# --- Preprocessing Function ---
def preprocess_input(data: dict, feature_cols: list, scaler) -> pd.DataFrame:
    """
    Transforms the raw JSON input from the frontend into the format 
    expected by the trained Logistic Regression model.
    """
    # 1. Create a DataFrame from the single input record
    input_df = pd.DataFrame([data])

    # 2. Feature Engineering
    # Calculate Total Income and apply log transformation
    input_df['Total_Income'] = input_df['income'] + input_df['coapplicant_income']
    input_df['Total_Income_log'] = np.log(input_df['Total_Income'] + 1)
    
    # Apply log transformation to Loan Amount
    input_df['LoanAmount_log'] = np.log(input_df['loan'] + 1)

    # 3. Create a DataFrame with all expected features initialized to 0
    # This ensures all one-hot encoded columns are present, even if their value is 0.
    processed_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    # 4. Map and populate non-encoded numerical/direct features
    processed_df['Dependents'] = input_df['dependents'].replace('3+', '3').astype(int)
    processed_df['Loan_Amount_Term'] = input_df['loan_term']
    processed_df['Credit_History'] = input_df['credit_history'].astype(float)

    # 5. Populate One-Hot Encoded features
    
    # Gender
    if input_df['gender'].iloc[0] == 'Male':
        processed_df['Gender_Male'] = 1

    # Married
    if input_df['married'].iloc[0] == 'Yes':
        processed_df['Married_Yes'] = 1

    # Education
    if input_df['education'].iloc[0] == 'Not Graduate':
        processed_df['Education_Not Graduate'] = 1
        
    # Self_Employed (The form uses 'Work Profile': Salaried/Self Employed)
    # 'Self Employed' maps to 'Self_Employed_Yes' = 1, 'Salaried' maps to 0.
    if input_df['work'].iloc[0] == 'Self Employed':
        processed_df['Self_Employed_Yes'] = 1

    # Property Area
    property_area = input_df['property'].iloc[0]
    if property_area == 'Semiurban':
        processed_df['Property_Area_Semiurban'] = 1
    elif property_area == 'Urban':
        processed_df['Property_Area_Urban'] = 1

    # 6. Apply Scaling to the log-transformed features
    # NOTE: The scaler was fit on X which only contained the log features and other binary/categorical ones.
    # In the training step, we fit the scaler on X[['LoanAmount_log', 'Total_Income_log']]
    numerical_features = ['LoanAmount_log', 'Total_Income_log']
    
    # Extract the features to be scaled from the pre-processed data
    features_to_scale = pd.DataFrame({
        'LoanAmount_log': input_df['LoanAmount_log'],
        'Total_Income_log': input_df['Total_Income_log']
    })
    
    # Apply the pre-fitted scaler
    scaled_features = scaler.transform(features_to_scale)

    # Update the processed_df with scaled values
    processed_df['LoanAmount_log'] = scaled_features[0, 0]
    processed_df['Total_Income_log'] = scaled_features[0, 1]

    # 7. Final check to ensure feature order is correct (should be guaranteed by initialization but safe to check)
    X_final = processed_df[feature_cols]
    
    return X_final

# --- API Route ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500
        
    try:
        data = request.get_json(force=True)
        
        # Preprocess the input data
        X_predict = preprocess_input(data, FEATURE_COLS, scaler)
        
        # Make Prediction
        # model.predict() returns a numpy array, which we extract
        prediction_val = model.predict(X_predict)[0]
        
        # Convert the prediction (0 or 1) to a readable string
        if prediction_val == 1:
            prediction_text = "Approved (Y)"
        else:
            prediction_text = "Not Approved (N)"
            
        return jsonify({'prediction': prediction_text})
        
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 400

# --- Run the App ---
if __name__ == '__main__':
    # The frontend is configured to call http://127.0.0.1:5000/predict
    app.run(host='127.0.0.1', port=5000, debug=True)