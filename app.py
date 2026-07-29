import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load('models/extra_trees_credit_model.pickle')
cols = ['sex', 'housing', 'saving_accounts', 'checking_account']
encoders = {col: joblib.load(f'models/{col}_encoder.pickle') for col in cols}

st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict credit risk (Good vs. Bad).")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0 to 3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
checking = st.selectbox("Checking Account", ["little", "moderate", "rich"])
amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (Months)", min_value=1, value=12)

if st.button("Predict Risk"):
    # Encode input values
    input_data = pd.DataFrame([{
        'age': age,
        'sex': encoders['sex'].transform([sex])[0],
        'job': job,
        'housing': encoders['housing'].transform([housing])[0],
        'saving_accounts': encoders['saving_accounts'].transform([saving])[0],
        'checking_account': encoders['checking_account'].transform([checking])[0],
        'credit_amount': amount,
        'duration': duration
    }])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("The predicted credit risk is **Good** (Low Risk).")
    else:
        st.error("The predicted credit risk is **Bad** (High Risk).")