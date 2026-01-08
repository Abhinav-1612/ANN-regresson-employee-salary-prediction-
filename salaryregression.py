import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Salary Predictor",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = load_model('regression_model.h5')
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler

try:
    model, le_gender, ohe_geo, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

st.sidebar.header("User Information")
st.sidebar.info("Enter customer details to predict their estimated salary.")

with st.sidebar:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    credit_score = st.number_input("Credit Score", 300, 850, 600)
    balance = st.number_input("Account Balance", 0.0, 300000.0, 50000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_card = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active = st.radio("Is Active Member?", ["Yes", "No"])
    exited = st.radio("Has Exited?", ["Yes", "No"])

st.title("💰 Customer Estimated Salary Predictor")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Customer Profile Overview")
    profile_data = {
        "Attribute": ["Geography", "Gender", "Age", "Tenure", "Credit Score", "Balance"],
        "Value": [geography, gender, age, tenure, credit_score, f"${balance:,.2f}"]
    }
    st.table(pd.DataFrame(profile_data))

with col2:
    st.subheader("Prediction Result")
    
    # Preprocessing Logic
    if st.button("Predict Salary"):
        # gender encoding
        gender_encoded = le_gender.transform([gender])[0]
        
        # Create input dict mirroring training data structure
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [le_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [1 if has_card == "Yes" else 0],
            'IsActiveMember': [1 if is_active == "Yes" else 0],
            'Exited': [1 if exited == "Yes" else 0]
        })

        # One-hot encode Geography
        geo_encoded = ohe_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
        
        # Combine 
        final_input_df = pd.concat([input_data, geo_encoded_df], axis=1)
        
        # Scale
        scaled_input = scaler.transform(final_input_df)
        
        # Predict
        prediction = model.predict(scaled_input)
        predicted_salary = prediction[0][0]

        st.markdown(f"""
         <div class="prediction-box">
             <h3 style="color: #28a745; margin-bottom: 0;">Estimated Salary</h3>
             <h1 style="font-size: 3.5em; margin-top: 10px; color: #1a1a1a;">${predicted_salary:,.2f}</h1>
             <p style="color: #666; font-style: italic;">Based on Artificial Neural Network Regression</p>
         </div>
          """, unsafe_allow_html=True)
        
        st.balloons()
    else:
        st.write("Click the button to generate a prediction.")

st.markdown("---")
st.caption("Model: TensorFlow Keras Regression")