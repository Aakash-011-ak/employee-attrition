# ============================
# app.py
# ============================
import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("EMPLOYEE ATTRITION")
st.write("Enter feature values and get a prediction")

# ---- Load model ----
@st.cache_resource
def load_model():
    with open("logistic_regression_model (1).pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---- Define feature names ----
# IMPORTANT: Update this list to EXACTLY match the features used during training
FEATURES = [
    "feature_1",
    "feature_2",
    "feature_3",
]

st.subheader("Input Features")

input_data = {}
for feature in FEATURES:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

input_df = pd.DataFrame([input_data])

st.write("### Input Data")
st.dataframe(input_df)

# ---- Prediction ----
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)
            st.success(f"Prediction: {prediction[0]}")
            st.write("Prediction Probability:")
            st.write(probability)
        else:
            st.success(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ============================
# requirements.txt
# ============================
# streamlit
# pandas
# numpy
# scikit-learn


