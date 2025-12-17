# ============================
# app.py (FULL CORRECTED VERSION)
# ============================
import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("EMPLOYEE ATTRITION APP")
st.write("Enter values for all features and get a prediction")

# ---- Load model ----
@st.cache_resource
def load_model():
    with open("logistic_regression_model (1).pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ---- Get feature names EXACTLY as used during training ----
if hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    st.error("❌ Model does not contain feature names. Retrain using pandas DataFrame.")
    st.stop()

st.subheader("Input Features")

# ---- User input ----
input_data = {}
for feature in FEATURES:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0,
        format="%.2f"
    )

# ---- Create DataFrame in SAME order ----
input_df = pd.DataFrame([input_data], columns=FEATURES)

st.write("### Input Data Sent to Model")
st.dataframe(input_df)

# ---- Prediction ----
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Prediction: {prediction[0]}")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)
            st.write("### Prediction Probability")
            st.write(probability)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


# ============================
# requirements.txt
# ============================
# streamlit
# pandas
# scikit-learn


