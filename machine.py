
import streamlit as st
import pandas as pd
import joblib  # For loading the trained model
import numpy as np
from PIL import Image

# Load the trained model
model = joblib.load("machine.pkl") 
le=joblib.load('le.pkl')
le1=joblib.load('le1.pkl')# Ensure the model file exists
scaler=joblib.load('scaler.pkl')


# App Title
st.title("Machine Failure Prediction")



st.image(Image.open(r'C:\Users\user\Downloads\gears-cogs-mechanism-industrial-machine-260nw-1901797519.webp'))
# Description
st.subheader("Enter Machine Features Below")
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("🌡 Temperature (°C)", min_value=0.0, max_value=200.0, value=50.0)
    vibration = st.number_input("🔊 Vibration (Hz)", min_value=0.0, max_value=10.0, value=2.0)
    equipment_options = ["Turbine", "Compressor", "Pump"]
    equipment = st.selectbox("🏭 Equipment Type", equipment_options)

with col2:
    pressure = st.number_input("⚙️ Pressure (Bar)", min_value=0.0, max_value=100.0, value=30.0)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)



# Convert equipment to numeric (if needed)
equipment_mapping = {"Turbine": 0, "Compressor": 1, "Pump": 2}
equipment_encoded = equipment_mapping[equipment]

# Predict button
if st.button("predict"):
    # Prepare the input data as a numpy array
    try:
        input_data = np.array([[temperature, pressure, vibration, humidity, equipment_encoded]])
        input_data_scaled=scaler.transform(input_data)
    # Make the prediction
        prediction = model.predict(input_data_scaled)[0]

    # Show result
        if prediction == 1:
            st.error("⚠️ The Machine is FAULTY!")
        else:
            st.success("✅ The Machine is WORKING FINE!")

    except Exception as e:
        st.error(f'Eror is prediction: {e}')

