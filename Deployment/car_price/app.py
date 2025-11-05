import streamlit as st
import pickle
import numpy as np

# Load trained model and encoders
model = pickle.load(open('model.pkl', 'rb'))
Fuel_Type_en = pickle.load(open('Fuel_Type.pkl', 'rb'))
Transmission_en = pickle.load(open('Transmission.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# --- Page Config ---
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f4f7fa;
        }
        .main-title {
            text-align: center;
            background: linear-gradient(90deg, #007bff, #00c6ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 1.1em;
            margin-bottom: 30px;
        }
        .card {
            background-color: white;
            padding: 30px 40px;
            border-radius: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }
        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #007bff, #00c6ff);
            color: white;
            border: none;
            border-radius: 10px;
            height: 3em;
            font-size: 1.1em;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #00c6ff, #007bff);
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 class='main-title'>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter the car details below to estimate its selling price</p>", unsafe_allow_html=True)

# --- Input Card ---
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        Car_Age = st.text_input("🔢 Car Age (years)", "5")
        Present_Price = st.text_input("💰 Present Price (in lakhs)", "5.0")
        Kms_Driven = st.text_input("📏 Kilometers Driven", "30000")

    with col2:
        Fuel_Type = st.selectbox("⛽ Fuel Type", ("Petrol", "Diesel", "CNG"))
        Transmission = st.selectbox("⚙️ Transmission", ("Manual", "Automatic"))
        Owner = st.text_input("👤 Number of Previous Owners", "0")
        Seller_Type_Individual = st.selectbox("🏪 Seller Type", ("Dealer", "Individual"))

    st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction Logic ---
try:
    Car_Age = int(Car_Age)
    Present_Price = float(Present_Price)
    Kms_Driven = float(Kms_Driven)
    Owner = int(Owner)

    # Encode categorical variables
    Fuel_Type_val = Fuel_Type_en.transform([Fuel_Type])[0]
    Transmission_val = Transmission_en.transform([Transmission])[0]
    Seller_Type_Individual_val = 1 if Seller_Type_Individual == "Individual" else 0

    # Prepare input data
    details = [Car_Age, Present_Price, Kms_Driven, Fuel_Type_val, Transmission_val, Owner, Seller_Type_Individual_val]
    data_out = np.array(details).reshape(1, -1)
    data_scaled = scaler.transform(data_out)

    # --- Predict Button ---
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Predict Car Price"):
        prediction = model.predict(data_scaled)[0]
        st.success(f"💵 Estimated Car Price: **₹ {round(prediction, 2)} Lakhs**")

except ValueError:
    st.warning("⚠️ Please enter valid numeric values for all required fields.")



    