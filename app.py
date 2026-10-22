
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Road Accident Severity Predictor",
    page_icon="🚦",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load("best_accident_severity_model.pkl")


st.title("🚦 Road Accident Severity Predictor")
st.write(
    "Enter road and environmental details to predict "
    "the accident severity class."
)

model = load_model()

with st.form("prediction_form"):
    city = st.selectbox(
        "City",
        ["Bengaluru", "Mumbai", "Delhi", "Chennai",
         "Hyderabad", "Pune", "Kolkata", "Ahmedabad"],
    )
    latitude = st.number_input("Latitude", value=12.97)
    longitude = st.number_input("Longitude", value=77.59)

    day_of_week = st.selectbox(
        "Day of week",
        ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"],
    )
    is_weekend = st.selectbox("Weekend?", [0, 1])
    road_type = st.selectbox(
        "Road type", ["Highway", "Urban", "Rural"]
    )
    lanes = st.number_input("Number of lanes", min_value=1, value=2)
    traffic_signal = st.selectbox("Traffic signal?", [0, 1])
    weather = st.selectbox(
        "Weather", ["Clear", "Rain", "Fog", "Cloudy"]
    )
    visibility = st.selectbox(
        "Visibility", ["low", "medium", "high"]
    )
    temperature = st.number_input(
        "Temperature (°C)", value=28.0
    )
    traffic_density = st.selectbox(
        "Traffic density", ["low", "medium", "high"]
    )
    is_peak_hour = st.selectbox("Peak hour?", [0, 1])
    year = st.number_input("Year", min_value=2022, value=2025)
    month = st.number_input(
        "Month", min_value=1, max_value=12, value=6
    )
    hour = st.slider("Hour of day", 0, 23, 17)

    predict = st.form_submit_button("Predict severity")

if predict:
    input_data = pd.DataFrame([{
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "road_type": road_type,
        "lanes": lanes,
        "traffic_signal": traffic_signal,
        "weather": weather,
        "visibility": visibility,
        "temperature": temperature,
        "traffic_density": traffic_density,
        "is_peak_hour": is_peak_hour,
        "year": year,
        "month": month,
        "hour_sin": __import__("math").sin(
            2 * __import__("math").pi * hour / 24
        ),
        "hour_cos": __import__("math").cos(
            2 * __import__("math").pi * hour / 24
        ),
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.subheader(f"Predicted severity: {prediction}")
        st.warning(
            "Educational prototype only. This prediction is "
            "not a validated road-safety assessment."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.caption(
            "Check that the input column names and category "
            "values match the training dataset."
        )

