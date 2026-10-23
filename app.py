
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Road Accident Severity Predictor",
    page_icon="🚦",
    layout="centered",
)

MODEL_PATH = Path(__file__).with_name(
    "best_accident_severity_model_reduced.pkl"
)


@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


st.title("🚦 Road Accident Severity Predictor")
st.write(
    "Enter road and environmental details to predict "
    "the accident severity class."
)

st.caption(
    "Educational prototype only. Not validated for real-world "
    "road-safety decisions."
)

if not MODEL_PATH.exists():
    st.error("The reduced model file was not found.")
    st.stop()

model = load_model(
    str(MODEL_PATH),
    MODEL_PATH.stat().st_mtime,
)

st.write("Model classes:", list(model.classes_))

with st.form("prediction_form"):
    city = st.selectbox(
        "City",
        [
            "Bengaluru", "Mumbai", "Delhi", "Chennai",
            "Hyderabad", "Pune", "Kolkata", "Ahmedabad",
        ],
    )
    day_of_week = st.selectbox(
        "Day of week",
        [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ],
    )
    is_weekend = st.selectbox("Weekend?", [0, 1])
    road_type = st.selectbox(
        "Road type", ["highway", "urban", "rural"]
    )
    lanes = st.number_input(
        "Number of lanes", min_value=1, value=2
    )
    traffic_signal = st.selectbox("Traffic signal?", [0, 1])
    weather = st.selectbox(
        "Weather", ["clear", "rain", "fog"]
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
    year = st.number_input(
        "Year", min_value=2000, max_value=2100, value=2025
    )
    month = st.number_input(
        "Month", min_value=1, max_value=12, value=6
    )
    hour = st.slider("Hour of day", 0, 23, 17)

    submitted = st.form_submit_button("Predict severity")

if submitted:
    input_data = pd.DataFrame([{
        "city": city,
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
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.subheader(
            f"Predicted severity: {str(prediction).title()}"
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            scores = pd.DataFrame({
                "Severity": model.classes_,
                "Model score (%)": probabilities * 100,
            })

            st.write("Scores for all severity classes")
            st.dataframe(
                scores.round(2),
                hide_index=True,
                use_container_width=True,
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

    st.warning(
        "This is an educational prototype, not a validated "
        "road-safety assessment."
    )
