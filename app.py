
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


@st.cache_resource
def load_model():
    model_path = Path(__file__).with_name(
        "best_accident_severity_model_reduced.pkl"
    )
    return joblib.load(model_path)


st.title("🚦 Road Accident Severity Predictor")
st.write(
    "Enter road and environmental details to predict "
    "the accident severity class."
)

st.info(
    "This is an educational prototype, not a validated "
    "real-world road safety assessment."
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


with st.form("prediction_form"):
    city = st.selectbox(
        "City",
        [
            "Bengaluru",
            "Mumbai",
            "Delhi",
            "Chennai",
            "Hyderabad",
            "Pune",
            "Kolkata",
            "Ahmedabad",
        ],
    )

    day_of_week = st.selectbox(
        "Day of week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
    )

    is_weekend = st.selectbox(
        "Weekend?",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No",
    )

    road_type = st.selectbox(
        "Road type",
        ["Highway", "Urban", "Rural"],
    )

    lanes = st.number_input(
        "Number of lanes",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

    traffic_signal = st.selectbox(
        "Traffic signal?",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No",
    )

    weather = st.selectbox(
        "Weather",
        ["Clear", "Rain", "Fog", "Cloudy"],
    )

    visibility = st.selectbox(
        "Visibility",
        ["low", "medium", "high"],
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=60.0,
        value=28.0,
        step=1.0,
    )

    traffic_density = st.selectbox(
        "Traffic density",
        ["low", "medium", "high"],
    )

    is_peak_hour = st.selectbox(
        "Peak hour?",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No",
    )

    year = st.number_input(
        "Year",
        min_value=2000,
        max_value=2100,
        value=2025,
        step=1,
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
    )

    hour = st.slider(
        "Hour of day",
        min_value=0,
        max_value=23,
        value=17,
        help="0 = midnight, 12 = noon, 18 = 6 PM.",
    )

    predict = st.form_submit_button(
        "Predict severity",
        type="primary",
    )


if predict:
    input_data = pd.DataFrame(
        [
            {
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
            }
        ]
    )

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction result")
        st.write(f"**Predicted severity: {prediction}**")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            probability_data = pd.DataFrame(
                {
                    "Severity": [str(label) for label in classes],
                    "Probability (%)": [
                        round(float(prob) * 100, 2)
                        for prob in probabilities
                    ],
                }
            )

            st.subheader("Class probabilities")
            st.dataframe(
                probability_data,
                hide_index=True,
                use_container_width=True,
            )

            st.bar_chart(
                probability_data.set_index("Severity")
            )

            st.caption(
                "Probabilities show the model's estimates, "
                "not the certainty or actual risk of an accident."
            )
        else:
            st.warning(
                "This model does not support predict_proba, "
                "so probability scores are unavailable."
            )

        with st.expander("Inspect model input"):
            st.dataframe(input_data, use_container_width=True)

        st.warning(
            "Educational demonstration only. Do not use "
            "these predictions for real-world safety decisions."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.caption(
            "The model's expected input features or category "
            "values may differ from the app."
        )
