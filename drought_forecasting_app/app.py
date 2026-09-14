import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load model and scaler
model = joblib.load("best_drought_model.pkl")
scaler = joblib.load("scaler.pkl")


# Page config
st.set_page_config(
    page_title="Drought Forecasting System",
    layout="centered"
)


# Title
st.title("AI-Based Meteorological Drought Forecasting System")

st.markdown("""
Machine Learning and Deep Learning Approaches for Meteorological Drought Forecasting Using Multivariate Climate Time-Series Data:
Case Study of Bahir Dar
""")


# Drought classification
def classify_drought(spi):

    if spi <= -2:
        return "Extreme Drought"

    elif spi <= -1.5:
        return "Severe Drought"

    elif spi <= -1:
        return "Moderate Drought"

    elif spi < 0:
        return "Mild Drought"

    elif spi < 1:
        return "Normal"

    else:
        return "Wet"


# Sidebar mode selection
mode = st.sidebar.selectbox(
    "Select Forecast Mode",
    [
        "Simple Forecast (1-6 Months)",
        "Advanced Forecast (6+ Months)"
    ]
)


# -----------------------------
# SIMPLE MODE
# -----------------------------

if mode == "Simple Forecast (1-6 Months)":


    st.header("Simple Forecast")


    target_month = st.selectbox(
        "Select Future Month",
        list(range(1, 13))
    )

    target_year = st.number_input(
        "Select Future Year",
        min_value=2025,
        max_value=2100,
        value=2026
    )


    st.markdown("""
    This mode predicts future drought conditions automatically
    using recent climate patterns learned by the AI model.
    """)


    if st.button("Predict Future Drought"):


        # Default recent conditions
        feature_values = pd.DataFrame([[
            70,     # RELHUM
            26,     # TMPMAX
            14,     # TMPMIN

            -0.2,   # SPI_lag_1
            -0.1,   # SPI_lag_2
            0.0,    # SPI_lag_3

            120,    # RAIN_lag_1
            100,    # RAIN_lag_2
            90,     # RAIN_lag_3

            103,    # rolling_mean_3
            15,     # rolling_std_3

            target_month
        ]], columns=[

            "RELHUM",
            "TMPMAX",
            "TMPMIN",

            "SPI_lag_1",
            "SPI_lag_2",
            "SPI_lag_3",

            "RAIN_lag_1",
            "RAIN_lag_2",
            "RAIN_lag_3",

            "rolling_mean_3",
            "rolling_std_3",

            "MONTH"
        ])


        # Scale
        scaled_features = scaler.transform(
            feature_values
        )


        # Predict
        predicted_spi = model.predict(
            scaled_features
        )[0]


        # Classify
        drought_class = classify_drought(
            predicted_spi
        )


        # Results
        st.subheader("Forecast Result")

        st.write(
            f"Forecast Date: {target_month}/{target_year}"
        )

        st.write(
            f"Predicted SPI-3: {round(predicted_spi, 3)}"
        )

        st.write(
            f"Drought Condition: {drought_class}"
        )


# -----------------------------
# ADVANCED MODE
# -----------------------------

else:


    st.header("Advanced Forecast")


    st.markdown("""
    This mode allows long-range forecasting
    using custom climate conditions.
    """)


    target_month = st.selectbox(
        "Future Month",
        list(range(1, 13))
    )

    target_year = st.number_input(
        "Future Year",
        min_value=2025,
        max_value=2100,
        value=2030
    )


    # Climate inputs
    relhum = st.slider(
        "Relative Humidity",
        0.0,
        100.0,
        70.0
    )

    tmpmax = st.slider(
        "Maximum Temperature",
        10.0,
        40.0,
        26.0
    )

    tmpmin = st.slider(
        "Minimum Temperature",
        0.0,
        30.0,
        14.0
    )

    rain1 = st.number_input(
        "Last Month Rainfall",
        value=120.0
    )

    rain2 = st.number_input(
        "2 Months Ago Rainfall",
        value=100.0
    )

    rain3 = st.number_input(
        "3 Months Ago Rainfall",
        value=90.0
    )

    spi1 = st.number_input(
        "Last SPI",
        value=-0.2
    )

    spi2 = st.number_input(
        "2 Months Ago SPI",
        value=-0.1
    )

    spi3 = st.number_input(
        "3 Months Ago SPI",
        value=0.0
    )


    if st.button("Run Advanced Forecast"):


        rolling_mean = np.mean([
            rain1,
            rain2,
            rain3
        ])

        rolling_std = np.std([
            rain1,
            rain2,
            rain3
        ])


        advanced_features = pd.DataFrame([[
            relhum,
            tmpmax,
            tmpmin,

            spi1,
            spi2,
            spi3,

            rain1,
            rain2,
            rain3,

            rolling_mean,
            rolling_std,

            target_month
        ]], columns=[

            "RELHUM",
            "TMPMAX",
            "TMPMIN",

            "SPI_lag_1",
            "SPI_lag_2",
            "SPI_lag_3",

            "RAIN_lag_1",
            "RAIN_lag_2",
            "RAIN_lag_3",

            "rolling_mean_3",
            "rolling_std_3",

            "MONTH"
        ])


        # Scale
        scaled_features = scaler.transform(
            advanced_features
        )


        # Predict
        predicted_spi = model.predict(
            scaled_features
        )[0]


        # Classification
        drought_class = classify_drought(
            predicted_spi
        )


        # Output
        st.subheader("Advanced Forecast Result")

        st.write(
            f"Forecast Date: {target_month}/{target_year}"
        )

        st.write(
            f"Predicted SPI-3: {round(predicted_spi, 3)}"
        )

        st.write(
            f"Drought Condition: {drought_class}"
        )