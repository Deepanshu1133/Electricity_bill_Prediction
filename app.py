import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Smart Electricity Bill Predictor",
    page_icon="⚡",
    layout="centered"
)

# ---------------------------------------------------
# LOAD MODEL AND SCALER
# ---------------------------------------------------

model = joblib.load("electricity_bill_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("⚡ Smart Electricity Bill Predictor")

st.write(
    "Enter the number of appliances and their average daily usage."
)

st.info(
    "The app calculates total appliance usage as: "
    "Number of Appliances × Hours Used Per Day"
)

# ---------------------------------------------------
# FANS
# ---------------------------------------------------

st.subheader("🌀 Fans")

number_of_fans = st.number_input(
    "Number of Fans",
    min_value=0,
    max_value=20,
    value=3,
    step=1
)

fan_hours = st.number_input(
    "Average Fan Usage (hours/day per fan)",
    min_value=0.0,
    max_value=24.0,
    value=8.0,
    step=1.0
)

total_fan_usage = number_of_fans * fan_hours

st.write(f"Total Fan Usage: **{total_fan_usage:.1f} hours/day**")


# ---------------------------------------------------
# REFRIGERATOR
# ---------------------------------------------------

st.subheader("❄️ Refrigerator")

number_of_refrigerators = st.number_input(
    "Number of Refrigerators",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

refrigerator_hours = st.number_input(
    "Average Refrigerator Usage (hours/day per refrigerator)",
    min_value=0.0,
    max_value=24.0,
    value=20.0,
    step=1.0
)

total_refrigerator_usage = (
    number_of_refrigerators * refrigerator_hours
)

st.write(
    f"Total Refrigerator Usage: "
    f"**{total_refrigerator_usage:.1f} hours/day**"
)


# ---------------------------------------------------
# AIR CONDITIONER
# ---------------------------------------------------

st.subheader("❄️ Air Conditioner")

number_of_ac = st.number_input(
    "Number of Air Conditioners",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

ac_hours = st.number_input(
    "Average AC Usage (hours/day per AC)",
    min_value=0.0,
    max_value=24.0,
    value=2.0,
    step=1.0
)

total_ac_usage = number_of_ac * ac_hours

st.write(
    f"Total AC Usage: **{total_ac_usage:.1f} hours/day**"
)


# ---------------------------------------------------
# TELEVISION
# ---------------------------------------------------

st.subheader("📺 Television")

number_of_tvs = st.number_input(
    "Number of Televisions",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

tv_hours = st.number_input(
    "Average TV Usage (hours/day per TV)",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=1.0
)

total_tv_usage = number_of_tvs * tv_hours

st.write(
    f"Total TV Usage: **{total_tv_usage:.1f} hours/day**"
)


# ---------------------------------------------------
# MONITOR
# ---------------------------------------------------

st.subheader("🖥️ Monitor")

number_of_monitors = st.number_input(
    "Number of Monitors",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

monitor_hours = st.number_input(
    "Average Monitor Usage (hours/day per monitor)",
    min_value=0.0,
    max_value=24.0,
    value=4.0,
    step=1.0
)

total_monitor_usage = number_of_monitors * monitor_hours

st.write(
    f"Total Monitor Usage: **{total_monitor_usage:.1f} hours/day**"
)


# ---------------------------------------------------
# BILLING INFORMATION
# ---------------------------------------------------

st.subheader("📅 Billing Information")

days_in_month = st.number_input(
    "Days in Month",
    min_value=28,
    max_value=31,
    value=30,
    step=1
)

tariff_rate = st.number_input(
    "Electricity Tariff Rate (₹/unit)",
    min_value=5.5,
    max_value=9.3,
    value=8.4,
    step=0.1
)


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("🔮 Predict Electricity Bill"):

    input_data = pd.DataFrame({
        "Fan": [total_fan_usage],
        "Refrigerator": [total_refrigerator_usage],
        "AirConditioner": [total_ac_usage],
        "Television": [total_tv_usage],
        "Monitor": [total_monitor_usage],
        "DaysInMonth": [days_in_month],
        "TariffRate": [tariff_rate]
    })

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Prevent negative bill
    prediction = max(0, prediction)

    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------

    st.success(
        f"💰 Estimated Monthly Electricity Bill: "
        f"₹{prediction:,.2f}"
    )

    # ------------------------------------------------
    # SUMMARY
    # ------------------------------------------------

    st.subheader("📋 Usage Summary")

    summary = pd.DataFrame({
        "Appliance": [
            "Fans",
            "Refrigerators",
            "Air Conditioners",
            "Televisions",
            "Monitors"
        ],
        "Number": [
            number_of_fans,
            number_of_refrigerators,
            number_of_ac,
            number_of_tvs,
            number_of_monitors
        ],
        "Hours/Day": [
            fan_hours,
            refrigerator_hours,
            ac_hours,
            tv_hours,
            monitor_hours
        ],
        "Total Usage": [
            total_fan_usage,
            total_refrigerator_usage,
            total_ac_usage,
            total_tv_usage,
            total_monitor_usage
        ]
    })

    st.dataframe(summary, use_container_width=True)