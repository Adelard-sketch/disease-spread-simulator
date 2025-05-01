import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle
import os

st.set_page_config(page_title="COVID-19 Risk & Spread Simulator", layout="wide")

# Title
st.title("COVID-19 Risk & Spread Simulator")
st.markdown("A tool for both **patient-level prediction** and **population-level simulation**.")

# Load patient-level prediction model
@st.cache_resource
def load_patient_model():
    model_path = 'covid_prediction_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model file 'covid_prediction_model.pkl' not found.")
        return None

# Load forest model for infection risk 
def load_forest_model():
    try:
        with open("covid_prediction_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Trained model file 'covid_prediction_model.pkl' not found.")
        return None

model = load_patient_model()
forest_model = load_forest_model()

# --- Layout with tabs ---
tab1, tab2 = st.tabs(["🧪 Patient Risk Prediction", "📈 Virus Spread Simulation"])

# --- Tab 1: Patient Risk Prediction ---
with tab1:
    st.header("Individual COVID-19 Risk Assessment")
    st.write("Enter patient symptoms and details to predict likelihood of a positive result.")

    with st.form("patient_form"):
        cough = st.selectbox("Cough", ["No", "Yes"])
        fever = st.selectbox("Fever", ["No", "Yes"])
        sore_throat = st.selectbox("Sore Throat", ["No", "Yes"])
        shortness_of_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age_60_and_above = st.selectbox("Is age 60 or above?", ["No", "Yes"])
        test_indication = st.selectbox("Test Indication", ["Contact with confirmed", "Other", "Abroad"])
        submitted = st.form_submit_button("Predict")

    if submitted and model is not None:
        patient = {
            'cough': 1 if cough == "Yes" else 0,
            'fever': 1 if fever == "Yes" else 0,
            'sore_throat': 1 if sore_throat == "Yes" else 0,
            'shortness_of_breath': 1 if shortness_of_breath == "Yes" else 0,
            'gender': 1 if gender == "Male" else 0,
            'age_60_and_above': 1 if age_60_and_above == "Yes" else 0,
            'test_indication_Abroad': 0,
            'test_indication_Contact with confirmed': 0,
            'test_indication_Other': 0
        }

        patient[f'test_indication_{test_indication}'] = 1

        sample_df = pd.DataFrame([patient])
        required_cols = model.feature_names_in_
        for col in required_cols:
            if col not in sample_df.columns:
                sample_df[col] = 0
        sample_df = sample_df[required_cols]

        prediction = model.predict(sample_df)[0]
        prob = model.predict_proba(sample_df)[0, 1]

        st.subheader("Prediction Result:")
        st.write(f"Prediction: **{'Positive' if prediction == 1 else 'Negative'}**")
        st.write(f"Probability of being COVID Positive: **{prob:.2%}**")

# --- Tab 2: Virus Spread Simulation ---
with tab2:
    st.header("Virus Spread Simulation (SIR Model)")

    st.subheader("Simulation Parameters")
    population = st.slider("Population Size", 100, 10000, 1000)
    initial_infected = st.slider("Initial Infected Individuals", 1, 100, 10)
    transmission_rate = st.slider("Transmission Rate (β)", 0.0, 1.0, 0.3)
    recovery_rate = st.slider("Recovery Rate (γ)", 0.0, 1.0, 0.1)
    simulation_days = st.slider("Simulation Days", 10, 200, 100)
    quarantine_factor = st.slider("Quarantine Strength (0 = None, 1 = Full)", 0.0, 1.0, 0.0)
    vaccination_rate = st.slider("Vaccination Rate (per day)", 0.0, 0.05, 0.0)

    def run_sir_simulation(population, initial_infected, beta, gamma, days, quarantine, vaccination):
        S = population - initial_infected
        I = initial_infected
        R = 0
        effective_beta = beta * (1 - quarantine)
        S_list, I_list, R_list = [S], [I], [R]

        for _ in range(days):
            new_infections = effective_beta * S * I / population
            new_recoveries = gamma * I
            new_vaccinations = vaccination * S

            S = S - new_infections - new_vaccinations
            I = I + new_infections - new_recoveries
            R = R + new_recoveries + new_vaccinations

            S, I, R = max(0, S), max(0, I), max(0, R)

            S_list.append(S)
            I_list.append(I)
            R_list.append(R)

        return S_list, I_list, R_list

    S_list, I_list, R_list = run_sir_simulation(
        population, initial_infected, transmission_rate, recovery_rate, simulation_days,
        quarantine_factor, vaccination_rate
    )

    R0 = transmission_rate / recovery_rate if recovery_rate != 0 else float("inf")
    peak_infection_time = np.argmax(I_list)
    total_infections = max(population - S_list[-1], 0)

    st.subheader("Simulation Results")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S_list, label="Susceptible", color="blue")
    ax.plot(I_list, label="Infected", color="red")
    ax.plot(R_list, label="Recovered", color="green")
    ax.set_xlabel("Days")
    ax.set_ylabel("Population")
    ax.set_title("SIR Model Simulation")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Epidemiological Metrics")
    st.write(f"**Basic Reproduction Number (R₀):** {R0:.2f}")
    st.write(f"**Peak Infection Time:** Day {peak_infection_time}")
    st.write(f"**Estimated Total Infections:** {int(total_infections)}")

    st.header("Ethical Considerations")
    st.markdown("""
    - **Privacy**: This tool should only use anonymized or synthetic data.
    - **Model Limitations**: Simplified assumptions; not for medical use.
    - **Educational Use**: Intended for understanding, not diagnosis or official predictions.
    """)

