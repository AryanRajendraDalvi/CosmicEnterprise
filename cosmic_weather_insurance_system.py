import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from pathlib import Path

# --- 1. CONFIGURATION & SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="Cosmic Weather Insurance System v2", layout="wide")

# --- Constants ---
NOAA_API_BASE_URL = "https://services.swpc.noaa.gov"
FORECAST_HORIZON_HOURS = [24, 48, 72]
MONTE_CARLO_SIMULATIONS = 1000
LOADING_FACTOR = 1.25
HISTORICAL_DATA_FILE = 'noaa_indices.csv'

# --- 2. AGENT TOOLS (MODULAR FUNCTIONS) ---

# Tool 1: Data Ingestion
@st.cache_data(ttl=900) # Cache for 15 minutes
def ingest_all_data():
    """
    Agent Tool: Ingests all required data sources (historical, live Kp, live solar wind).
    """
    logging.info("Executing Data Ingestion Tool...")

    # --- a) Load Historical Kp Data (Optional) ---
    try:
        hist_df = pd.read_csv(HISTORICAL_DATA_FILE)
        hist_df['datetime'] = pd.to_datetime(hist_df['datetime'])
        hist_df.set_index('datetime', inplace=True)
        logging.info(f"Loaded {len(hist_df)} historical records.")
    except FileNotFoundError:
        hist_df = pd.DataFrame()
        logging.warning(f"'{HISTORICAL_DATA_FILE}' not found. Proceeding without historical data.")
        st.warning("Historical data file not found. Forecast will be based on recent data only.")

    # --- b) Fetch Live Planetary K-index ---
    try:
        kp_url = f"{NOAA_API_BASE_URL}/json/planetary_k_index_1m.json"
        kp_data = requests.get(kp_url).json()
        kp_df = pd.DataFrame(kp_data)
        kp_df['time_tag'] = pd.to_datetime(kp_df['time_tag'])
        kp_df.set_index('time_tag', inplace=True)
        kp_df = kp_df[['estimated_kp']].rename(columns={'estimated_kp': 'Kp'})
    except Exception as e:
        logging.error(f"Failed to fetch live Kp-index: {e}")
        kp_df = pd.DataFrame()

    # --- c) Fetch Live Solar Wind Data (DSCOVR) ---
    try:
        sw_url = f"{NOAA_API_BASE_URL}/json/dscovr/realtime.json"
        sw_data = requests.get(sw_url).json()
        sw_df = pd.DataFrame(sw_data)
        sw_df['time_tag'] = pd.to_datetime(sw_df['time_tag'])
        sw_df.set_index('time_tag', inplace=True)
        sw_df = sw_df[['speed', 'bz_gsm']].astype(float)
        sw_df.rename(columns={'speed': 'solar_wind_speed', 'bz_gsm': 'imf_bz'}, inplace=True)
    except Exception as e:
        logging.error(f"Failed to fetch live solar wind data: {e}")
        sw_df = pd.DataFrame()

    # --- d) Combine all data ---
    combined_df = pd.concat([hist_df, kp_df, sw_df], axis=1)
    combined_df = combined_df.resample('H').mean() # Resample to a consistent hourly frequency
    combined_df.ffill(inplace=True) # Forward-fill missing values
    combined_df.dropna(inplace=True) # Drop any remaining NaNs

    logging.info(f"Data ingestion complete. Combined data shape: {combined_df.shape}")
    return combined_df.reset_index().rename(columns={'index': 'datetime'})

# Tool 2: Feature Engineering & Forecasting
@st.cache_data
def generate_severity_forecast(_data: pd.DataFrame):
    """
    Agent Tool: Creates lagged features and uses a CatBoost model to forecast storm severity probabilities.
    """
    logging.info("Executing Forecasting Tool...")
    if _data.empty:
        logging.error("Cannot run forecast: Input data is empty.")
        return None

    # --- a) Feature Engineering ---
    df = _data.copy()
    df.set_index('datetime', inplace=True)

    # Create lagged features for the last 24 hours
    for lag in [1, 3, 6, 12, 24]:
        df[f'Kp_lag_{lag}h'] = df['Kp'].shift(lag)
        if 'solar_wind_speed' in df.columns:
            df[f'speed_lag_{lag}h'] = df['solar_wind_speed'].shift(lag)
        if 'imf_bz' in df.columns:
            df[f'bz_lag_{lag}h'] = df['imf_bz'].shift(lag)

    # Create a synthetic target variable based on Kp index for demonstration
    # 0: Quiet, 1: Moderate, 2: Strong, 3: Severe
    df['severity'] = pd.cut(df['Kp'], bins=[-1, 4, 5, 6, 10], labels=[0, 1, 2, 3], right=False)
    df.dropna(inplace=True)

    if len(df) < 50:
        st.error("Not enough recent data to generate a reliable forecast. Please try again later.")
        logging.error("Not enough data for training after feature engineering.")
        return None

    # --- b) Train Multi-Output Forecasting Model ---
    features = [col for col in df.columns if col != 'severity' and 'datetime' not in col]
    X = df[features]
    y = df['severity']

    model = CatBoostClassifier(iterations=200, verbose=0, loss_function='MultiClass', random_seed=42)
    model.fit(X, y)

    # --- c) Predict for Future Horizons ---
    # We predict using the most recent data point as the basis for the forecast
    last_known_data = X.iloc[[-1]]

    predictions = {}
    for horizon in FORECAST_HORIZON_HOURS:
        # In a real system, you'd use a more sophisticated model for each horizon.
        # Here, we simulate by using the same model for simplicity.
        probs = model.predict_proba(last_known_data)
        predictions[f'{horizon}h'] = {
            'Quiet': probs[0][0],
            'Moderate': probs[0][1],
            'Strong': probs[0][2],
            'Severe': probs[0][3]
        }

    logging.info("Forecasting tool complete.")
    return predictions

# Tool 3: Risk & Impact Modeling
def assess_asset_impact(predictions, asset_value, orbit, shielding):
    """
    Agent Tool: Runs a Monte Carlo simulation based on forecasted severity probabilities.
    """
    logging.info("Executing Risk Assessment Tool...")
    if not predictions:
        return np.array([0]), 0

    # Define anomaly probability based on storm severity and asset characteristics
    # These are empirical values for demonstration
    severity_weights = {'Quiet': 0.0, 'Moderate': 0.001, 'Strong': 0.01, 'Severe': 0.05}
    risk_factor = get_risk_factors(orbit, shielding)

    # Calculate the overall probability of anomaly for each forecast horizon
    horizon_anomaly_prob = {}
    for horizon, probs in predictions.items():
        base_prob = sum(severity_weights[severity] * prob for severity, prob in probs.items())
        horizon_anomaly_prob[horizon] = np.clip(base_prob * risk_factor, 0, 1)

    # Take the maximum probability across the 72h window as the event risk
    max_anomaly_prob = max(horizon_anomaly_prob.values())

    # Run Monte Carlo simulation
    simulated_outcomes = np.random.rand(MONTE_CARLO_SIMULATIONS)
    loss_distribution = np.where(simulated_outcomes < max_anomaly_prob, asset_value, 0)
    expected_loss = np.mean(loss_distribution)

    logging.info(f"Risk assessment complete. Expected Loss: ${expected_loss:,.2f}")
    return loss_distribution, expected_loss

# Tool 4: Premium Calculation
def calculate_final_premium(expected_loss):
    """
    Agent Tool: Calculates the final insurance premium with a loading factor.
    """
    logging.info("Executing Premium Calculation Tool...")
    premium = expected_loss * LOADING_FACTOR
    assumptions = {
        "Loading Factor": f"{LOADING_FACTOR} (covers profit, admin costs, and uncertainty)",
        "Loss Model": f"Total asset loss based on the max anomaly probability over the next 72 hours.",
        "Forecast Basis": "CatBoost model predicting probabilities of G0-G5 storm severities.",
        "Risk Factors": "Empirical multipliers for satellite orbit and shielding levels."
    }
    logging.info(f"Premium calculation complete. Premium: ${premium:,.2f}")
    return premium, assumptions

# --- Helper functions (used by tools) ---
def get_risk_factors(orbit, shielding):
    """Returns risk multipliers based on satellite characteristics."""
    factors = {'orbit': {'LEO': 1.0, 'MEO': 1.2, 'GEO': 1.5, 'HEO': 1.8}, 'shielding': {'Low': 1.5, 'Medium': 1.0, 'High': 0.7}}
    return factors['orbit'].get(orbit, 1.0) * factors['shielding'].get(shielding, 1.0)

def plot_forecast_probabilities(predictions):
    """Generates a bar chart of forecast probabilities."""
    df = pd.DataFrame(predictions).T
    df.index.name = 'Forecast Horizon'
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', ax=ax, colormap='viridis')
    ax.set_title('Forecasted Storm Severity Probabilities', fontsize=16)
    ax.set_xlabel('Forecast Horizon (Hours)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    return fig

# --- 5. AGENTIC WORKFLOW & UI ---

def run_agentic_workflow(asset_value, orbit, shielding):
    """Orchestrates the agentic workflow by calling tools in sequence."""
    st.session_state.clear()

    # Step 1: Ingest Data
    with st.spinner('Agent at work... [Step 1/4: Ingesting Data]'):
        data = ingest_all_data()
        if data.empty or len(data) < 50:
            st.error("Agent failed: Not enough data to create a reliable forecast.")
            return
        st.success("Agent action complete: [Data Ingested]")
        time.sleep(1)

    # Step 2: Generate Forecast
    with st.spinner('Agent at work... [Step 2/4: Generating Forecast]'):
        predictions = generate_severity_forecast(data)
        if predictions is None:
            st.error("Agent failed: Could not generate forecast.")
            return
        st.session_state.predictions = predictions
        st.session_state.forecast_fig = plot_forecast_probabilities(predictions)
        st.success("Agent action complete: [Forecast Generated]")
        time.sleep(1)

    # Step 3: Assess Impact & Loss
    with st.spinner('Agent at work... [Step 3/4: Assessing Risk]'):
        loss_dist, expected_loss = assess_asset_impact(predictions, asset_value, orbit, shielding)
        st.session_state.loss_dist = loss_dist
        st.session_state.expected_loss = expected_loss
        st.success("Agent action complete: [Risk Assessed]")
        time.sleep(1)

    # Step 4: Calculate Premium
    with st.spinner('Agent at work... [Step 4/4: Calculating Premium]'):
        premium, assumptions = calculate_final_premium(expected_loss)
        st.session_state.premium = premium
        st.session_state.assumptions = assumptions
        st.success("Agent action complete: [Premium Calculated]")
        time.sleep(1)

    st.session_state.workflow_complete = True

# --- UI Layout ---
st.title("🛰️ Cosmic Weather Insurance System (v2)")
st.markdown("An agent-driven system to price insurance for space weather events using multi-factor forecasting.")

st.sidebar.header("Asset Details")
with st.sidebar.form(key='asset_form'):
    asset_value = st.number_input("Asset Value ($)", min_value=1_000_000, max_value=1_000_000_000, value=50_000_000, step=1_000_000, format="%d")
    orbit_type = st.selectbox("Orbit Type", ['LEO', 'MEO', 'GEO', 'HEO'], index=2)
    shielding_level = st.selectbox("Shielding Level", ['Low', 'Medium', 'High'], index=1)
    submit_button = st.form_submit_button(label='Calculate Premium')

if submit_button:
    run_agentic_workflow(asset_value, orbit_type, shielding_level)

if 'workflow_complete' in st.session_state and st.session_state.workflow_complete:
    st.header("Insurance Premium Recommendation")

    col1, col2 = st.columns(2)
    col1.metric("Suggested Premium (72h)", f"${st.session_state.premium:,.2f}")
    col2.metric("Expected Loss", f"${st.session_state.expected_loss:,.2f}")

    prob_of_loss = np.mean(st.session_state.loss_dist > 0)
    st.info(f"**Calculated Probability of Loss in Next 72 Hours:** {prob_of_loss:.2%}")

    with st.expander("View Calculation Assumptions", expanded=False):
        for key, value in st.session_state.assumptions.items():
            st.markdown(f"- **{key}:** {value}")

    st.header("Agent Analysis Results")
    tab1, tab2 = st.tabs(["Severity Forecast", "Loss Distribution"])

    with tab1:
        st.pyplot(st.session_state.forecast_fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(st.session_state.loss_dist, bins=50, ax=ax, kde=False)
        ax.set_title("Distribution of Potential Losses (Monte Carlo Simulation)")
        ax.set_xlabel("Loss Amount ($)")
        ax.set_ylabel("Frequency")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1_000_000:.1f}M'))
        st.pyplot(fig)
else:
    st.info("Enter asset details in the sidebar and click 'Calculate Premium' to activate the agent.")