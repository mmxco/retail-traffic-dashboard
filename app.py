import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# ------------------------------------------------------------------------
# Data Synthesis Module
# ------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(days=30):
    """
    Generates a synthetic dataset containing hourly weather observations 
    and corresponding retail foot traffic over a specified number of days.
    """
    # Define time index
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Initialize dataframe
    df = pd.DataFrame({'timestamp': date_range})
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Generate synthetic weather variables
    # Temperature: diurnal cycle (cooler at night, warmer in day) + noise
    np.random.seed(42)
    df['temp_2m'] = 285 + 10 * np.sin(np.pi * (df['hour'] - 6) / 12) + np.random.normal(0, 2, len(df))
    
    # Precipitation: mostly zero, occasional spikes
    df['precip_1hr'] = np.where(np.random.rand(len(df)) > 0.9, np.random.exponential(0.01, len(df)), 0)
    
    # Wind speed: baseline + random gusts
    df['wind_10m'] = np.abs(np.random.normal(3, 2, len(df)))
    
    # Generate baseline foot traffic (business hours 9 AM - 9 PM)
    baseline_traffic = np.where((df['hour'] >= 9) & (df['hour'] <= 21), 50, 5)
    
    # Weekend multiplier (traffic increases on Saturdays and Sundays)
    weekend_multiplier = np.where(df['day_of_week'] >= 5, 1.5, 1.0)
    
    # Weather penalty calculations
    # Penalize extreme cold (below 275K) or extreme heat (above 305K)
    temp_penalty = np.where(df['temp_2m'] < 275, 0.7, np.where(df['temp_2m'] > 305, 0.8, 1.0))
    # Penalize traffic heavily for precipitation
    precip_penalty = np.where(df['precip_1hr'] > 0, 0.4, 1.0)
    # Penalize traffic slightly for high winds
    wind_penalty = np.where(df['wind_10m'] > 8, 0.8, 1.0)
    
    # Calculate final synthetic visitor count, add Poisson noise for realism
    expected_traffic = baseline_traffic * weekend_multiplier * temp_penalty * precip_penalty * wind_penalty
    df['visitor_count'] = np.random.poisson(expected_traffic)
    
    return df

# ------------------------------------------------------------------------
# Modeling Module
# ------------------------------------------------------------------------
@st.cache_resource
def train_model(df):
    """
    Trains a Random Forest regression pipeline to predict visitor counts 
    based on temporal and weather features.
    """
    features = ['hour', 'day_of_week', 'temp_2m', 'precip_1hr', 'wind_10m']
    X = df[features]
    y = df['visitor_count']
    
    # Construct regression pipeline with standard scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X, y)
    return pipeline

# ------------------------------------------------------------------------
# Streamlit UI Configuration
# ------------------------------------------------------------------------
st.set_page_config(page_title="Retail Traffic Prediction", layout="wide")
st.title("Near-Term Foot Traffic Prediction Dashboard")
st.markdown("Predictive modeling utilizing synthetic historical baselines and localized weather variables.")

# Generate data and train model execution
data = generate_synthetic_data()
model = train_model(data)

# Layout: Sidebar for forecasting inputs
st.sidebar.header("Forecast Inputs (Next Hour)")
current_hour = datetime.now().hour
current_day = datetime.now().weekday()

# Input sliders for hypothetical WeatherNext 3 variables
input_temp_k = st.sidebar.slider("Temperature (Kelvin)", 260.0, 315.0, 293.0)
input_precip = st.sidebar.slider("Precipitation (m/hr)", 0.0, 0.05, 0.0, format="%.3f")
input_wind = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 3.0)

# Construct feature dataframe for prediction
forecast_df = pd.DataFrame({
    'hour': [current_hour],
    'day_of_week': [current_day],
    'temp_2m': [input_temp_k],
    'precip_1hr': [input_precip],
    'wind_10m': [input_wind]
})

# Execute prediction
prediction = model.predict(forecast_df)[0]

# Display Key Performance Indicators
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Visitors (Next Hour)", f"{int(prediction)}")
col2.metric("Historical Avg (This Hour)", f"{int(data[data['hour'] == current_hour]['visitor_count'].mean())}")
col3.metric("Current Weather Condition", "Adverse" if input_precip > 0.005 else "Clear")

# Visualize synthetic historical data
st.subheader("Historical Traffic vs Temperature Baseline (Last 7 Days)")
recent_data = data.tail(24 * 7).set_index('timestamp')
st.line_chart(recent_data[['visitor_count']])