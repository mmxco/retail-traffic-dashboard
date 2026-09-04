import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ------------------------------------------------------------------------
# Timezone Configuration
# ------------------------------------------------------------------------
LOCAL_TZ = ZoneInfo("America/Denver")

# ------------------------------------------------------------------------
# Data Synthesis Module
# ------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(days=30):
    """
    Generates a synthetic dataset containing hourly weather observations 
    (Fahrenheit, inches/hour, miles per hour) and retail foot traffic.
    """
    now_mst = datetime.now(LOCAL_TZ)
    start_date = now_mst - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=now_mst, freq='h', tz=LOCAL_TZ)
    
    df = pd.DataFrame({'timestamp': date_range})
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    np.random.seed(42)
    
    # Temperature in Fahrenheit (diurnal cycle: ~50°F to 75°F + noise)
    df['temp_f'] = 62.0 + 15.0 * np.sin(np.pi * (df['hour'] - 6) / 12) + np.random.normal(0, 3, len(df))
    
    # Precipitation in inches per hour
    df['precip_in_hr'] = np.where(
        np.random.rand(len(df)) > 0.9, 
        np.random.exponential(0.35, len(df)), 
        0.0
    )
    
    # Wind speed in miles per hour (mean ~8 mph with gusts)
    df['wind_mph'] = np.abs(np.random.normal(8.0, 4.5, len(df)))
    
    # Store operating hours: 9 AM to 9 PM baseline = 50, closed = 0
    baseline_traffic = np.where((df['hour'] >= 9) & (df['hour'] <= 21), 50, 0)
    
    # Weekend volume boost (1.5x)
    weekend_multiplier = np.where(df['day_of_week'] >= 5, 1.5, 1.0)
    
    # Temperature impact: penalize freezing (< 35°F) or extreme heat (> 95°F)
    temp_penalty = np.where(df['temp_f'] < 35.0, 0.7, np.where(df['temp_f'] > 95.0, 0.8, 1.0))
    
    # Rain impact: heavy (> 0.25 in/hr), light (> 0.05 in/hr)
    precip_penalty = np.where(
        df['precip_in_hr'] > 0.25, 0.35,
        np.where(df['precip_in_hr'] > 0.05, 0.75, 1.0)
    )
    
    # Wind impact: high winds (> 20 mph) introduce foot traffic friction
    wind_penalty = np.where(df['wind_mph'] > 20.0, 0.8, 1.0)
    
    # Calculate visitor count
    expected_traffic = baseline_traffic * weekend_multiplier * temp_penalty * precip_penalty * wind_penalty
    df['visitor_count'] = np.random.poisson(expected_traffic)
    
    return df

# ------------------------------------------------------------------------
# Modeling Module
# ------------------------------------------------------------------------
@st.cache_resource
def train_model(df):
    """
    Trains a Random Forest regression pipeline using Imperial weather parameters.
    """
    features = ['hour', 'day_of_week', 'temp_f', 'precip_in_hr', 'wind_mph']
    X = df[features]
    y = df['visitor_count']
    
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

# Current timestamp pinned to Mountain Time
now_mst = datetime.now(LOCAL_TZ)
formatted_now = now_mst.strftime("%A, %B %d, %Y - %I:%M %p %Z")
st.caption(f"Current Store Time: **{formatted_now}**")

# Prepare data and trained estimator
data = generate_synthetic_data()
model = train_model(data)

# ------------------------------------------------------------------------
# Sidebar Forecast Controls
# ------------------------------------------------------------------------
st.sidebar.header("Forecast Settings")

# Time of day selection (defaults to current MST hour)
target_hour = st.sidebar.slider(
    "Target Forecast Hour (0:00 - 23:00)", 
    min_value=0, 
    max_value=23, 
    value=now_mst.hour,
    format="%d:00"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hypothetical Weather Conditions")

# Weather input controls in Imperial units
input_temp_f = st.sidebar.slider("Temperature (°F)", -10.0, 110.0, 68.0, step=1.0)
input_precip_in = st.sidebar.slider("Precipitation (in/hr)", 0.0, 2.0, 0.0, step=0.01, format="%.2f")
input_wind_mph = st.sidebar.slider("Wind Speed (mph)", 0.0, 60.0, 8.0, step=1.0)

# Build feature set for inference
forecast_df = pd.DataFrame({
    'hour': [target_hour],
    'day_of_week': [now_mst.weekday()],
    'temp_f': [input_temp_f],
    'precip_in_hr': [input_precip_in],
    'wind_mph': [input_wind_mph]
})

# Predict visitor count
predicted_count = int(model.predict(forecast_df)[0])

# Classify weather condition
if input_precip_in >= 0.25:
    weather_desc = "Adverse Weather"
elif input_precip_in >= 0.05:
    weather_desc = "Light Rain"
else:
    weather_desc = "Clear"

# Historical average for selected target hour
hist_avg = int(data[data['hour'] == target_hour]['visitor_count'].mean())

# ------------------------------------------------------------------------
# KPI Summary Metrics
# ------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric(f"Predicted Visitors ({target_hour:02d}:00)", f"{predicted_count}")
col2.metric(f"Historical Avg ({target_hour:02d}:00)", f"{hist_avg}")
col3.metric("Weather Condition", weather_desc)

# ------------------------------------------------------------------------
# Historical Trend Chart
# ------------------------------------------------------------------------
st.subheader("Historical Traffic Pattern (Last 7 Days)")
recent_data = data.tail(24 * 7).set_index('timestamp')
st.line_chart(recent_data[['visitor_count']])
