import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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
    (Fahrenheit, inches/hour, mph) and corresponding retail foot traffic.
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
    
    # Wind impact: high winds (> 20 mph) introduce friction
    wind_penalty = np.where(df['wind_mph'] > 20.0, 0.8, 1.0)
    
    # Calculate visitor count with Poisson distribution
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
st.title("Near-Term Foot Traffic & Weather Analysis")

# Store timestamp in MST/MDT
now_mst = datetime.now(LOCAL_TZ)
formatted_now = now_mst.strftime("%A, %B %d, %Y - %I:%M %p %Z")
st.caption(f"Current Store Time: **{formatted_now}**")

# Data preparation and model training
data = generate_synthetic_data()
model = train_model(data)

# Generate predictions across the historical set for comparison
features = ['hour', 'day_of_week', 'temp_f', 'precip_in_hr', 'wind_mph']
data['predicted_count'] = model.predict(data[features]).round().astype(int)

# ------------------------------------------------------------------------
# Sidebar Forecast Controls
# ------------------------------------------------------------------------
st.sidebar.header("Forecast Settings")

target_hour = st.sidebar.slider(
    "Target Forecast Hour (0:00 - 23:00)", 
    min_value=0, 
    max_value=23, 
    value=now_mst.hour,
    format="%d:00"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hypothetical Weather Conditions")

input_temp_f = st.sidebar.slider("Temperature (°F)", -10.0, 110.0, 68.0, step=1.0)
input_precip_in = st.sidebar.slider("Precipitation (in/hr)", 0.0, 2.0, 0.0, step=0.01, format="%.2f")
input_wind_mph = st.sidebar.slider("Wind Speed (mph)", 0.0, 60.0, 8.0, step=1.0)

# Build feature set for upcoming inference
forecast_df = pd.DataFrame({
    'hour': [target_hour],
    'day_of_week': [now_mst.weekday()],
    'temp_f': [input_temp_f],
    'precip_in_hr': [input_precip_in],
    'wind_mph': [input_wind_mph]
})

predicted_count = int(model.predict(forecast_df)[0])

if input_precip_in >= 0.25:
    weather_desc = "Adverse Weather"
elif input_precip_in >= 0.05:
    weather_desc = "Light Rain"
else:
    weather_desc = "Clear"

hist_avg = int(data[data['hour'] == target_hour]['visitor_count'].mean())

# ------------------------------------------------------------------------
# KPI Summary Metrics
# ------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric(f"Predicted Visitors ({target_hour:02d}:00)", f"{predicted_count}")
col2.metric(f"Historical Avg ({target_hour:02d}:00)", f"{hist_avg}")
col3.metric("Weather Condition", weather_desc)

st.markdown("---")

# ------------------------------------------------------------------------
# Synchronized Altair Visualizations
# ------------------------------------------------------------------------
st.subheader("Historical Validation: Actual vs. Predicted Traffic & Weather Dynamics")

# Subset to the last 5 days for clear viewing
recent_data = data.tail(24 * 5).copy()

# Melt actual and predicted for multi-line plotting
traffic_melted = recent_data.melt(
    id_vars=['timestamp'], 
    value_vars=['visitor_count', 'predicted_count'],
    var_name='Metric', 
    value_name='Visitors'
)
traffic_melted['Metric'] = traffic_melted['Metric'].map({
    'visitor_count': 'Actual Foot Traffic',
    'predicted_count': 'Model Predicted Traffic'
})

# Define shared selection for interactive x-axis zoom/pan synchronization
shared_zoom = alt.selection_interval(bind='scales', encodings=['x'])

# Subplot 1: Foot Traffic (Actual vs. Predicted)
traffic_chart = alt.Chart(traffic_melted).mark_line().encode(
    x=alt.X('timestamp:T', title=None, axis=alt.Axis(labels=False, ticks=False)),
    y=alt.Y('Visitors:Q', title='Store Visitors'),
    color=alt.Color('Metric:N', title=None, legend=alt.Legend(orient='top-left')),
    tooltip=[
        alt.Tooltip('timestamp:T', title='Time', format='%b %d, %I:%M %p'),
        alt.Tooltip('Metric:N', title='Series'),
        alt.Tooltip('Visitors:Q', title='Count')
    ]
).properties(
    height=260
).add_params(
    shared_zoom
)

# Subplot 2A: Precipitation (Bar Chart)
precip_chart = alt.Chart(recent_data).mark_bar(opacity=0.6).encode(
    x=alt.X('timestamp:T', title='Date & Time (MST)', axis=alt.Axis(format='%b %d, %I %p')),
    y=alt.Y('precip_in_hr:Q', title='Precipitation (in/hr)', scale=alt.Scale(zero=True)),
    tooltip=[
        alt.Tooltip('timestamp:T', title='Time', format='%b %d, %I:%M %p'),
        alt.Tooltip('precip_in_hr:Q', title='Precipitation (in/hr)', format='.2f'),
        alt.Tooltip('temp_f:Q', title='Temp (°F)', format='.1f'),
        alt.Tooltip('wind_mph:Q', title='Wind (mph)', format='.1f')
    ]
)

# Subplot 2B: Temperature (Line Chart overlaying the weather subplot)
temp_chart = alt.Chart(recent_data).mark_line(strokeDash=[4, 4]).encode(
    x=alt.X('timestamp:T'),
    y=alt.Y('temp_f:Q', title='Temperature (°F)', scale=alt.Scale(zero=False), axis=alt.Axis(orient='right')),
    tooltip=[
        alt.Tooltip('timestamp:T', title='Time', format='%b %d, %I:%M %p'),
        alt.Tooltip('temp_f:Q', title='Temp (°F)', format='.1f')
    ]
)

weather_combined = alt.layer(precip_chart, temp_chart).resolve_scale(
    y='independent'
).properties(
    height=160
).add_params(
    shared_zoom
)

# Vertically concatenate the synchronized charts
synchronized_dashboard = alt.vconcat(
    traffic_chart, 
    weather_combined
).resolve_scale(
    x='shared'
)

st.altair_chart(synchronized_dashboard, use_container_width=True)
