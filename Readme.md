# Retail Foot Traffic Predictor 🌦️🚶‍♂️

An interactive, machine-learning-powered dashboard built with Streamlit that predicts near-term (1-2 hour) retail foot traffic based on localized weather conditions. 

## Overview
Foot traffic is highly susceptible to sudden shifts in the weather. This application utilizes a Random Forest regression model to learn historical traffic baselines (e.g., time of day, day of week) and quantify the penalty of adverse weather conditions like extreme temperatures, high winds, and heavy precipitation.

**⚠️ Note on Data:** 
This current repository serves as a **prototype** and utilizes an internal data synthesizer to mock 30 days of historical foot traffic and weather data. 

In a **production environment**, this pipeline is designed to ingest real historical foot traffic data (e.g., from POS systems, door sensors, or Wi-Fi analytics) and integrate directly with high-resolution forecasting APIs, specifically the **Google WeatherNext 3 API**, to fetch highly accurate 1-2 hour predictive weather variables.

## Features
*   **Predictive Modeling:** Uses `scikit-learn` (Random Forest Regressor) to predict upcoming visitor volume.
*   **Interactive UI:** Adjustable sidebar sliders allow users to input hypothetical near-term weather conditions to see real-time impact on predicted traffic.
*   **Key Performance Indicators:** Instant comparison between the predicted upcoming hour and the historical average for that specific time block.
*   **Historical Visualization:** A 7-day lookback chart displaying synthesized foot traffic baselines.

## Architecture

### Current Prototype State
*   **Data Ingestion:** Synthetic data generation (`app.py`) mimicking diurnal cycles, business hours, and weather penalties.
*   **Model:** Scikit-Learn `RandomForestRegressor` trained on-the-fly.
*   **Deployment:** Streamlit Community Cloud.

### Target Production Architecture
*   **Traffic Data:** Real-time streaming from physical retail sensors / POS databases (e.g., via Google Cloud Pub/Sub or BigQuery).
*   **Weather Data:** Integration with **WeatherNext 3** via Google Cloud. The model will query hourly initialized datasets extracting variables like `total_precipitation_1hr`, `temperature_2m`, and `wind_speed_10m` at a 0.05° to 0.1° resolution.
*   **Storage & Training:** BigQuery for data warehousing and Vertex AI for automated model retraining and endpoint hosting.

## Installation and Usage

To run this prototype locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/retail-traffic-dashboard.git
   cd retail-traffic-dashboard
   ```

2. **Install the dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **View the Dashboard:**
   Open your browser and navigate to `http://localhost:8501`.

## Future Enhancements
*   **Integrate WeatherNext 3 API:** Replace the synthetic inputs with live API calls to Google Cloud for hyper-local 1-hour forecasts.
*   **Model Optimization:** Implement hyperparameter tuning and cross-validation for the Random Forest model, or upgrade to XGBoost.
*   **Multi-Store Support:** Expand the dashboard to handle filtering by different store locations and geographical coordinates.
*   **Holiday / Event Features:** Add calendar integrations to account for holidays, local events, and promotional days which heavily skew baseline traffic.

## License
MIT License. See `LICENSE` for more information.