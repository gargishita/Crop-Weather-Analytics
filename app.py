import altair as alt
import os
import requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scripts.weather import get_weather
from scripts.geolocation import get_coordinates
from dotenv import load_dotenv
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import joblib
import torch.nn as nn



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----- App Title -----
st.set_page_config(page_title="Crop-Weather-Analytics Dashboard", layout="wide")
st.title("🌾 Welcome")

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


# ----- Image Upload Section -----
uploaded_image = st.file_uploader("📷 Upload an image of the diseased crop", type=["jpg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    

    # Paths
    MODEL_PATH = "models/crop_disease_resnet18.pth"
    CROP_ENCODER_PATH = "models/crop_label_encoder.pkl"
    DISEASE_ENCODER_PATH = "models/disease_label_encoder.pkl"

    # Set device and verify GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Warning: GPU not detected. Prediction will run on CPU, which is slower.")

    # Load encoders
    crop_encoder = joblib.load(CROP_ENCODER_PATH)
    disease_encoder = joblib.load(DISEASE_ENCODER_PATH)

    # Define transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # Custom Fully Connected Layer (must match training script)
    class CustomFC(nn.Module):
        def __init__(self, in_features, num_crops, num_diseases):
            super(CustomFC, self).__init__()
            self.crop = nn.Linear(in_features, num_crops)
            self.disease = nn.Linear(in_features, num_diseases)

        def forward(self, x):
            return {
                "crop": self.crop(x),
                "disease": self.disease(x)
            }

    # Load model
    model = models.resnet18(weights=None)  # Use weights=None since we load custom weights
    num_crops = len(crop_encoder.classes_)
    num_diseases = len(disease_encoder.classes_)
    in_features = model.fc.in_features
    model.fc = CustomFC(in_features, num_crops, num_diseases)  # Use CustomFC instead of ModuleDict
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Image prediction
    image = Image.open(uploaded_image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model(input_tensor)
        crop_output, disease_output = outputs["crop"], outputs["disease"]
        pred_crop_idx = crop_output.argmax(1).item()
        pred_disease_idx = disease_output.argmax(1).item()

        predicted_crop = crop_encoder.inverse_transform([pred_crop_idx])[0]
        predicted_disease = disease_encoder.inverse_transform([pred_disease_idx])[0]

    # Adjust disease name if "none"
    if predicted_disease == "none":
        predicted_disease = "healthy"

    # Store in session state (optional if used elsewhere)
    st.session_state["predicted_crop"] = predicted_crop
    st.session_state["predicted_disease"] = predicted_disease

    # Display prediction
    st.markdown("### 🧠 ML Prediction")
    st.success(f"🟢 Crop: **{predicted_crop.title()}**")
    st.error(f"🔴 Disease: **{predicted_disease.title()}**")
    
    

    # ----- AI-Based Treatment Advisory -----
    st.markdown("### 💊 AI-Based Treatment Advisory")

    def fetch_treatment_advisory(crop, disease):
        prompt = f"""Provide concise, bullet-point treatment advice for {disease} in {crop} crops.
    Include:
    - Recommended pesticides/fungicides
    - Application dosage
    - Optimal timing
    - Cultural control methods
    - Resistance management tips"""

        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are an expert agronomist."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 1,
            "stop": None
        }

        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Groq API Error: {str(e)}")
            return None



    if st.button("🔍 Get AI Treatment Advisory"):
        with st.spinner("Fetching treatment advice..."):
            advice = fetch_treatment_advisory(predicted_crop, predicted_disease)
            if advice:
                st.write(advice)
            else:
                st.warning("No treatment advice could be retrieved.")


# ----- Location Input -----
st.markdown("---")
st.subheader("📍 Enter your location to get relevant insights")
location = st.text_input("Enter a city/town name")

if location:
    lat, lng = get_coordinates(location)
    weather = get_weather(location)

    if lat and lng and weather:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latitude", lat)
        with col2:
            st.metric("Longitude", lng)

        st.success(f"🌤️ Weather Summary: {weather['weather'][0]['description'].title()}")

        # ----- Dynamic Weather-Based Disease Insights -----
        with st.expander(" ☁ Smart Weather Insights", expanded=True):
            condition = weather["weather"][0]["main"].lower()
            description = weather["weather"][0]["description"].lower()
            temp = weather["main"]["temp"]
            humidity = weather["main"]["humidity"]
            wind_speed = weather["wind"]["speed"]

            # 🌤️ Visual weather summary with icons
            condition_emoji = {
                "clear": "☀️", "clouds": "☁️", "rain": "🌧️", "drizzle": "🌦️", "thunderstorm": "⛈️",
                "snow": "❄️", "mist": "🌫️", "fog": "🌫️", "haze": "🌫️"
            }
            emoji = condition_emoji.get(condition, "🌡️")

            st.subheader(f"{emoji} Current Weather Conditions")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌡 Temperature", f"{temp} °C")
            with col2:
                st.metric("💧 Humidity", f"{humidity} %")
            with col3:
                st.metric("🌬 Wind Speed", f"{wind_speed} m/s")

            st.caption(f"**Condition:** {condition.title()} | **Description:** {description.title()}")

            # 💡 Weather-Based Insights
            st.markdown("### 🧠 General Weather Based Disease Risk Insights")

            insight_shown = False

            if humidity > 80:
                st.warning("⚠️ High humidity detected. Fungal diseases like leaf rust and mildew may be more prevalent.")
                insight_shown = True
            if humidity > 80 and ("rain" in condition or "drizzle" in condition):
                st.error("🚨 Prolngged humidity and rainfall can accelerate the spread of fungal infections in crops.")
                insight_shown = True
            if temp < 15 and humidity > 70:
                st.info("🧊 Cold and damp conditions may increase risk of soil-borne pathogens like root rot.")
                insight_shown = True
            if wind_speed > 10:
                st.warning("🌬️ High wind speeds may promote the spread of airborne diseases like rust or blight.")
                insight_shown = True
            if condition in ["fog", "mist", "haze"]:
                st.warning("🌫️ Foggy or misty conditions detected. Watch for moisture-loving diseases like powdery mildew.")
                insight_shown = True
            if "light rain" in description:
                st.info("☔ Light rain can lead to dew formation, creating favorable conditions for early-stage fungal growth.")
                insight_shown = True
            if temp > 35 and humidity < 40:
                st.warning("🔥 Hot and dry weather may attract pests like aphids and mites.")
                insight_shown = True
            if not insight_shown:
                st.info("✅ No concerning weather-based disease risks detected under current conditions.")
    else:
        st.error("❌ Could not fetch weather or geolocation data.")

# ----- Visual Dashboard Section -----
st.markdown("---")
st.subheader("📊 Analytics Dashboard")

selected_tab = st.sidebar.radio("📂 Choose a section", ["Live Disease Risk Map", "Day Wise Weather Analytics", "Month Wise Weather Analytics"])

#######################################################################################################################################

if selected_tab == "Live Disease Risk Map":
    st.markdown("### 🌍 Live Disease Risk Map")

    from streamlit import cache_data

    if st.button("📡 Generate Real-Time Disease Risk Map"):
        try:
            region_df = pd.read_csv("data/IndianCities.csv")
            st.success("🗺️ Loaded city dataset successfully.")

            @st.cache_data(ttl=3600)
            def get_cached_weather(city):
                return get_weather(city)

            st.info("📡 Fetching weather data and assigning disease risk levels...")
            progress = st.progress(0)

            results = []
            for i, (_, row) in enumerate(region_df.iterrows()):
                city = row["City"]
                state = row["State"]
                lat = row["Latitude"]
                lng = row["Longitude"]

                weather = get_cached_weather(city)

                if weather:
                    temp = weather["main"]["temp"]
                    humidity = weather["main"]["humidity"]
                    wind = weather["wind"]["speed"]
                    desc = weather["weather"][0]["description"].title()

                    # --- Enhanced logic ---
                    if humidity > 85 and temp > 30 and wind < 5:
                        risk = "Very High"
                    elif humidity > 80 and temp > 25:
                        risk = "High"
                    elif 60 < humidity <= 80 and 20 < temp <= 30:
                        if wind < 7:
                            risk = "Medium"
                        else:
                            risk = "Low"
                    elif humidity < 40 and temp > 35:
                        risk = "Dry Heat Risk"
                    elif wind >= 12 and humidity < 50:
                        risk = "Wind-Pest Risk"
                    else:
                        risk = "Low"

                    results.append({
                        "region": city,
                        "state": state,
                        "lat": lat,
                        "lng": lng,
                        "temp": temp,
                        "humidity": humidity,
                        "wind": wind,
                        "description": desc,
                        "risk": risk
                    })
                else:
                    results.append({
                        "region": city,
                        "state": state,
                        "lat": lat,
                        "lng": lng,
                        "temp": None,
                        "humidity": None,
                        "wind": None,
                        "description": "Unavailable",
                        "risk": "Unknown"
                    })

                progress.progress((i + 1) / len(region_df))

            progress.empty()

            weather_df = pd.DataFrame(results)

            color_map = {
                "Very High": "darkred",
                "High": "red",
                "Medium": "orange",
                "Low": "green",
                "Dry Heat Risk": "purple",
                "Wind-Pest Risk": "blue",
                "Unknown": "gray"
            }

            fig = px.scatter_map(
                weather_df,
                lat="lat",
                lon="lng",
                color="risk",
                color_discrete_map=color_map,
                hover_name="region",
                hover_data={
                    "state": True,
                    "temp": ":.1f°C",
                    "humidity": ":.1f%",
                    "wind": ":.1f km/h",
                    "description": True,
                    "risk": True,
                    "lat": False,
                    "lng": False
                },
                size=[10] * len(weather_df),
                size_max=10,
                zoom=5,
                height=600,
                title="<b>🌱 Real-Time Crop Disease Risk by Region</b>",
            )

            fig.update_traces(
                marker=dict(opacity=0.85, sizemode='diameter', size=16),
                selector=dict(mode='markers')
            )

            fig.update_traces(
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "<b>%{customdata[0]}</b><br>"
                    "<b>Temp:</b> %{customdata[1]}<br>"
                    "<b>Humidity:</b> %{customdata[2]}<br>"
                    "<b>Wind:</b> %{customdata[3]}<br>"
                    "<b>Conditions:</b> %{customdata[4]}<br>"
                    "<b>Risk:</b> %{customdata[5]}<extra></extra>"
                )
            )

            fig.update_layout(
                mapbox_style="stamen-terrain",
                margin={"r": 0, "t": 60, "l": 0, "b": 0},
                hovermode="closest",
                legend_title_text="<b>Disease Risk</b>",
                plot_bgcolor="rgba(0,0,0,0)",
                mapbox=dict(
                    bearing=0,
                    pitch=0,
                    zoom=4,
                    center=dict(lat=20.5937, lon=78.9629)
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Failed to load or process map: {str(e)}")
    else:
        st.info("👆 Click the button above to generate the disease risk map.")

###########################################################################################################################################

elif selected_tab == "Day Wise Weather Analytics":
    st.markdown("### 🌤️ Weather Trends & Insights")

    # Load city data
    try:
        city_coords = pd.read_csv("data/IndianCities.csv")
        city = st.selectbox("Select a Region to Analyze", 
                            options=sorted(city_coords["City"].unique()),
                            index=None,
                            placeholder="Choose a city...")

        if not city:
            st.info("Please select a city out of the agri prone cities available here to begin analysis")
            st.stop()

    except Exception as e:
        st.error("⚠️ Could not load city database. Please check the data file.")
        st.stop()

    # Get coordinates
    try:
        row = city_coords[city_coords["City"] == city].iloc[0]
        lat, lon = row["Latitude"], row["Longitude"]
        st.success(f"📍 Analyzing **{city}** (Lat: {lat:.4f}, Lon: {lon:.4f})")
    except Exception as e:
        st.error(f"🚨 Could not find coordinates for {city}")
        st.stop()

    # ===== WEATHER ANALYSIS =====
    st.subheader("⛅ Weather Conditions")

    def get_safe_weather(lat, lon, days=30):
        base_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "past_days": days,
            "timezone": "auto"
        }

        try:
            response = requests.get("https://api.open-meteo.com/v1/forecast", params=base_params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Weather API error: {str(e)}")
            return None

    try:
        with st.spinner("Loading weather data..."):
            weather_data = get_safe_weather(lat, lon)

            if weather_data and "daily" in weather_data:
                weather_df = pd.DataFrame({
                    "date": pd.to_datetime(weather_data["daily"]["time"]),
                    "temp_max": weather_data["daily"]["temperature_2m_max"],
                    "temp_min": weather_data["daily"]["temperature_2m_min"],
                    "precipitation": weather_data["daily"].get("precipitation_sum", [0]*len(weather_data["daily"]["time"]))
                })

                # Temperature visualization
                st.subheader("🌡️ Temperature Trends")
                fig_temp = px.line(weather_df, x="date", y=["temp_max", "temp_min"],
                                   title=f"Temperature Trend in {city}",
                                   labels={"value": "Temperature (°C)", "date": "Date"})
                st.plotly_chart(fig_temp, use_container_width=True)

                # Precipitation visualization
                st.subheader("🌧️ Rainfall Patterns")
                fig_rain = px.bar(weather_df, x="date", y="precipitation",
                                  title=f"Daily Rainfall in {city} (mm)")
                st.plotly_chart(fig_rain, use_container_width=True)

                # Heatmap
                st.subheader("🌡️ Temperature vs Rainfall Heatmap")
                heatmap_df = weather_df.copy()
                heatmap_df["temp_avg"] = heatmap_df[["temp_max", "temp_min"]].mean(axis=1)
                fig_heatmap = px.density_heatmap(heatmap_df, x="temp_avg", y="precipitation",
                                                 nbinsx=30, nbinsy=30,
                                                 title="Heatmap: Avg Temp vs Precipitation",
                                                 labels={"temp_avg": "Average Temp (°C)", "precipitation": "Rainfall (mm)"})
                st.plotly_chart(fig_heatmap, use_container_width=True)


                # Summary Metrics
                st.subheader("📊 Summary Insights")
                cols = st.columns(3)
                metrics = [
                    ("🌡️ Avg Temp", f"{weather_df[['temp_max','temp_min']].mean().mean():.1f}°C"),
                    ("🔥 Max Temp", f"{weather_df['temp_max'].max():.1f}°C"),
                    ("🌧️ Total Rain", f"{weather_df['precipitation'].sum():.1f} mm")
                ]

                for col, (label, value) in zip(cols, metrics):
                    col.metric(label, value)

            else:
                st.warning("Weather data unavailable. Try again later.")

    except Exception as e:
        st.error(f"Weather data error: {str(e)}")

###########################################################################################################################################

# ===== CROP MARKET ANALYTICS =====
elif selected_tab == "Month Wise Weather Analytics":
    import streamlit as st
    import pandas as pd
    import requests
    from scipy.stats import ttest_ind, chi2_contingency, f_oneway, zscore, pearsonr
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import itertools

    st.markdown("🌬️ Weather-Based Statistical Analysis")

    # --- Load weather data from Open-Meteo API ---
    def fetch_weather_data(latitude, longitude):
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={latitude}&longitude={longitude}"
            f"&start_date={start_date}&end_date={end_date}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean"
            "&timezone=Asia%2FKolkata"
        )
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data["daily"])
        df["time"] = pd.to_datetime(df["time"])
        return df

    # --- Get city coordinates using OpenStreetMap ---
    @st.cache_data
    def get_coordinates(city_name):
        url = f"https://nominatim.openstreetmap.org/search?city={city_name}&country=India&format=json"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.ok and response.json():
            data = response.json()[0]
            return float(data['lat']), float(data['lon'])
        return None, None

    # --- User input for multiple cities ---
    st.subheader("📍 Enter multiple Indian cities (comma separated)")
    city_inputs = st.text_input("Cities (e.g., Delhi, Mumbai, Chennai)")

    summary = []

    # --- Define crop conditions ---
    crop_conditions = {
        "Wheat": {"temp": (20, 25), "humidity": (50, 60), "rain": (0, 10)},
        "Rice": {"temp": (25, 35), "humidity": (70, 90), "rain": (20, 100)},
        "Mustard": {"temp": (18, 25), "humidity": (40, 60), "rain": (0, 10)},
        "Maize": {"temp": (21, 27), "humidity": (60, 70), "rain": (5, 20)}
    }

    if city_inputs:
        cities = [c.strip() for c in city_inputs.split(",") if c.strip()]
        all_dfs = []
        for city in cities:
            lat, lon = get_coordinates(city)
            if lat is None:
                st.warning(f"Couldn't fetch coordinates for {city}. Skipping...")
                continue
            df = fetch_weather_data(lat, lon)
            df["City"] = city.title()
            df["month"] = df["time"].dt.month_name()

            # Crop suitability check
            for crop, conds in crop_conditions.items():
                df[f"{crop}_Suitable"] = df.apply(
                    lambda row: conds["temp"][0] <= row["temperature_2m_max"] <= conds["temp"][1] and
                                conds["humidity"][0] <= row["relative_humidity_2m_mean"] <= conds["humidity"][1] and
                                conds["rain"][0] <= row["precipitation_sum"] <= conds["rain"][1], axis=1)
            all_dfs.append(df)

        if all_dfs:
            df_all = pd.concat(all_dfs)

            st.header("📈 Monthly Avg Max Temperature by City")
            avg_temp = df_all.groupby(["City", "month"])["temperature_2m_max"].mean().unstack()
            st.line_chart(avg_temp.T)

            st.header("📈 Monthly Avg Humidity by City")
            avg_hum = df_all.groupby(["City", "month"])["relative_humidity_2m_mean"].mean().unstack()
            st.line_chart(avg_hum.T)

            st.header("📈 Monthly Total Rainfall by City")
            total_rain = df_all.groupby(["City", "month"])["precipitation_sum"].sum().unstack()
            st.line_chart(total_rain.T)

            st.header("🌾 Crop Suitability by City")
            crop = st.selectbox("Choose Crop to Compare", list(crop_conditions.keys()))
            for df in all_dfs:
                city = df["City"].iloc[0]
                suitable_by_month = df.groupby("month")[f"{crop}_Suitable"].sum()
                best_month = suitable_by_month.idxmax()
                st.markdown(f"**{city}** — Best month to grow {crop}: **{best_month}**")
                st.bar_chart(suitable_by_month)

            st.header("📊 Detailed Statistical Insights (Interpreted)")
            for df1, df2 in itertools.combinations(all_dfs, 2):
                city1, city2 = df1["City"].iloc[0], df2["City"].iloc[0]

                st.subheader(f"🔍 Comparing {city1} and {city2}")

                # T-Test
                st.markdown("### ✅ T-Test (Mean Temperature Comparison)")
                t_stat, p_t = ttest_ind(df1["temperature_2m_max"], df2["temperature_2m_max"])
                st.write(f"T-statistic: {t_stat:.2f}, P-value: {p_t:.4f}")
                if p_t < 0.05:
                    st.success(f"Since p < 0.05, **{city1} and {city2} have significantly different average maximum temperatures**, which means one of the cities could be more or less favorable for specific crops requiring narrow temperature ranges.")
                else:
                    st.info(f"Since p > 0.05, **no significant difference in average maximum temperatures** — meaning similar climatic suitability.")

                # Correlation
                st.markdown("### ✅ Pearson Correlation")
                corr, _ = pearsonr(df1["temperature_2m_max"], df2["temperature_2m_max"][:len(df1)])
                st.write(f"Correlation Coefficient: {corr:.2f}")
                if abs(corr) > 0.7:
                    st.success("Strong temperature trend similarity — similar seasonal cycles.")
                else:
                    st.warning("Weak or moderate correlation — could imply different seasonal timings.")

                # Chi-Square
                st.markdown(f"### ✅ Chi-Square Test (Month vs {crop} Suitability)")
                chi_df = df_all[["month", f"{crop}_Suitable"]].copy()
                chi_table = pd.crosstab(chi_df["month"], chi_df[f"{crop}_Suitable"])
                chi2, chi_p, _, _ = chi2_contingency(chi_table)
                st.write(f"Chi-Square value: {chi2:.2f}, P-value: {chi_p:.4f}")
                if chi_p < 0.05:
                    st.success(f"Month significantly affects the suitability of growing {crop} — helpful for sowing time decisions.")
                else:
                    st.info(f"No clear month-wise impact on {crop} suitability across selected cities.")

                summary.append({
                    "Comparison": f"{city1} vs {city2}",
                    "T-Test P": round(p_t, 4),
                    "Correlation": round(corr, 2),
                    "Chi-Square P": round(chi_p, 4)
                })

            # 🔎 All Cities Comparison (ANOVA only)
            if len(all_dfs) >= 3:
                st.subheader("🔎 Overall Comparison Across All Cities")

                # ANOVA across all cities
                st.markdown("### ✅ ANOVA (Overall Max Temperature Differences)")
                anova_val, anova_p = f_oneway(*[df["temperature_2m_max"] for df in all_dfs])
                st.write(f"ANOVA F-value: {anova_val:.2f}, P-value: {anova_p:.4f}")
                if anova_p < 0.05:
                    st.success("Yes, temperature differences across cities are statistically significant.")
                else:
                    st.info("No significant temperature variation across all cities.")

            st.header("📄 Downloadable Summary")
            summary_df = pd.DataFrame(summary)
            st.dataframe(summary_df)
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button("📅 Download Summary CSV", csv, "weather_summary.csv", "text/csv")
    else:
        st.info("Enter at least two cities to begin analysis.")


# ----- Footer -----
st.markdown("---")
st.caption("© 2025 Crop Disease Analytics | Built with Streamlit")
