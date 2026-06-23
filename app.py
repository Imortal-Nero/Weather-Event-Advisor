import pandas as pd
import numpy as np
import joblib
import gradio as gr
import os

from dotenv import load_dotenv
import google.generativeai as genai

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# -------------------------
# Load Trained Model
# -------------------------
model = joblib.load("models/knn_weather.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# -------------------------
# Input Validation
# -------------------------
def validate_inputs(
    temperature,
    humidity,
    wind_speed,
    precipitation,
    cloud_cover,
    pressure,
    uv_index,
    season,
    visibility,
    location
):

    if temperature < -60 or temperature > 60:
        return False, "Temperature must be between -60 and 60°C"

    if humidity < 0 or humidity > 100:
        return False, "Humidity must be between 0 and 100%"

    if wind_speed < 0 or wind_speed > 300:
        return False, "Wind speed must be between 0 and 300 km/h"

    if precipitation < 0 or precipitation > 100:
        return False, "Precipitation must be between 0 and 100%"

    if pressure < 800 or pressure > 1200:
        return False, "Atmospheric pressure out of range"

    if uv_index < 0 or uv_index > 15:
        return False, "UV Index must be between 0 and 15"

    if visibility < 0 or visibility > 50:
        return False, "Visibility must be between 0 and 50 km"

    return True, ""

# -------------------------
# Gemini Advisory
# -------------------------
def generate_advisory(weather_type):

    prompt = f"""
You are an event management consultant.

Predicted Weather Type: {weather_type}

Generate:

1. Logistics Adjustments
2. Safety Precautions
3. Vendor Notifications

Use clear bullet points.
"""

    try:

        response = gemini_model.generate_content(prompt)

        return response.text

    except Exception as e:

        return f"""
Advisory service unavailable.

Fallback Advisory

LOGISTICS
• Prepare backup arrangements.

SAFETY
• Monitor weather updates.

VENDOR NOTIFICATIONS
• Inform all vendors about possible weather impacts.

Error:
{str(e)}
"""

# -------------------------
# Prediction Function
# -------------------------
def predict_weather(
    temperature,
    humidity,
    wind_speed,
    precipitation,
    cloud_cover,
    pressure,
    uv_index,
    season,
    visibility,
    location
):

    valid, message = validate_inputs(
        temperature,
        humidity,
        wind_speed,
        precipitation,
        cloud_cover,
        pressure,
        uv_index,
        season,
        visibility,
        location
    )

    if not valid:
        return message, ""

    sample = pd.DataFrame({
        "Temperature":[temperature],
        "Humidity":[humidity],
        "Wind Speed":[wind_speed],
        "Precipitation (%)":[precipitation],
        "Cloud Cover":[cloud_cover],
        "Atmospheric Pressure":[pressure],
        "UV Index":[uv_index],
        "Season":[season],
        "Visibility (km)":[visibility],
        "Location":[location]
    })

    prediction = model.predict(sample)

    weather_type = label_encoder.inverse_transform(
        prediction
    )[0]

    advisory = generate_advisory(weather_type)

    return weather_type, advisory

# -------------------------
# Gradio UI
# -------------------------
app = gr.Interface(
    fn=predict_weather,

    inputs=[
        gr.Number(label="Temperature (°C)"),
        gr.Number(label="Humidity (%)"),
        gr.Number(label="Wind Speed (km/h)"),
        gr.Number(label="Precipitation (%)"),

        gr.Dropdown(
            ["clear", "partly cloudy", "overcast"],
            label="Cloud Cover"
        ),

        gr.Number(label="Atmospheric Pressure"),

        gr.Number(label="UV Index"),

        gr.Dropdown(
            ["Winter", "Spring", "Summer", "Autumn"],
            label="Season"
        ),

        gr.Number(label="Visibility (km)"),

        gr.Dropdown(
            ["inland", "coastal", "mountain"],
            label="Location"
        )
    ],

    outputs=[
        gr.Textbox(label="Predicted Weather Type"),
        gr.Textbox(label="Event Management Advisory")
    ],

    title="AI Weather Event Advisor",

    description="""
Predict weather conditions using a KNN classifier
and generate event management recommendations
using Gemini AI.
"""
)

app.launch()