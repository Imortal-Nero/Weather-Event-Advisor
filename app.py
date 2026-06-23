import pandas as pd
import numpy as np
import joblib
import gradio as gr
import os
import pyperclip   # ✅ NEW (for copy support)

from dotenv import load_dotenv
import google.generativeai as genai

# -------------------------
# ENV
# -------------------------
load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# -------------------------
# MODEL
# -------------------------
model = joblib.load("models/knn_weather.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# -------------------------
# VALIDATION
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
# GEMINI ADVISORY
# -------------------------
def generate_advisory(weather_type):

    prompt = f"""
You are an event management consultant.

Predicted Weather Type: {weather_type}

Generate:
1. Logistics Adjustments
2. Safety Precautions
3. Vendor Notifications
"""

    try:
        return gemini_model.generate_content(prompt).text
    except:
        return "Advisory service unavailable"

# -------------------------
# MAIN FUNCTION
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
        return message, "", ""

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

    weather_type = label_encoder.inverse_transform(prediction)[0]

    advisory = generate_advisory(weather_type)

    # 📄 FINAL REPORT
    report = f"""
Weather Intelligence Report

Prediction: {weather_type}

Inputs:
Temp: {temperature}
Humidity: {humidity}
Wind: {wind_speed}
Precipitation: {precipitation}
Cloud: {cloud_cover}
Pressure: {pressure}
UV: {uv_index}
Season: {season}
Visibility: {visibility}
Location: {location}

ADVISORY:
{advisory}
"""

    return weather_type, advisory, report

# -------------------------
# COPY FUNCTION
# -------------------------
def copy_text(text):
    pyperclip.copy(text)
    return "Copied to clipboard ✅"

# -------------------------
# UI
# -------------------------
with gr.Blocks() as app:

    gr.Markdown("# 🌦 Weather Intelligence System")

    with gr.Row():

        with gr.Column():
            t = gr.Number(label="Temperature")
            h = gr.Number(label="Humidity")
            w = gr.Number(label="Wind Speed")
            p = gr.Number(label="Precipitation")
            c = gr.Dropdown(["clear","partly cloudy","overcast"], label="Cloud Cover")
            pr = gr.Number(label="Pressure")
            uv = gr.Number(label="UV Index")
            s = gr.Dropdown(["Winter","Spring","Summer","Autumn"], label="Season")
            v = gr.Number(label="Visibility")
            l = gr.Dropdown(["inland","coastal","mountain"], label="Location")

            btn = gr.Button("Run")

        with gr.Column():
            out1 = gr.Textbox(label="Prediction")
            out2 = gr.Textbox(label="AI Advisory")
            report_box = gr.Textbox(label="Full Report", lines=15)

            copy_btn = gr.Button("📋 Copy Report")
            copy_status = gr.Textbox(label="Status")

    btn.click(
        predict_weather,
        inputs=[t,h,w,p,c,pr,uv,s,v,l],
        outputs=[out1,out2,report_box]
    )

    copy_btn.click(
        copy_text,
        inputs=report_box,
        outputs=copy_status
    )

app.launch()