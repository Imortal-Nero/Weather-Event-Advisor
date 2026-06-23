\# Weather Event Advisor System (AI + ML + Gemini)



\## Overview

This project is an AI-based weather classification system that uses a K-Nearest Neighbors (KNN) model to predict weather conditions and integrates Google Gemini API to generate intelligent weather advisories.



\---



\## Features



\### Weather Prediction

\- K-Nearest Neighbors (KNN) classifier

\- Predicts: Sunny, Rainy, Cloudy, Snowy

\- Data preprocessing using StandardScaler and OneHotEncoder



\### Input Validation

\- Validates sensor inputs before prediction

\- Checks realistic ranges for temperature, humidity, wind speed, and pressure

\- Prevents invalid data from reaching the model



\### AI Advisory System

\- Uses Google Gemini API to generate weather advisories

\- Provides safety and logistics recommendations



\### API Error Recovery

\- Implements fallback advisory system

\- Ensures system does not crash if API fails



\---



\## System Flow



User Input  

→ Input Validation  

→ Preprocessing  

→ KNN Model  

→ Weather Prediction  

→ Gemini API  

→ Advisory Output (Gradio Interface)



\---



\## Tech Stack

\- Python

\- Scikit-learn

\- Pandas

\- NumPy

\- Gradio

\- Google Gemini API



\---



\## Project Structure

Weather\_Event\_Advisor/

├── app.py

├── train\_model.py

├── evaluate\_model.py

├── test\_api.py

├── dataset/

├── requirements.txt

└── README.md



\---



\## Testing

The system has been tested for:

\- Normal weather conditions

\- Extreme weather inputs

\- Invalid sensor ranges

\- API failure scenarios



\---



\## Future Improvements

\- Real-time weather API integration

\- Mobile application version

\- Cloud deployment

\- IoT sensor integration



\---



\## Author

AI and Machine Learning project for weather classification and advisory generation.

