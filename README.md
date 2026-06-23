# Weather Event Advisor System (AI + ML + Gemini)

## Overview
This project is an AI-based weather classification system that uses a K-Nearest Neighbors (KNN) machine learning model to predict weather conditions and integrates Google Gemini API to generate intelligent weather advisories.

The system also includes input validation and API error recovery to ensure reliability and robustness in real-world scenarios.

---

## Features

### Weather Prediction (Machine Learning)
- K-Nearest Neighbors (KNN) classifier
- Predicts weather types:
  - Sunny
  - Rainy
  - Cloudy
  - Snowy
- Uses preprocessing techniques:
  - StandardScaler
  - OneHotEncoder

---

### Input Validation System
Ensures only valid sensor data is processed:
- Temperature range validation
- Humidity validation
- Wind speed validation
- Atmospheric pressure validation

This prevents unrealistic inputs from reaching the ML model.

---

### AI Advisory System (Gemini API)
- Uses Google Gemini API to generate intelligent weather advisories
- Provides:
  - Safety recommendations
  - Travel guidance
  - Emergency instructions
  - Logistics suggestions

---

### API Error Recovery
- Implements try-except handling for API failures
- If Gemini API fails:
  - System does not crash
  - Fallback advisory is displayed

Ensures continuous system operation.

---

## System Architecture

User Input  
→ Input Validation  
→ Data Preprocessing  
→ KNN Model Prediction  
→ Weather Classification  
→ Gemini API Advisory Generation  
→ Output Display (Gradio Interface)

---

## Tech Stack
- Python
- Scikit-learn
- Pandas
- NumPy
- Gradio
- Google Gemini API

---

## Project Structure
