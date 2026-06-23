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

```
Weather_Event_Advisor/
│
├── app.py
├── train_model.py
├── evaluate_model.py
├── test_api.py
├── dataset/
│   └── weather_classification_data.csv
├── requirements.txt
├── README.md
└── models/ (optional trained artifacts)
```

---

## Testing

The system has been tested for the following scenarios:

### Valid Weather Conditions
- Sunny conditions
- Rainy conditions
- Cloudy conditions
- Snowy conditions

### Edge Cases
- Extreme temperature values
- High humidity conditions
- Low visibility scenarios

### Validation Testing
- Out-of-range sensor inputs
- Invalid numerical values

### API Failure Testing
- Invalid API key
- Network failure
- Gemini service downtime

---

## Input Validation Rules

- Temperature: -60°C to 60°C
- Humidity: 0% to 100%
- Wind Speed: 0 to 300 km/h
- Atmospheric Pressure: 800 to 1200 hPa

Invalid inputs are rejected before prediction.

---

## API Error Handling

If Gemini API fails, the system returns a fallback response:

- System remains functional
- Basic safety advisory is shown
- No system crash occurs

---

## Future Improvements
- Real-time weather API integration
- Mobile application development
- Cloud deployment
- IoT sensor integration
- Deep learning-based weather prediction

---

## Dataset Source

The dataset used in this project is publicly available and used for training the weather classification model.

- Dataset Name: Weather Classification Dataset  
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/saurabhshahane/weather-classification

This dataset contains weather-related parameters such as temperature, humidity, wind speed, pressure, and visibility, which are used to classify weather conditions.

---

## References

- Google Gemini API Documentation: https://ai.google.dev/
- Scikit-learn Documentation: https://scikit-learn.org/
- Gradio Documentation: https://www.gradio.app/

---

## Author
Developed as an AI and Machine Learning project for intelligent weather classification and advisory generation.

---

## License
This project is for educational purposes.
