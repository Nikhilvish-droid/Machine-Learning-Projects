import joblib
from pathlib import Path
import pandas as pd
import requests

model_dir = Path(__file__).resolve().parent.parent/"models"

def load_model():
    model_pipeline = joblib.load(model_dir/'energy_model.pkl')

    return model_pipeline


def get_coordinate(city):

    url = 'https://geocoding-api.open-meteo.com/v1/search'

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    if not data:
        return None

    location = data["results"][0]

    return location['latitude'], location['longitude']


def get_weather(latitude, longitude):

    url = 'https://api.open-meteo.com/v1/forecast'

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "precipitation_sum",
            "shortwave_radiation_sum"
        ],
        "hourly": [
            "relative_humidity_2m",
            "wind_speed_10m"
        ],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)

    response.raise_for_status()

    data = response.json()

    weather = {
        "temp_avg_c": data["daily"]["temperature_2m_mean"][0],
        "temp_min_c": data["daily"]["temperature_2m_min"][0],
        "temp_max_c": data["daily"]["temperature_2m_max"][0],
        "precip_mm": data["daily"]["precipitation_sum"][0],
        "solar_index": data["daily"]["shortwave_radiation_sum"][0],

        "humidity_pct": sum(
            data["hourly"]["relative_humidity_2m"]
        ) / len(data["hourly"]["relative_humidity_2m"]),

        "wind_kph": sum(
            data["hourly"]["wind_speed_10m"]
        ) / len(data["hourly"]["wind_speed_10m"])
    }

    return weather


def user_input():
    print("-----------------------------")
    print("fill the following question :")
    city = input("enter your city : ")

    latitude, longitude = get_coordinate(city)
    weather_data = get_weather(latitude, longitude)

    home_sqft = float(input("enter your home sqft area : "))
    num_residents = int(input("how many person live in the house : "))
    has_ev = int(input("Has EV? (1 = Yes, 0 = No): "))
    has_solar = int(input("Has Solar? (1 = Yes, 0 = No): "))
    has_pool = int(input("Has Pool? (1 = Yes, 0 = No): "))
    heating_type = input("Heating type (electric/gas/heat_pump): ").strip().lower()
    hvac_age_years = float(input("HVAC age (years): "))
    is_weekend = int(input("Is weekend? (1 = Yes, 0 = No): "))
    is_holiday = int(input("Is holiday? (1 = Yes, 0 = No): "))
    prior_day_kwh = float(input("Previous day's kWh: "))
    prior_week_avg_kwh = float(input("Previous week's average kWh: "))

    user_data = pd.DataFrame([{
        "num_residents": num_residents,
        "home_sqft": home_sqft,
        "has_ev": has_ev,
        "has_solar": has_solar,
        "has_pool": has_pool,
        "heating_type": heating_type,
        "hvac_age_years": hvac_age_years,
        "temp_avg_c": weather_data['temp_avg_c'],
        "temp_min_c": weather_data['temp_min_c'],
        "temp_max_c": weather_data['temp_max_c'],
        "humidity_pct": weather_data['humidity_pct'],
        "wind_kph": weather_data['wind_kph'],
        "precip_mm": weather_data['precip_mm'],
        "solar_index": weather_data['solar_index'],
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "prior_day_kwh": prior_day_kwh,
        "prior_week_avg_kwh": prior_week_avg_kwh
    }])


    model_pipeline = load_model()

    predict_kwh = model_pipeline.predict(user_data)

    print("kwh -->", predict_kwh)

    return predict_kwh




if __name__ == "__main__":

    # print(get_coordinate("switzerland"))
    ans = user_input()
    print("done succesfully")