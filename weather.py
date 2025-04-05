import pandas as pd
import requests
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
DATE_RANGE = ("2022-01-01", "2024-08-31")
WEATHER_PARAMS = "temperature_2m,relative_humidity_2m,cloudcover,precipitation"

# City coordinates for Spain and Italy

CITY_COORDINATES = {
    "ES": {
        "Madrid": (40.4168, -3.7038),
        "Barcelona": (41.3851, 2.1734),
        "Malaga": (36.7213, -4.4214),      
    },
    "IT": {
        "Milan": (45.4642, 9.1900),
        "Turin": (45.0703, 7.6869),
        "Venice": (45.4408, 12.3155),
    }
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_city_weather(processed_data_dir: str, lat: float, lon: float, city: str, country_code: str) -> pd.DataFrame:
    """
    Fetch weather data for a given city using Open Meteo API.
    
    Args:
        lat (float): Latitude of the city.
        lon (float): Longitude of the city.
        city (str): Name of the city.
        country_code (str): Country code ("IT" or "ES").
        
    Returns:
        pd.DataFrame: Weather data for the specified city.
    """
    cache_path = os.path.join(processed_data_dir, f"weather_{country_code}_{city}.csv")

    # Use cached data if available
    if os.path.exists(cache_path):
        logger.info(f"Loading cached weather data for {city}, {country_code}.")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    
    # Build API request URL
    params = {
       "latitude": lat,
        "longitude": lon,
        "start_date": DATE_RANGE[0],
        "end_date": DATE_RANGE[1],
        "hourly": WEATHER_PARAMS,
        "timezone": "auto"
    }

    logger.info(f"Fetching weather data for {city}, {country_code} from Open Meteo API.")
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    # Process the response
    data = response.json()["hourly"]
    df = pd.DataFrame(data).set_index("time")
    df.to_csv(cache_path, index=True)
    return df


def calculate_country_average(processed_data_dir:str, country_code: str) -> pd.DataFrame:
    """
    Calculate the average weather data for several cities in a country.

    Args:
        processed_data_dir (str): Directory to save processed data.
        country_code (str): Country code ("IT" or "ES").

    Returns:
        pd.DataFrame: Average weather data for the specified country.
    """
    combined = None

    for city, (lat, lon) in CITY_COORDINATES[country_code].items():
        if combined is None:
            combined = fetch_city_weather(processed_data_dir, lat, lon, city, country_code)
        else:
            city_df = fetch_city_weather(processed_data_dir, lat, lon, city, country_code)
            combined = combined.add(city_df, fill_value=0)

    # Average the data
    combined /= len(CITY_COORDINATES[country_code])


    combined.rename(columns={"time": "DATETIME"}, inplace=True)
    combined.index.rename("DATETIME", inplace=True)
    return combined