import math
import numpy as np
import pandas as pd


class Forecast:

    COLUMNS = ['date', 'n_trips', 'avg_duration', 'avg_distance', 'avg_fare', 'avg_speed']

    # Default settings for the forecast
    drop_monday = True
    force_gamma = False
    intercept = False

    # Stored values for the lmaos
    data = None
    generic_forecast = None
    weather_merged = False


    def __init__(self, data_filepath):
        try:
            data = pd.read_csv(data_filepath)
        except:
            print("Error reading raw data; data object is None, try again.")



