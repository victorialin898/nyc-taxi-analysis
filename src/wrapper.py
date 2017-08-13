import math
import numpy as np
import pandas as pd
import utils as util
import forecasting as fc


class Forecast:

    # Default settings for the forecast
    drop_monday = True
    force_gamma = True
    intercept = True
    interval = 10

    # Stored values for the lmaos
    raw_data = None
    weather_generic = None

    # Data values
    forecasts = None
    coefficients = None

    def __init__(self, filepath):
        self.set_raw_data(filepath)

    def set_raw_data(self, filepath):
        try:
            self.raw_data = pd.read_csv(filepath)
        except:
            print("Error reading raw data.")

    def set_params(self, drop_monday = True, force_gamma = True, intercept = True, interval = 10):
        self.drop_monday = drop_monday
        self.force_gamma = force_gamma
        self.intercept = intercept
        self.interval = interval

    def create_generic(self, weather_filepath, interval=10):
        self.interval = interval
        if self.raw_data is None:
            print("Error: no raw data. Use set_raw_data to set raw data.")
        else:
            formatted_data = util.merge_day_month(self.raw_data)
            rolling_data = util.roll(formatted_data, self.interval)
            merged_rolling = util.merge_rolling(formatted_data, rolling_data)
            merged_weather = util.merge_weather(merged_rolling, weather_filepath)
            self.weather_generic = merged_weather

    def get_forecasts(self):
        if self.raw_data is None:
            print("Error: no raw data. Use set_raw_data to set raw data.")
            return
        data_dummies = fc.get_dummies(self.weather_generic)
        preds = fc.predict(data_dummies, self.drop_monday, self.force_gamma, self.intercept)
        dev_preds = util.calculate_deviation(preds)
        self.forecasts = dev_preds
        return dev_preds

    def get_coefficients(self):
        if self.raw_data is None:
            print("Error: no raw data. Use set_raw_data to set raw data.")
            return
        data_dummies = fc.get_dummies(self.weather_generic)
        coeffs = fc.coeff(data_dummies, self.drop_monday, self.force_gamma, self.intercept)
        self.coefficients = coeffs
        return coeffs

    def summary(self):
        if self.coefficients is None:
            print("Error: no coefficients. Run get_coefficients to get coefficients.")
            return
        fc.analyze(self.coefficients)


