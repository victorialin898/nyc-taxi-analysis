import math
import numpy as np
import pandas as pd
import utils as util
import forecasting as fc
import analyze as an

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
    autocorr_data = None

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

    def coefficient_summary(self):
        if self.coefficients is None:
            print("Error: no coefficients. Run get_coefficients to get coefficients.")
            return
        fc.analyze(self.coefficients)

    # Finds daily autocorrelation on forecast (0), actual (1), or deviation (2) values, specified by column parameter
    def autocorrelation_daily(self, lag=1, morning=False, evening=False, column=0):
        if self.autocorr_data is None:
            self.autocorr_data = an.load_df(self.forecasts.reset_index())
        data = self.autocorr_data
        if morning:
            data = data.drop(data[(data.interval_start < 730) | (data.interval_start > 1130)].index)
        elif evening:
            data = data.drop(data[(data.interval_start < 1530) | (data.interval_start > 2030)].index)
        return an.route_summary_autocorr(data, lag=lag, col=column).unstack('route')

    # Finds avg autocorrelation on forecast (0), actual (1), or deviation (2) values, specified by column parameter
    def autocorrelation_summary(self, lag=8, morning=False, evening=False, column=0):
        if self.autocorr_data is None:
            self.autocorr_data = an.load_df(self.forecasts.reset_index(), autocorr=True)
        data = self.autocorr_data
        if morning:
            data = data.drop(data[(data.interval_start < 730) | (data.interval_start > 1130)].index)
        elif evening:
            data = data.drop(data[(data.interval_start < 1530) | (data.interval_start > 2030)].index)
        return an.route_summary_autocorr(data, lag=lag, col=column)

    def forecast_summary(self, by_day=True, by_route=False, by_route_dayofweek=False):
        data = an.load_df(self.forecasts.reset_index(), autocorr=False)
        return an.summarize(data, by_day=by_day, by_route=by_route, by_route_dayofweek=by_route_dayofweek)


