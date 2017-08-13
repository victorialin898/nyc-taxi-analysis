import math
import numpy as np
import pandas as pd
import utils as util

print('START')

filepath = '../../../../PycharmProjects/taxi_data/5_percentile_forecasts/combined_2015_clean.csv'
weather = '../../../python/joined_avgweather_2015_copy.csv'
raw_data = pd.read_csv(filepath)


formatted_data = util.merge_day_month(raw_data)
rolling_data = util.roll(formatted_data, 10)
merged_rolling = util.merge_rolling(formatted_data, rolling_data)
merged_weather = util.merge_weather(merged_rolling, weather)

print(merged_weather.head())
# merged_weather.to_csv('weathered.csv', header=True)
