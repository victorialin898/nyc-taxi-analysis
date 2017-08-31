# nyc-taxi-forecasts

Tools for creating forecasts and performing analysis on datasets provided by the New York City Taxi & Limousine Commission.

Uses the pandas, np, math, sk-learn, datetime libraries.

Includes a set of tools for pre/post forecast data processing in utils.py, forecast creation in forecasting.py.

Both are encapsulated in wrapper.py with the Forecast class. All forecast creation/analysis should be done using the Forecast class. Autocorrelation and variation analysis functionality included in the Forecast class.

Refer to test.py for examples on how to instantiate and use the Forecast class.

## Data Formatting

Raw data for the Forecast class should have 10 columns: 'day' (int), 'month' (int), 'interval_start' (int), pickup_zone (int), dropoff_zone (int), n_trips (int), avg_duration (float), avg_distance (float), avg_speed (float).

Pickup/Dropoff zones should be 1-4, corresponding to: 1: UWS, 2: UES, 3: MTW, 4: MTE

Weather Data should have 8 columns: 'date' (int, mmdd), interval_start (int), pickup_zone (int), dropoff_zone (int), rain (float), rain_ind (int), snow (float), snow_ind (int).

## Creating a Forecast

Import wrapper.py, which contains the Forecast class. Instantiate a Forecast with a string file path to the raw data. Use the set_params() function to specify what parameters the forecast should use.
1. drop_monday: When True, forecast will be created without Monday coefficient. Default True.
2. force_gamma = When True, gamma (coefficient of generic) will be forced to equal 1. Default True.
3. intercept = When True, forecast will be performed with intercept. Default True.
4. interval = The size of window for rolling (generic) forecast, in days. Default 10.

Create the generic forecast from the raw data by calling the create_generic() function, giving path to weather file. You can now retrieve the forecast values and coefficients with get_forecasts() and get_coefficients(). You can also print a summary of the forecast, which gives coefficient, r-value, MSE averages.

Retrieve basic summary statistics with forecast_summary(). Summarize by day-of-week, by route, and by route/day-of-week combinations.

Retrieve autocorrelation analysis results with autocorrelation_daily() and autocorrelation_summary(). You can specify if you want the morning period (7:30-11:30), evening period (15:30-20:30), daytime period (7:30-20:30), and also specify the column to perform analysis on. 

Raw data, combined generic and weather data, forecasts, coefficients, and autocorrelation data are stored in the Forecast object, able to be retrieved at any time.

## Helper Modules

utils.py contains pre-forecast data processing functions, like formatting date, creating generic forecasts, merging dataframes, etc.

forecasting.py contains forecast creation functions, and performs the regression work. 

analyze.py contains post-forecast analysis functions, specifically for autocorrelation and basic summary statistics.

test.py contains an example of how to create a Forecast and use it to perform analysis.
