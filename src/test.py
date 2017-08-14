from wrapper import Forecast

print('START')

filepath = '../../../../PycharmProjects/taxi_data/5_percentile_forecasts/combined_2015_clean.csv'
weather = '../../../python/joined_avgweather_2015_copy.csv'

# Create new instance of forecast with default values for drop_monday, force_gamma, intercept
fc = Forecast(filepath)

# Specify certain parameters for this forecast to be used in forecast creation
fc.set_params(drop_monday=True, interval=10)

# Create generic forecast, specify path to weather file
fc.create_generic(weather)

# Get forecasts and coefficients; these forecasts/coefficients will be stored as well
forecasts = fc.get_forecasts()
coeffs = fc.get_coefficients()

# Get a summary of the forecast
fc.coefficient_summary()

# Get basic summary statistics
summary_stats = fc.forecast_summary(by_route=True)
print summary_stats.head()

# Autocorrelation analysis
autocorr = fc.autocorrelation_summary()
print autocorr.head()

