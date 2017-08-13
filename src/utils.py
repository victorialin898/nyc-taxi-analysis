import pandas as pd

COLUMNS = ['n_trips', 'avg_duration', 'avg_distance', 'avg_fare', 'avg_speed']

# Helper function for merge_day_month
def format_day(i):
    s = str(i)
    if len(s) == 1:
        s= str(0) + s
    return s


def merge_day_month(data):
    data['day'] = data['month'].map(str) + data['day'].map(format_day)
    data = data.drop(['month'], axis=1).rename(columns={'day': 'date'})
    return data


# Helper function for roll
def roll_columns(df, interval):
    dfout = pd.DataFrame(columns=['date']+COLUMNS)
    dfout['date'] = df['date']
    for col in COLUMNS:
        dfout[col] = df[col].shift().rolling(min_periods=interval, window=interval, center=False).mean()
    return dfout


def roll(data, interval):
    df = data.groupby(['pickup_zone', 'dropoff_zone', 'interval_start'])['date','n_trips', 'avg_duration', 'avg_distance', 'avg_fare', 'avg_speed'].apply(roll_columns, interval = interval)
    reindexed_df = df.reset_index().drop(['level_3'], axis=1)
    rolled_names = ['rolling_' + col for col in COLUMNS]
    reindexed_df.rename(index=str, columns=dict(zip(COLUMNS, rolled_names)), inplace=True)
    return reindexed_df


def merge_rolling(formatted_data, rolling_data):
    ind = ['date', 'interval_start', 'pickup_zone', 'dropoff_zone']
    # Casting index to common dtype so join function will work properly
    for i in ind:
        rolling_data[i] = rolling_data[i].astype(int)
        formatted_data[i] = formatted_data[i].astype(int)
    formatted_data.set_index(ind, inplace=True)
    rolling_data.set_index(ind, inplace=True)
    return formatted_data.join(rolling_data)


def merge_weather(merged_rolling, weather_filepath):
    weather = pd.read_csv(weather_filepath, index_col=[0,1,2,3])
    joined_weather = weather.join(merged_rolling)
    joined_weather.drop_duplicates(inplace=True)
    return joined_weather.reset_index()


def calculate_deviation(data):
    data['n_trips_deviation'] = data['n_trips_forecast'] - data['n_trips']
    data['duration_deviation'] = data['duration_forecast'] - data['avg_duration']
    data['distance_deviation'] = data['distance_forecast'] - data['avg_distance']
    data['fare_deviation'] = data['fare_forecast'] - data['avg_fare']
    data['speed_deviation'] = data['speed_forecast'] - data['avg_speed']
    return data