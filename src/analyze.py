"""A module for post-forecast analysis, specifically autocorrelation and basic summary statistics."""

import pandas as pd
import datetime

# Location dict for easier route interpretation
LOC_DICT = {1: 'UWS', 2: 'UES', 3: 'MTW', 4: 'MTE'}

# Dictionary with names of current column to perform autocorrelation on
COL_DICT = {0: 'n_trips_forecast', 1: 'n_trips', 2: 'n_trips_deviation'}

# Global list of columns to analyze, change these to do analysis on deviation profiles, need exacts col names
COLUMNS = ['n_trips_forecast', 'duration_forecast', 'distance_forecast', 'fare_forecast', 'speed_forecast']


# Load, clean, augment the given filename into a df, default only considers time intervals between 7:30 and 20:30
def load_df(data, autocorr=True, daytime=True, only_weekdays=True):
    data.replace({'pickup_zone': LOC_DICT, 'dropoff_zone': LOC_DICT}, inplace=True)
    data['route'] = data['pickup_zone'].map(str) + ',' + data['dropoff_zone'].map(str)

    if autocorr:
        data.drop(['pickup_zone', 'dropoff_zone', 'avg_duration', 'avg_distance', 'avg_fare', 'avg_speed',
                   'duration_forecast', 'distance_forecast', 'fare_forecast', 'speed_forecast'],
                  axis=1, inplace=True)
    else:
        data.drop(['pickup_zone', 'dropoff_zone', 'n_trips', 'avg_duration', 'avg_distance', 'avg_fare', 'avg_speed'],
                  axis=1, inplace=True)

    if daytime:
        data.drop(data[(data.interval_start < 730) | (data.interval_start > 2030)].index, inplace=True)

    # add column for day of week
    month = (data['date'] // 100).tolist()
    day = (data['date'] % 100).tolist()
    year = 2015
    daysofweek = []
    for i in range(len(month)):
        d = datetime.datetime(year, month[i], day[i], 1, 1, 1)
        daysofweek.append(d.weekday())
    data['day_of_week'] = daysofweek

    if only_weekdays:
        data.drop(data[(data.day_of_week > 4)].index, inplace=True)

    return data


# Perform autocorrelation on each day given lag amount
def autocorrelate(df, lag, col):
    key = '{:02d}:{:02d}'.format(*divmod(15 * lag, 60))
    dfout = pd.DataFrame({key: [df[COL_DICT[col]].autocorr(lag=lag)] })
    return dfout


# Given df, groups by date and route, then calls autocorrelate function on each day, route group
def daily_route_autocorr(df, lag, col):
    df_corr = df.groupby(['date', 'day_of_week', 'route']).apply(autocorrelate, lag, col)
    return df_corr


# Perform autocorrelation on each day and then average over the year for a summary autocorrelation value
def avg_autocorrelate(df, lag, col):
    keys = ['{:02d}:{:02d}'.format(*divmod(15*l, 60)) for l in range(0, lag+1)]
    vals = []
    for l in range(0, lag+1):
        dra = daily_route_autocorr(df, lag=l, col=col)
        mean = dra[keys[l]].mean()
        vals.append([mean])

    lag_dict = dict(zip(keys, vals))
    dfout = pd.DataFrame(lag_dict)
    return dfout


# Given df, groups by route and calls avg_correlate to retrieve summary autocorrelation values per route
def route_summary_autocorr(df, lag, col):
    df_corr = df.groupby(['route']).apply(avg_autocorrelate, lag, col)
    df_corr = df_corr.reset_index().drop(['level_1'], axis=1).set_index(['route'])
    return df_corr


# Computes summary statistics after grouping by desired group method
def summarize(data, by_day=True, by_route=False, by_route_dayofweek=False):
    if by_route:
        summary = data.groupby(['route']).apply(compute_summary)
    elif by_route_dayofweek:
        summary = data.groupby(['route', 'day_of_week']).apply(compute_summary)
    elif by_day:
        summary = data.groupby(['date', 'day_of_week', 'route']).apply(compute_summary)
    return summary


# Helper function for summarize, computes summary of given df for each category
def compute_summary(df):
    avgs = []
    maxes = []
    mins = []
    maxes_avgs = []
    mins_avgs = []
    stdevs = []

    # Calculate values
    for col in COLUMNS:
        mean = df[col].mean()
        col_max = df[col].max()
        col_min = df[col].min()
        avgs.append(mean)
        maxes.append(col_max)
        mins.append(col_min)
        maxes_avgs.append(col_max/mean)
        mins_avgs.append(col_min / mean)
        stdevs.append(df[col].std())

    cols = ['avg', 'max', 'min', 'max/avg', 'min/avg', 'stdev']
    collection = [avgs, maxes, mins, maxes_avgs, mins_avgs, stdevs]
    dfout = pd.DataFrame(index=COLUMNS, columns=cols)

    # Populate outgoing dataframe
    for y in range(len(COLUMNS)):
        for x in range(len(cols)):
            dfout.loc[COLUMNS[y], cols[x]] = collection[x][y]

    return dfout
