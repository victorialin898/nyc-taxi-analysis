"""A module for the process of forecast creation."""

import pandas as pd
import math
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression


# regression function, returns data frame of coefficients, R^2, MSE for each category
def ijt_regression(df, drop_monday=True, force_gamma=False, predictions=False, intercept=False):

    df.fillna(0.001, inplace=True)

    # Create responses
    if force_gamma:
        response_ntrips = (df['n_trips']).apply(np.log).sub(df['rolling_n_trips'], axis=0)
        response_duration = (df['avg_duration']).apply(np.log).sub(df['rolling_avg_duration'], axis=0)
        response_distance = (df['avg_distance']).apply(np.log).sub(df['rolling_avg_distance'], axis=0)
        response_fare = (df['avg_fare']).apply(np.log).sub(df['rolling_avg_fare'], axis=0)
        response_speed = (df['avg_speed']).apply(np.log).sub(df['rolling_avg_speed'], axis=0)
    else:
        response_ntrips = (df['n_trips']).apply(np.log)
        response_duration = (df['avg_duration']).apply(np.log)
        response_distance = (df['avg_distance']).apply(np.log)
        response_fare = (df['avg_fare']).apply(np.log)
        response_speed = (df['avg_speed']).apply(np.log)

    # Create predictors
    predictors_ntrips_ind = df[['rain_ind', 'snow_ind', 'rolling_n_trips',
                               'day_of_week_0', 'day_of_week_1', 'day_of_week_2',
                               'day_of_week_3','day_of_week_4', 'day_of_week_5',
                               'day_of_week_6']].copy()

    predictors_duration_ind = df[['rain_ind', 'snow_ind', 'rolling_avg_duration',
                                'day_of_week_0', 'day_of_week_1', 'day_of_week_2',
                                'day_of_week_3', 'day_of_week_4', 'day_of_week_5',
                                'day_of_week_6']].copy()

    predictors_distance_ind = df[['rain_ind', 'snow_ind', 'rolling_avg_distance',
                                  'day_of_week_0', 'day_of_week_1', 'day_of_week_2',
                                  'day_of_week_3', 'day_of_week_4', 'day_of_week_5',
                                  'day_of_week_6']].copy()

    predictors_fare_ind = df[['rain_ind', 'snow_ind', 'rolling_avg_fare',
                                  'day_of_week_0', 'day_of_week_1', 'day_of_week_2',
                                  'day_of_week_3', 'day_of_week_4', 'day_of_week_5',
                                  'day_of_week_6']].copy()

    predictors_speed_ind = df[['rain_ind', 'snow_ind', 'rolling_avg_speed',
                                  'day_of_week_0', 'day_of_week_1', 'day_of_week_2',
                                  'day_of_week_3', 'day_of_week_4', 'day_of_week_5',
                                  'day_of_week_6']].copy()

    indices = ['n_trips', 'duration', 'distance', 'fare', 'speed']

    cols = ['rain_ind', 'snow_ind', 'rolling_generic',
            'day_of_week_0', 'day_of_week_1', 'day_of_week_2',
            'day_of_week_3', 'day_of_week_4', 'day_of_week_5',
            'day_of_week_6', 'intercept', 'R^2', 'MSE']

    if force_gamma:
        predictors_ntrips_ind = predictors_ntrips_ind.drop(['rolling_n_trips'], axis=1)
        predictors_duration_ind = predictors_duration_ind.drop(['rolling_avg_duration'], axis=1)
        predictors_distance_ind = predictors_distance_ind.drop(['rolling_avg_distance'], axis=1)
        predictors_fare_ind = predictors_fare_ind.drop(['rolling_avg_fare'], axis=1)
        predictors_speed_ind = predictors_speed_ind.drop(['rolling_avg_speed'], axis=1)
        cols.remove('rolling_generic')

    if drop_monday:
        predictors_ntrips_ind = predictors_ntrips_ind.drop(['day_of_week_0'], axis=1)
        predictors_duration_ind = predictors_duration_ind.drop(['day_of_week_0'], axis=1)
        predictors_distance_ind = predictors_distance_ind.drop(['day_of_week_0'], axis=1)
        predictors_fare_ind = predictors_fare_ind.drop(['day_of_week_0'], axis=1)
        predictors_speed_ind = predictors_speed_ind.drop(['day_of_week_0'], axis=1)
        cols.remove('day_of_week_0')

    # Run regressions
    lr_ntrips = LinearRegression(fit_intercept=intercept)
    lr_ntrips.fit(predictors_ntrips_ind, response_ntrips)

    lr_duration = LinearRegression(fit_intercept=intercept)
    lr_duration.fit(predictors_duration_ind, response_duration)

    lr_distance = LinearRegression(fit_intercept=intercept)
    lr_distance.fit(predictors_distance_ind, response_distance)

    lr_fare = LinearRegression(fit_intercept=intercept)
    lr_fare.fit(predictors_fare_ind, response_fare)

    lr_speed = LinearRegression(fit_intercept=intercept)
    lr_speed.fit(predictors_speed_ind, response_speed)


    # Create and populate the outgoing dataframe
    if predictions:
        cols = ['n_trips_forecast', 'duration_forecast', 'distance_forecast', 'fare_forecast', 'speed_forecast']

        # If we force gamma we need some the generic values as well as unchanged predictions for later calculations
        if force_gamma:
            lr_ntrips_pred = lr_ntrips.predict(predictors_ntrips_ind)
            lr_duration_pred = lr_duration.predict(predictors_duration_ind)
            lr_distance_pred = lr_distance.predict(predictors_distance_ind)
            lr_fare_pred = lr_fare.predict(predictors_fare_ind)
            lr_speed_pred = lr_speed.predict(predictors_speed_ind)
            dfout = df[['date', 'n_trips', 'avg_duration', 'avg_distance', 'avg_fare', 'avg_speed',
                        'rolling_n_trips', 'rolling_avg_duration', 'rolling_avg_distance',
                        'rolling_avg_fare', 'rolling_avg_speed']].copy()
        else:
            lr_ntrips_pred = [math.exp(x) for x in lr_ntrips.predict(predictors_ntrips_ind)]
            lr_duration_pred = [math.exp(x) for x in lr_duration.predict(predictors_duration_ind)]
            lr_distance_pred = [math.exp(x) for x in lr_distance.predict(predictors_distance_ind)]
            lr_fare_pred = [math.exp(x) for x in lr_fare.predict(predictors_fare_ind)]
            lr_speed_pred = [math.exp(x) for x in lr_speed.predict(predictors_speed_ind)]
            dfout = df[['date', 'n_trips', 'avg_duration', 'avg_distance', 'avg_fare', 'avg_speed']].copy()

        collection = [lr_ntrips_pred, lr_duration_pred, lr_distance_pred, lr_fare_pred, lr_speed_pred]
        dfout.set_index('date', inplace=True)

        # Add predictions to dataframe
        for col, pred in zip(cols, collection):
            dfout[col] = np.asarray(pred)

        # If gamma is forced, predictions are technically the difference of logged forecast and logged generic, so
        # here we must retrieve them
        if force_gamma:
            generics = ['rolling_n_trips', 'rolling_avg_duration', 'rolling_avg_distance',
                        'rolling_avg_fare', 'rolling_avg_speed']
            for forecast, generic in zip(cols, generics):
                dfout[forecast] = ((dfout[generic]).add(dfout[forecast], axis=0)).apply(np.exp)

            dfout.drop(['rolling_n_trips', 'rolling_avg_duration', 'rolling_avg_distance',
                        'rolling_avg_fare', 'rolling_avg_speed'], axis=1, inplace=True)

    else:
        # Modify output lists to include R^2 and MSE values
        coefs_ntrips = np.append([math.exp(x) for x in lr_ntrips.coef_], [math.exp(lr_ntrips.intercept_),
                                  lr_ntrips.score(predictors_ntrips_ind, response_ntrips),
                                  np.mean((lr_ntrips.predict(predictors_ntrips_ind) - response_ntrips) ** 2)])

        coefs_duration = np.append([math.exp(x) for x in lr_duration.coef_], [math.exp(lr_duration.intercept_),
                                    lr_duration.score(predictors_duration_ind, response_duration),
                                    np.mean((lr_duration.predict(predictors_duration_ind) - response_duration) ** 2)])

        coefs_distance = np.append([math.exp(x) for x in lr_distance.coef_], [math.exp(lr_distance.intercept_),
                                   lr_distance.score(predictors_distance_ind, response_distance),
                                    np.mean((lr_distance.predict(predictors_distance_ind) - response_distance) ** 2)])

        coefs_fare = np.append([math.exp(x) for x in lr_fare.coef_], [math.exp(lr_fare.intercept_),
                                lr_fare.score(predictors_fare_ind, response_fare),
                                np.mean((lr_fare.predict(predictors_fare_ind) - response_fare) ** 2)])

        coefs_speed = np.append([math.exp(x) for x in lr_speed.coef_], [math.exp(lr_speed.intercept_),
                                 lr_speed.score(predictors_speed_ind, response_speed),
                                 np.mean((lr_speed.predict(predictors_speed_ind) - response_speed) ** 2)])
        collection = [coefs_ntrips, coefs_duration, coefs_distance, coefs_fare, coefs_speed]
        dfout = pd.DataFrame(index=indices, columns=cols)
        for y in range(len(indices)):
            for x in range(len(cols)):
                dfout.loc[indices[y], cols[x]] = collection[y][x]

    return dfout


def get_dummies(data):
    # drop January 1-10, since they don't have rolling averages
    data = data.dropna(subset=['rolling_n_trips', 'rolling_avg_duration', 'rolling_avg_distance',
                               'rolling_avg_fare', 'rolling_avg_speed'])

    # set rolling averages to 0.01 if 0
    data['rolling_n_trips'][data['rolling_n_trips'] == 0] = 0.001
    data['rolling_avg_duration'][data['rolling_avg_duration'] == 0] = 0.001
    data['rolling_avg_distance'][data['rolling_avg_distance'] == 0] = 0.001
    data['rolling_avg_fare'][data['rolling_avg_fare'] == 0] = 0.001
    data['rolling_avg_speed'][data['rolling_avg_speed'] == 0] = 0.001

    # add column for day of week
    month = (data['date'] // 100).tolist()
    day = (data['date'] % 100).tolist()
    year = 2015

    daysofweek = []

    for i in range(len(month)):
        d = datetime.datetime(year, month[i], day[i], 1, 1, 1)
        daysofweek.append(d.weekday())

    data['day_of_week'] = daysofweek

    data_dummies = pd.get_dummies(data, columns=['day_of_week'])

    data_dummies['n_trips'][data_dummies['n_trips'] == 0] = 0.001
    data_dummies['avg_duration'][data_dummies['avg_duration'] == 0] = 0.001
    data_dummies['avg_distance'][data_dummies['avg_distance'] == 0] = 0.001
    data_dummies['avg_fare'][data_dummies['avg_fare'] == 0] = 0.001
    data_dummies['avg_speed'][data_dummies['avg_speed'] == 0] = 0.001

    data_dummies['rolling_n_trips'] = (data_dummies['rolling_n_trips']).apply(math.log)
    data_dummies['rolling_avg_duration'] = (data_dummies['rolling_avg_duration']).apply(math.log)
    data_dummies['rolling_avg_distance'] = (data_dummies['rolling_avg_distance']).apply(math.log)
    data_dummies['rolling_avg_fare'] = (data_dummies['rolling_avg_fare']).apply(math.log)
    data_dummies['rolling_avg_speed'] = (data_dummies['rolling_avg_speed']).apply(math.log)

    return data_dummies


def coeff(data_dummies, drop_monday, force_gamma, intercept):
    lr_coeff = data_dummies.groupby(['pickup_zone', 'dropoff_zone', 'interval_start']).apply(
        ijt_regression, drop_monday=drop_monday, force_gamma=force_gamma, predictions=False, intercept=intercept)
    return lr_coeff


def predict(data_dummies, drop_monday, force_gamma, intercept):
    lr_preds = data_dummies.groupby(['pickup_zone', 'dropoff_zone', 'interval_start']).apply(
        ijt_regression, drop_monday=drop_monday, force_gamma=force_gamma, predictions=True, intercept=intercept)
    lr_preds.drop_duplicates(inplace=True)

    df = lr_preds.reset_index().set_index(['date', 'interval_start', 'pickup_zone', 'dropoff_zone']).sort_index()

    df.drop_duplicates(inplace=True)
    return df


def analyze(preds):
    preds = preds.astype(float)
    preds = preds.groupby(level=3)
    print('mean, min, max summary tables:')
    print(preds.mean())
    print(preds.min())
    print(preds.max())
