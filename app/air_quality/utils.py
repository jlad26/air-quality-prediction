"""Utilities for the airquality package.
"""

import pandas as pd
from air_quality.dataset import Dataset
from air_quality.timeseriesset import TimeSeriesSet


def get_sliced_datasets(loaded_datasets, start = ''):
    """
    Slices each loaded dataset so that the timeseries begin at the datetime
    specified by `start`. Returns the full dataset if start is an empty string.

    Args:
        loaded_datasets: A dictionary of datasets with keys as dataset names.
        start:
            A string representing the datetime from which all timeseries should
            start. If empty string, then use the timeseries start time.

    Returns:
        A dictionary with the sliced timeseries.
    """

    if start == '':
        return loaded_datasets

    output = {}

    data_start_timestamp = pd.to_datetime(start)

    for ds_name, ds in loaded_datasets.items():

        reduced_ds = {}

        for key, ts in ds.items():

            # Check that the start time is valid.
            if not ts.is_within_range(data_start_timestamp):
                raise ValueError(f"{start} is not in the range of the time series {ds} - {key}")

            # Split if we need to, or use the whole dataset.
            if data_start_timestamp == ts.start_time():
                reduced_ds[key] = ts
            else:
                _, reduced_ds[key] = ts.split_before(data_start_timestamp)

        output[ds_name] = reduced_ds

    return output


def get_model_dataset(datasets, data_choices):
    """Gets a dataset instance ready for input to
    a Darts model wrapper.

    Args:
        datasets: A dictionary of datasets with keys of 'air quality, 'historical_weather' and
        'weather_forecast.

        data_choices: A dictionary in the form {
            'training_type' : one of 'VAL', 'TEST' or 'PROD',
            'forecast_pollutants' : list of strings representing pollutants,
            'covariates_types' : a list of covariate types from the choice of 'past' and 'future',
            'feature_covariates' : a list of feature types from the choice of 'time' and 'data'
        }

    Returns:
        A Dataset instance
    """

    return Dataset(
        TimeSeriesSet(datasets['air_quality']),
        TimeSeriesSet(datasets['historical_weather']),
        TimeSeriesSet(datasets['weather_forecast']),
        data_choices
    )

def get_ts_first_last(ts):
    return ts.pd_dataframe().iloc[[0, -1]]
