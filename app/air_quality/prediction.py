"""Module for Prediction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import shutil
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from air_quality.model import Model
import air_quality.data_management as dm
import air_quality.constants as C
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape, mae

class PollutantPredictor:
    """For predicting concentrations of a given pollutant.

    Uses trained models, instances of air_quality.model.Model, to produce predictions and
    metrics as data and in plot form.

    Note that class assumes that there is a separate model for each pollutant rather than a
    single model trained on all pollutants.

    Attributes:
        PREDICTION_MODELS_DIR: String of path to the folder where the trained models are located.
        IMAGE_PLOTS_ABS_DIR: String of path to directory where plots are saved.
        PLOT_CACHE_SIZE: Size of cache (for caching image plots). Once exceeded, the folder is
            emptied.
        pollutant: String representing pollutant to be predicted.
        model: Instance of trained air_quality.model.Model corresponding to this pollutant.
        darts_model: Trained Darts model to be used for prediction.
        train_series_unscaled: Darts timeseries of the unscaled target series used for training.
        val_series_unscaled: Darts timeseries of the unscaled target series used for validation.
        target_series_train_val_unscaled: Darts timeseries of the unscaled target series covering
            both training and validation.
        target_series_unscaled: Darts timeseries of the entirety of the unscaled target series
            (i.e., including training, validation, and untrained actual data).
        target_scaler: Scaler instance used to scale target series during training.
        target_series: Darts timeseries of the entirety of the scaled target series (i.e.,
            including training, validation, and untrained actual data).
        historical_weather_ts: Darts timeseries of the historical weather data for this
            pollutant.
        weather_forecast_ts: Darts timeseries of the weather forecast data for this
            pollutant.
        past_covariates: Darts timeseries of the past covariates for this pollutant (i.e.,
            weather data and seasonal time data).
        future_covariates: Darts timeseries of the future covariates for this pollutant (i.e.,
            weather data and seasonal time data).
    """

    PREDICTION_MODELS_DIR = os.path.join(os.getcwd(), 'prediction_models')

    IMAGE_PLOTS_ABS_DIR = os.path.join(os.getcwd(), 'static', 'image-plots')

    PLOT_CACHE_SIZE = 100

    def __init__(self, pollutant : str):

        # Load the right model.
        self.pollutant = pollutant
        self.model = Model.load_model(os.path.join(self.PREDICTION_MODELS_DIR, pollutant))
        self.darts_model = self.model.get_best_darts_model()

        # Set the series.
        self.train_series_unscaled = self.model.target_series_unscaled[0]
        self.val_series_unscaled = self.model.val_series_unscaled[0]
        self.target_series_train_val_unscaled = self.model.target_series_train_val_unscaled[0]
        self.target_series_unscaled = self._get_unscaled_target_series(pollutant)
        self.target_scaler = self.model.target_scalers[0]
        self.target_series = self._get_scaled_target_series()

        self.historical_weather_ts = self._get_historical_weather_ts(pollutant)
        self.weather_forecast_ts = self._get_weather_forecast_ts(pollutant)

        self.past_covariates = self._get_past_covariates()
        self.future_covariates = self._get_future_covariates()


    def _get_forecast_location(self, pollutant):
        return 'Marseille' if pollutant == 'SO2' else 'Montpellier'


    def _convert_existing_df_to_timeseries_dict(self, dataframe : pd.DataFrame):
        ts_categories = {}
        category_col_name = dataframe.columns[0]

        for category in dataframe[category_col_name].unique():
            category_data = (
                dataframe[dataframe[category_col_name] == category]
                .set_index('Datetime')
                .drop(columns = category_col_name)
            )
            ts_categories[category] = TimeSeries.from_dataframe(category_data)

        return ts_categories


    def _get_unscaled_target_series(self, pollutant : str):
        dm_air_quality = dm.AirQualityDataManager()
        existing_data_df = dm_air_quality.get_existing_data_df()
        ts_categories = self._convert_existing_df_to_timeseries_dict(existing_data_df)
        return ts_categories[pollutant].astype(np.float32)


    def _get_historical_weather_ts(self, pollutant):
        dm_historical_weather = dm.HistoricalWeatherDataManager()
        existing_data_df = dm_historical_weather.get_existing_data_df()
        ts_categories = self._convert_existing_df_to_timeseries_dict(existing_data_df)
        location = self._get_forecast_location(pollutant)
        return ts_categories[location].astype(np.float32)


    def _get_weather_forecast_ts(self, pollutant):
        dm_weather_forecast = dm.WeatherForecastDataManager()
        existing_data_df = dm_weather_forecast.get_existing_data_df()
        ts_categories = self._convert_existing_df_to_timeseries_dict(existing_data_df)
        location = self._get_forecast_location(pollutant)
        return ts_categories[location].astype(np.float32)


    def _get_scaled_target_series(self):
        return self.target_scaler.transform(self.target_series_unscaled)


    def _get_past_covariates(self):

        # Get the extra chunk of data we will use from forecast data.
        _, historical_extension = self.weather_forecast_ts.split_after(
            self.historical_weather_ts.end_time()
        )

        # Reformat and rename so we have the same data columns as historical.
        data_column_mapping = {
            'DewPointC' : 'dwpt',
            'pressure' : 'pres',
            'humidity' : 'rhum',
            'tempC' : 'temp',
            'winddirDegree' : 'wdir',
            'windspeedKmph' : 'wspd'
        }

        historical_extension_df = (
            historical_extension.pd_dataframe()
            .rename(columns = data_column_mapping)
            .drop(columns = 'cloudcover')
        )

        # Combine historical with the extensions as a dataframe and convert to timeseries.
        historical_extended_df = pd.concat(
            [self.historical_weather_ts.pd_dataframe(), historical_extension_df])
        data_covariates_unscaled = TimeSeries.from_dataframe(historical_extended_df).astype(np.float32)

        # Add on time covariates.
        time_covariates = self._get_time_covariates(data_covariates_unscaled)
        unscaled_covariates = data_covariates_unscaled.concatenate(
            time_covariates, axis = 1
        )

        # Note we are using the past scaler even though some of our data is weather
        # forecast data. This is correct because all weather forecast data is in the
        # same units as historical data and we need to use the same scaler as was
        # trained on for past covariates.
        scaler = self.model.covariates_scalers['past'][0]

        return scaler.transform(unscaled_covariates)


    def _get_future_covariates(self):
        time_covariates = self._get_time_covariates(self.weather_forecast_ts)
        unscaled_covariates = self.weather_forecast_ts.concatenate(time_covariates, axis = 1)
        scaler = self.model.covariates_scalers['future'][0]
        return scaler.transform(unscaled_covariates)


    def _get_time_covariates(self, ts):
        """Returns unscaled time covariates of hour, day of week and month
        for the given timseries.
        """

        time_index = ts.time_index
        ts_time_features = datetime_attribute_timeseries(
            time_index, attribute="hour", one_hot=False)
        ts_time_features = ts_time_features.stack(
            datetime_attribute_timeseries(time_index, attribute="day_of_week", one_hot=False))
        ts_time_features = ts_time_features.stack(
            datetime_attribute_timeseries(time_index, attribute="month", one_hot=False))

        return ts_time_features.astype(np.float32)


    def _get_prediction_forecast_horizon(self, target_series, end_time):
        """Returns the prediction forecast horizon as the number of time
        steps between the end of the target series and the end time.
        """

        if end_time < self.future_covariates.end_time():
            future_covs, _ = self.future_covariates.split_after(end_time)
        else:
            future_covs = self.future_covariates

        # Drop from future covariates anything that overlaps with the target series.
        _, gap_ts = future_covs.split_after(target_series.end_time())
        return gap_ts.n_timesteps


    def _get_predicted_series(self, start : pd.Timestamp, end : pd.Timestamp = None):

        if self.target_series.is_within_range(start):
            target_series = self.target_series.drop_after(start)
        else:
            target_series = self.target_series.copy()

        end_time = self.future_covariates.end_time()
        if end is not None:
            if end < self.future_covariates.end_time():
                end_time = end

        forecast_horizon = self._get_prediction_forecast_horizon(target_series, end_time)

        pred_series = self.model.get_predicted_series(
            darts_model = self.darts_model,
            series = target_series,
            forecast_horizon = forecast_horizon,
            past_covariates = self.past_covariates,
            future_covariates = self.future_covariates
        )

        return pred_series


    def _get_predicted_series_for_dates(self, start_day : str, end_day : str = ''):

        start = pd.to_datetime(start_day)
        if end_day:
            end = pd.to_datetime(end_day) + pd.Timedelta(23, 'hours')
        else:
            end = None

        return self._get_predicted_series(start, end)


    def _get_historical_forecasts(
        self,
        start_day : str,
        end_day : str,
        days_per_forecast = 1,
        stride_days = 1
    ):

        current_forecast_start_day = pd.to_datetime(start_day).date()

        forecasts, start_dates = [], []

        while current_forecast_start_day <= pd.to_datetime(end_day).date():

            start_dates.append(current_forecast_start_day)

            current_forecast_start_day_str = current_forecast_start_day.strftime('%Y-%m-%d')

            current_forecast_end_day = current_forecast_start_day + pd.Timedelta(days_per_forecast - 1, 'day')
            current_forecast_end_day_str = current_forecast_end_day.strftime('%Y-%m-%d')

            # Generate the forecast.
            forecasts.append(self._get_predicted_series_for_dates(
                current_forecast_start_day_str, current_forecast_end_day_str))

            # Move on by the stride amount.
            current_forecast_start_day += pd.Timedelta(stride_days, 'days')

        return forecasts, start_dates


    def get_historical_metrics(
        self,
        start_day : str,
        end_day : str,
        days_per_forecast = 1,
        stride_days = 1
    ):
        """Gets metrics for historical forecasts.

        Args:
            start_day: String of start date for forecasts in form yyyy-mm-dd.
            end_day: String of end date for forecasts in form yyyy-mm-dd.
            days_per_forecast: Integer of how many days to include per forecast.
            stride_days: Integer of how many days to advance from the start date of one forecast
            to the next forecast.

        Returns:
            Pandas dataframe of metrics with columns 'start_date', 'mape' and 'mae'.
        """

        forecasts, start_dates = self._get_historical_forecasts(
            start_day,
            end_day,
            days_per_forecast = days_per_forecast,
            stride_days = stride_days
        )

        metrics = {
            'start_date' : start_dates,
            'mape' : [],
            'mae' : [],
        }

        for forecast in forecasts:
            actuals = self.target_series_unscaled.slice_intersect(forecast)
            try:
                mape_value = mape(actuals, forecast)
            except ValueError:
                mape_value = 0
            metrics['mape'].append(mape_value)
            metrics['mae'].append(mae(actuals, forecast))

        output = pd.DataFrame(metrics)
        output['start_date'] = pd.to_datetime(output['start_date'])
        return output


    def _get_bounded_timeseries(self, ts, start, end):

        # Check whether we have an overlap at all.
        if (
            ts.end_time() < start or
            ts.start_time() > end
        ):
            return None

        start_index = ts.get_index_at_point(start) if ts.is_within_range(start) else 0
        end_index = ts.get_index_at_point(end) if ts.is_within_range(end) else len(ts) - 1

        return ts[start_index:(end_index + 1)]


    def _generate_plot_images(
        self,
        forecasts,
        show_actuals = True,
        files_dir = None,
        fig_params = {},
        titles = {},
        metrics_with_titles = True,
        languages = ['en', 'fr'],
    ):

        default_fig_params = {
            'layout' : 'tight'
        }

        fig_params = {**default_fig_params, **fig_params}

        for size in ['narrow', 'wide']:

            fig_params['figsize'] = (16, 8) if size == 'wide' else (8, 8)
            fontsize = 20 if size == 'wide' else 20
            rc = {'font' : {'size' : fontsize}}

            for lang in languages:

                img_type = f"{lang}-{size}"

                filepath = os.path.join(files_dir, f"{img_type}.png")

                self.plot_series(
                    forecasts,
                    show_actuals = show_actuals,
                    filepath = filepath,
                    fig_params = fig_params,
                    title = titles[img_type] if img_type in titles else '',
                    metrics_with_titles = metrics_with_titles,
                    lang = lang,
                    rc = rc,
                )


    def _get_pollutant_plot_dir(self, start, end, absolute_path = True):

        base_dir = self.IMAGE_PLOTS_ABS_DIR
        start_str = start.strftime('%Y%m%d-%H%M%S')
        end_str = end.strftime('%Y%m%d-%H%M%S')
        final_dir = f"pollutant={self.pollutant}_start={start_str}_end={end_str}"
        abs_filepath = os.path.join(base_dir, 'individual-pollutants', final_dir)

        if absolute_path:
            return abs_filepath

        return os.path.relpath(abs_filepath, os.getcwd())


    def _generate_pollutant_plot_images(
        self,
        forecasts,
        show_actuals = True,
        fig_params = {},
    ):

        forecasts_start, forecasts_end = self._get_bounding_times(forecasts)
        datefmt = '%-d %b %y'
        start_date = forecasts_start.strftime(datefmt)
        end_date = forecasts_end.strftime(datefmt)
        date_text = f"{start_date}" if start_date == end_date else f"{start_date} - {end_date}"

        wide_titles = {
            'en-wide' : f"Pollutant {self.pollutant}: Prediction {date_text}",
            'fr-wide' : f"Polluant {self.pollutant}: Prédiction {date_text}"
        }

        narrow_titles = {}
        for img_type, title in wide_titles.items():
            narrow_titles[img_type.replace('wide', 'narrow')] = title.replace(': ', '\n')

        files_dir = self._get_pollutant_plot_dir(forecasts_start, forecasts_end)

        self._generate_plot_images(
            forecasts,
            show_actuals = show_actuals,
            files_dir = files_dir,
            fig_params = fig_params,
            titles = {**wide_titles, **narrow_titles}
        )


    def plot_series(
        self,
        forecasts,
        show_actuals = True,
        filepath = None,
        fig_params = {
            'figsize' : (16, 8)
        },
        title = '',
        metrics_with_titles = True,
        lang = 'en',
        rc = {},
    ):
        """Plots provided forecasts series (against actuals if so desired and where possible).

        Plot is either displayed or saved.

        Args:
            forecasts: List of Darts timeseries representing forecasts.
            show_actuals: Boolean of whether to display actuals if possible.
            filepath: String of filepath for where to save file. If not provided, plot is
                displayed.
            fig_params: Dictionary of parameters to be passed to Matplotlib plt.figure.
            title: String of title to be displayed on plot.
            metrics_with_titles: Boolean of whether to append metrics to title.
            lang: String of 'en' or 'fr' for English or French.
            rc: Dictionary to be used by Matplotlib plt.rc() where dictionary keys are groups
                and values are parameters for the group.
        """

        for group, params in rc.items():
            plt.rc(group, **params)

        plt.figure(**fig_params)

        # If we have been given a single forecast timeseries, convert to a list.
        if not isinstance(forecasts, list):
            forecasts = [forecasts]

        if show_actuals:

            forecasts_start, forecasts_end = self._get_bounding_times(forecasts)

            # Get all the timeseries for plotting.
            trained_actual = self.train_series_unscaled
            plot_trained_actual = self._get_bounded_timeseries(
                trained_actual, forecasts_start, forecasts_end)

            val_actual = self.val_series_unscaled
            plot_val_actual = self._get_bounded_timeseries(
                val_actual, forecasts_start, forecasts_end)

            if self.target_series_unscaled.end_time() > val_actual.end_time():
                _, untrained_actual = self.target_series_unscaled.split_after(val_actual.end_time())
                plot_untrained_actual = self._get_bounded_timeseries(
                    untrained_actual, forecasts_start, forecasts_end)
            else:
                untrained_actual, plot_untrained_actual = None, None

            if plot_trained_actual is not None:
                if lang == 'en':
                    label = 'Actual trained'
                else:
                    label = 'Réelles entrainées'
                plot_trained_actual.plot(
                    label = label, c = '#21618C')

            if plot_val_actual is not None:
                if lang == 'en':
                    label = 'Actual validation'
                else:
                    label = 'Réelles validées'
                plot_val_actual.plot(
                    label = label, c = '#5DADE2')

            if plot_untrained_actual is not None:
                if lang == 'en':
                    label = 'Actual untrained'
                else:
                    label = 'Réelles non entrainées'
                plot_untrained_actual.plot(
                    label = label, c = '#1E8449')


        forecast_count = 1
        for forecast in forecasts:

            if len(forecasts) > 1:
                label_end = f" {forecast_count}: {forecast.start_time()} - {forecast.end_time()}"
            else:
                label_end = ''

            if lang == 'en':
                label = f"Forecast{label_end}"
            else:
                label = f"Prédiction{label_end}"
            forecast.plot(
                label = label, c = '#ffbb05')

            forecast_count += 1

        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_label_text('')
        ax.yaxis.set_label_text('μg / m3')

        if title:

            # Make one single combined actual timeseries for metric calcs.
            if show_actuals and metrics_with_titles:

                combined_actual = self.target_series_train_val_unscaled
                if untrained_actual:
                    combined_actual = combined_actual.concatenate(untrained_actual)

                # Make the combined actuals into a sequence to match the sequence of forecasts.
                # Only add in when there is an overlap between forecast and actual.
                overlapping_actuals, overlapping_forecasts = [], []
                for forecast in forecasts:
                    if combined_actual.end_time() >= forecast.start_time():
                        overlapping_actuals.append(combined_actual)
                        overlapping_forecasts.append(forecast)

                if overlapping_actuals:

                    try:
                        mape_value = f"{mape(overlapping_actuals, forecasts, inter_reduction = np.mean):.2f}%"
                    except ValueError:
                        mape_value = 'N/A'
                    mae_value = f"{mae(overlapping_actuals, forecasts, inter_reduction = np.mean):.2f}"

                    title += f"\nMAE: {mae_value} | MAPE: {mape_value}"

            plt.title(title)

        if filepath is None:
            plt.show()
        else:
            self._maybe_create_dir(filepath)
            plt.savefig(filepath)
            plt.close()


    def _maybe_create_dir(self, filepath):
        dir_path = os.path.dirname(filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    def _get_bounding_times(self, forecasts : list):

        start, end = None, None

        for forecast in forecasts:

            if start is None:
                start = forecast.start_time()
            else:
                if start > forecast.start_time():
                    start = forecast.start_time()

            if end is None:
                end = forecast.end_time()
            else:
                if end < forecast.end_time():
                    end = forecast.end_time()

        return start, end


    def _jsonify_images_dir(self, dir):
        return {'plots_dir' : dir}


    def get_prediction_result_dir(self, start_day, end_day):
        """Gets the url for the folder where plots are for prediction of the given time period.

        If the plots do not already exist, they are first generated. There are four plots for
        each prediction representing the two languages and two screen sizes.

        Args:
            start_day: String of data in form yyyy-mm-dd representing prediction start date.
            end_day: String of data in form yyyy-mm-dd representing prediction end date.

        Returns:
            Dictionary with key 'plot_dir' and value being a string of the url to the folders
            containing the plots.
        """

        self._maybe_clear_plots_cache()

        start = pd.to_datetime(start_day)
        end = pd.to_datetime(end_day) + pd.Timedelta(23, 'hours')

        # Check that dates are valid.
        key_times = get_key_times()
        if start.date() < key_times['earliest_forecast_start']:
            return {
                'error' : (
                    f"Start date of {start} is before earliest possible of {key_times['earliest_forecast_start']}"
                )
            }

        if start.date() > key_times['last_forecast_start']:
            return {
                'error' : (
                    f"Start date of {start} is after latest possible of {key_times['last_forecast_start']}"
                )
            }

        if end.date() > key_times['last_forecast_end']:
            return {
                'error' : (
                    f"End date of {start} is after latest possible of {key_times['last_forecast_end']}"
                )
            }

        # Check if files already exist.
        plots_abs_dir = self._get_pollutant_plot_dir(start, end)
        plots_rel_dir = '/' + self._get_pollutant_plot_dir(start, end, absolute_path = False)

        if not os.path.exists(plots_abs_dir):
            forecasts = self._get_predicted_series_for_dates(start_day, end_day)
            self._generate_pollutant_plot_images(forecasts)

        return self._jsonify_images_dir(plots_rel_dir)


    def _maybe_clear_plots_cache(self):
        """Removes the plots cache when we have more than 100 plots
        stored.
        """

        # TODO move this to parent class once there is one.
        if not os.path.exists(self.IMAGE_PLOTS_ABS_DIR):
            return None

        total_plots = 0

        # Gets the sub-folders which are the types of plots (e.g., individual-pollutants)
        dir_contents_list = os.listdir(self.IMAGE_PLOTS_ABS_DIR)

        # For each subfolder count up how many items are within (should be folders
        # containing the en.png and fr.png files) and add to the total.
        for dir in dir_contents_list:
            dir_path = os.path.join(self.IMAGE_PLOTS_ABS_DIR, dir)
            if os.path.isdir(dir_path):
                total_plots += len(os.listdir(dir_path))

        if total_plots >= self.PLOT_CACHE_SIZE:
            self.clear_plots_cache()


    def clear_plots_cache(self):
        """Clears the cache of all saved plot images."""
        shutil.rmtree(self.IMAGE_PLOTS_ABS_DIR)


def get_key_times():
    """Gets a dictionary of various key times.

    If the key times are not already found in the cache, the key times are calculated and then
    cached.

    Returns:
        Dictionary of key times where keys describe the particular key time and value is the Pandas
        timestamp.
    """

    updater = dm.DataUpdater()
    cached_key_times = updater.fetch_cached_key_times()
    if cached_key_times:
        return cached_key_times

    air_quality_dm = dm.AirQualityDataManager()
    weather_forecast_dm = dm.WeatherForecastDataManager()
    historical_weather_dm = dm.HistoricalWeatherDataManager()
    dmanager = dm.DataManager()

    # Load the NO2 model so we can get the training end time
    no2predict = PollutantPredictor('NO2')

    train_start_time = air_quality_dm.get_existing_data_df()['Datetime'].min()
    train_end_time = no2predict.model.train_data_end
    validation_end_time = no2predict.model.train_val_data_end
    actual_end_time = air_quality_dm.get_existing_data_last_time()

    air_quality_next_day = air_quality_dm.get_next_day_to_fetch()
    historical_weather_next_day = historical_weather_dm.get_next_day_to_fetch()

    last_forecast_start = min(air_quality_next_day, historical_weather_next_day)

    total_time = (actual_end_time - train_start_time).total_seconds()
    train_percent = (train_end_time - train_start_time).total_seconds() / total_time * 100
    validation_percent = (validation_end_time - train_end_time).total_seconds() / total_time * 100
    untrained_actual_percent = 100 - validation_percent - train_percent


    key_times = {
        'train_start_time' : train_start_time,
        'train_end_time' : train_end_time,
        'last_3_training' : train_end_time - pd.Timedelta(2, 'days'),
        'validation_end_time' : validation_end_time,
        'actual_end_time' : actual_end_time,
        'last_forecast_start' : last_forecast_start,
        'day_before_last_forecast_start' : last_forecast_start - pd.Timedelta(1, 'day'),
        'last_forecast_end' : weather_forecast_dm.get_last_date_current_data(),
        'earliest_forecast_start' : pd.to_datetime(dmanager.PREDICTION_EARLIEST_START).date(),
        'train_percent' : train_percent,
        'validation_percent' : validation_percent,
        'untrained_actual_percent' : untrained_actual_percent
    }

    updater.cache_key_times(key_times)

    return key_times