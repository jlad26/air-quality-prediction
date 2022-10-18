"""Module for producing performance metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import pickle
import shutil
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from air_quality.prediction import PollutantPredictor
import air_quality.constants as C
from air_quality.prediction import get_key_times


class MetricsManager:

    POLLUTANTS = ['NO2', 'O3', 'PM10', 'PM2.5', 'SO2']

    EXISTING_DATA_DIR = os.path.join(C.WORK_DIR, 'data', 'metrics')
    EXISTING_DATA_PATH = os.path.join(EXISTING_DATA_DIR, 'metrics.pkl')
    METRICS_PLOTS_ABS_DIR = os.path.join(os.getcwd(), 'static', 'metrics-plots')

    PLOT_COLORS = {
        'train' : '#2161bc',
        'validation' : '#5dade2',
        'untrained_actual' : '#1e8449',
        'forecast' : '#ffbb05'
    }

    def __init__(self):

        self.existing_data = None

        self.key_times = get_key_times()

        if not os.path.exists(self.EXISTING_DATA_DIR):
            os.makedirs(self.EXISTING_DATA_DIR)

        if not os.path.exists(self.METRICS_PLOTS_ABS_DIR):
            os.makedirs(self.METRICS_PLOTS_ABS_DIR)


    def get_metrics_df(
        self,
        start_day : str,
        end_day : str,
        days_per_forecast = 1,
        stride_days = 1
        ):

        metric_dfs = []

        for pollutant in self.POLLUTANTS:

            predictor = PollutantPredictor(pollutant)
            metric_df = predictor.get_historical_metrics(
                start_day,
                end_day,
                days_per_forecast = days_per_forecast,
                stride_days = stride_days
            )
            metric_df['pollutant'] = pollutant
            metric_dfs.append(metric_df)

        output = pd.concat(metric_dfs).reset_index(drop = True)

        return output


    def save_metrics(self, metrics_df):
        self.existing_data = metrics_df
        with open(self.EXISTING_DATA_PATH, 'wb') as file:
            pickle.dump(metrics_df, file)


    def get_existing_data(self):
        if self.existing_data is None:
            if not os.path.exists(self.EXISTING_DATA_PATH):
                return None
            with open(f"{self.EXISTING_DATA_PATH}", 'rb') as file:
                self.existing_data = pickle.load(file)
        return self.existing_data.copy()


    def merge_and_save_new_metrics(self, new_metrics_df):
        existing_metrics = self.get_existing_data()
        merged = pd.concat([existing_metrics, new_metrics_df]).sort_values(
            by = ['pollutant', 'start_date'])
        merged.drop_duplicates(
            subset = ['start_date', 'pollutant'], keep = 'first', inplace = True)
        self.save_metrics(merged)


    def get_required_metrics_last_day(self):
        return self.key_times['last_forecast_start'] - pd.Timedelta(1, 'day')


    def get_required_metrics_first_day(self):
        existing_data = self.get_existing_data()
        if existing_data is None:
            return pd.to_datetime('2022-01-01').date()
        return (existing_data['start_date'].max() + pd.Timedelta(1, 'day')).date()


    def get_required_metrics(self):
        first_day = self.get_required_metrics_first_day()
        last_day = self.get_required_metrics_last_day()
        if first_day > last_day:
            return None

        metrics_df = self.get_metrics_df(
            start_day = first_day.strftime('%Y-%m-%d'),
            end_day = last_day.strftime('%Y-%m-%d')
        )

        return metrics_df


    def get_plot_urls(self):
        if not os.path.exists(self.METRICS_PLOTS_ABS_DIR):
            return None

        urls = {}
        url_base = "/static/metrics-plots"
        filenames = os.listdir(self.METRICS_PLOTS_ABS_DIR)
        for filename in filenames:
            filename_no_ext = filename.replace('.png', '')
            name_components = filename.split('_')
            key = 'url_' + filename_no_ext.replace(name_components[0] + '_', '')
            urls[key] = f"{url_base}/{filename}"
        return urls


    def update_metrics(self):

        metrics_df = self.get_required_metrics()

        if metrics_df is not None:
            self.merge_and_save_new_metrics(metrics_df)

        plot_data = self.generate_daily_metrics_plot_data()

        self.generate_metrics_plots(
            plot_data,
            metrics = ['mae'],
            base_filepath = self.METRICS_PLOTS_ABS_DIR
        )



    def generate_daily_metrics_plot_data(
        self,
        start_days_before_train_end = 30
    ):

        val_end_date = self.key_times['validation_end_time'].date()
        start_date = pd.to_datetime(val_end_date) - pd.Timedelta(start_days_before_train_end, 'days')

        metrics = self.get_existing_data()

        sliced_metrics = metrics[metrics['start_date'] >= start_date]

        # We also produce normalized data for the MAE data for comparison.
        pollutant_dfs = []
        for pollutant in self.POLLUTANTS:
            pollutant_df = sliced_metrics[sliced_metrics['pollutant'] == pollutant].copy()
            pollutant_df['norm_mae'] = (
                (pollutant_df['mae'] - pollutant_df['mae'].min()) /
                (pollutant_df['mae'].max() - pollutant_df['mae'].min())
            )

            pollutant_dfs.append(pollutant_df)

        norm_metrics = (
            pd.concat(pollutant_dfs)
            .reset_index(drop = True)
        )

        plot_data = {}
        metrics = ['mape', 'mae', 'norm_mae']
        for metric in metrics:
            drop_cols = [m for m in metrics if m != metric]
            plot_data[metric] = (
                norm_metrics
                .drop(columns = drop_cols)
                .pivot_table(
                    index = ['start_date'],
                    columns = ['pollutant'],
                    values = metric
                )
            )

        return plot_data


    def generate_metrics_plots(
        self,
        plot_data,
        metrics = ['mape', 'norm_mae', 'mae'],
        base_filepath = '',
        rolling_average_window = 7
    ):

        titles = {
            'mae' : 'MAE',
            'mape' : 'MAPE',
            'norm_mae' : 'Normalised MAE'
        }

        val_last_date = pd.to_datetime(self.key_times['validation_end_time'].date())

        # TODO better way of managing plot files = this way there is the possiblity of files
        # not being available while they are being created.
        if base_filepath:
            if os.path.exists(self.METRICS_PLOTS_ABS_DIR):
                shutil.rmtree(self.METRICS_PLOTS_ABS_DIR)
            os.makedirs(self.METRICS_PLOTS_ABS_DIR)

        legend_labels = {
            'en' : {
                'val' : 'Validation',
                'after_val' : 'Post validation',
                'val_rolling' : 'Validation 7 day rolling average',
                'after_val_rolling' : 'Post validation 7 day rolling average'
            },
            'fr' : {
                'val' : 'Validation',
                'after_val' : 'Après validation',
                'val_rolling' : 'Validation moyenne mobile à 7 jours',
                'after_val_rolling' : 'Après validation moyenne mobile à 7 jours'
            }
        }

        for size in ['narrow', 'wide']:

            font_size = 24 if size == 'narrow' else 20
            figsize = (8, 30) if size == 'narrow' else (16, 30)
            hspace = 0.65 if size == 'narrow' else 0.45
            top = 0.9 if size == 'narrow' else 0.94
            ncol = 1 if size == 'narrow' else 2

            for lang in ['en', 'fr']:

                for metric, data in plot_data.items():

                    if metric not in metrics:
                        continue

                    plt.rcParams['font.size'] = font_size
                    fig, axs = plt.subplots(5, 1, figsize = figsize)
                    plt_num = 0

                    if rolling_average_window:
                        rolling = data.rolling(rolling_average_window).mean()

                    for pollutant in self.POLLUTANTS:

                        val_data = data[data.index <= val_last_date]
                        after_val_data = data[data.index > val_last_date]

                        # Validation MAE
                        axs[plt_num].plot(
                            val_data.index,
                            val_data[pollutant],
                            c = self.PLOT_COLORS['validation'],
                            ls = 'dotted',
                            label = legend_labels[lang]['val']
                        )

                        # Untrained actual MAE
                        axs[plt_num].plot(
                            after_val_data.index,
                            after_val_data[pollutant],
                            c = self.PLOT_COLORS['untrained_actual'],
                            ls = 'dotted',
                            label = legend_labels[lang]['after_val']
                        )

                        if rolling_average_window:

                            val_rolling_data = rolling[rolling.index <= val_last_date]
                            after_val_rolling = rolling[rolling.index > val_last_date]

                            # Validation rolling MAE
                            axs[plt_num].plot(
                                val_rolling_data.index,
                                val_rolling_data[pollutant],
                                c = self.PLOT_COLORS['validation'],
                                label = legend_labels[lang]['val_rolling']
                            )

                            # Untrained actual rolling MAE
                            axs[plt_num].plot(
                                after_val_rolling.index,
                                after_val_rolling[pollutant],
                                c = self.PLOT_COLORS['untrained_actual'],
                                label = legend_labels[lang]['after_val_rolling']
                            )

                        axs[plt_num].xaxis.set_major_locator(mdates.DayLocator(interval=14))
                        axs[plt_num].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        axs[plt_num].set_title(f"{titles[metric]}: {pollutant}")

                        plt.setp(axs[plt_num].get_xticklabels(), rotation = 30, ha = 'right')
                        plt.tight_layout()

                        plt_num += 1

                    # Convert all legend labels into a dictionary so as to remove duplicates
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    fig.legend(
                        by_label.values(),
                        by_label.keys(),
                        loc = 'upper center',
                        ncol = ncol,
                    )

                    plt.subplots_adjust(hspace = hspace, top = top)

                    if base_filepath:
                        existing_data = self.get_existing_data()
                        existing_data_last_date = existing_data['start_date'].max().date()
                        filename = f"{existing_data_last_date}_{metric}_{lang}_{size}.png"
                        filepath = os.path.join(self.METRICS_PLOTS_ABS_DIR, filename)
                        plt.savefig(filepath)
                        plt.close()
                    else:
                        plt.show()
