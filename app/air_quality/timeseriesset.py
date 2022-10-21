"""Module for the TimeSeriesSet class.
"""

import numpy as np

class TimeSeriesSet:
    """A class to create the various timeseries required for training, validation
    and testing.

    Attributes
    ----------
    scalers: Dictionary of (Key, Scaler)

    ts:
        A dictionary that splits the arg `timeseries_dict` into various training /
        validation / production timeseries corresponding to the following keys: 'train',
        'val', 'train_val', 'val_test', 'test', 'train_val_test', 'entire'.

    ts_names: A list of strings being the keys of the arg `timeseries_dict`.
    """

    def __init__(
        self,
        timeseries_dict,
        split_train_valtest = 0.8,
        split_val_test = 0.5
    ):

        self.ts = {}
        self.ts_names = list(timeseries_dict.keys())

        ts_types = [
            'train',
            'val',
            'train_val',
            'val_test',
            'test',
            'train_val_test',
            'entire'
        ]

        for ts_type in ts_types:
            self.ts[ts_type] = {}

        # Split into train, val and test sets and the required combinations
        # of train_val and train_val_test.
        for key, ts in timeseries_dict.items():

            # Set to float32 for improved performance.
            ts = ts.astype(np.float32)

            self.ts['train_val_test'][key] = ts
            self.ts['train'][key], self.ts['val_test'][key] = ts.split_after(split_train_valtest)
            self.ts['val'][key], self.ts['test'][key] = \
                self.ts['val_test'][key].split_after(split_val_test)
            self.ts['train_val'][key] = self.ts['train'][key].concatenate(self.ts['val'][key])
            self.ts['entire'][key] = ts


    def get_ts_sequence(
        self,
        ts_type : str,
        subset : list = None,
        end_time = None
    ):
        """Gets a list of timeseries for the given timeseries type.

        Args:
            ts_type: String matching one of the timeseries types.
            subset: List of strings corresponding to which elements to include.
            end_time: Pandas timestamp representing point after which data should be excluded.

        Returns:
            List of Darts timeseries.
        """

        ts_names = subset if subset else self.ts_names
        ts_sequence = []
        for ts_name in ts_names:
            if ts_type == 'none':
                ts = None
            else:
                ts = self.ts[ts_type][ts_name]
                if end_time is not None:
                    if ts.is_within_range(end_time) and end_time != ts.end_time():
                        ts, _ = ts.split_after(end_time)
            ts_sequence.append(ts)

        return ts_sequence


    def start_time(self, ts_type : str):
        """Gets the start time for the Darts timeseries of the provided type.

        Args:
            ts_type: String of the timeseries type (e.g., 'train', 'train_val').

        Returns:
            Pandas timestamp of the start time.
        """

        timeseries = self.ts[ts_type].values()
        return list(timeseries)[0].start_time()


    def end_time(self, ts_type : str):
        """Gets the end time for the Darts timeseries of the provided type.

        Args:
            ts_type: String of the timeseries type (e.g., 'train', 'train_val').

        Returns:
            Pandas timestamp of the end time.
        """

        timeseries = self.ts[ts_type].values()
        return list(timeseries)[0].end_time()
