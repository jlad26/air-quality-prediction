"""Module for the Air Quality model, which is a wrapper to the models of the Darts library.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts.models import TransformerModel, TFTModel, TCNModel, NBEATSModel, BlockRNNModel
from darts.models import AutoARIMA, NaiveDrift, NaiveMean, LinearRegressionModel, RandomForest, ExponentialSmoothing
from darts.models.forecasting.torch_forecasting_model \
    import GlobalForecastingModel, TorchForecastingModel
from darts.metrics import mape, rmse, mae
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError
from torch import nn
import air_quality.constants as C

class Model:
    """Wrapper for a model from the Darts library.

    This is a wrapper for a model from the Darts library which allows
    a standardized usage of those models.

    Attributes:
        MODELS_DIR_PATH: String of the path to the directory holding the file
            'Models parameters.csv' that contains the parameters of models.
        models_attributes: Pandas dataframe containing parameters of models
            read from the file 'Models parameters.csv'.
        MODEL_CHOICE_PARAMS: List of parameters that are not set in the file 'Models
            parameters.csv' but are set on initialising an instance.
        init_params: Dictionary of arguments used when creating this instance of model.
        model_name: Model name as a string
        target_series_names: List of the names of the target series to be trained on.
        forecast_horizon: Integer number of time steps for which the model will predict.
        quantiles: List of floats representing the quantiles to be used in quantile regression.
        probabilistic_num_samples: Integer number of samples to be used in probabilistic
            forecasting.
        max_epochs: Integer maximum number of epochs for training.
        early_stopper_patience: Integer number of epochs to train for without improvement
            before stopping.
        log_tensorboard: Boolean of whether to log to tensorboard.
        target_series: List of the training components of the scaled target series, each
            a Darts timeseries.
        target_series_unscaled: List of the training components of the unscaled target series,
            each a Darts timeseries.
        val_series: List of the validation components of the scaled target series, each
            a Darts timeseries.
        val_series_unscaled: List of the training components of the unscaled target series,
            each a Darts timeseries.
        target_series_train_val: List of the combined training and validation components
            of the scaled target series, each a Darts timeseries.
        target_series_train_val_unscaled: List of the combined training and validation components
            of the unscaled target series, each a Darts timeseries.
        target_scalers: List of scalers that were used to scale the target series, in the
            same order as the lists of target series timeseries. In other words, the first
            scaler was used to scale the first in the list of target_series, and so on.
        train_val_data_start: Pandas timestamp of the start of validation.
        train_val_data_end: Pandas timestamp of the end of validation.
        train_data_end: Pandas timestamp of the end of training.
        additional_model_info: Dictionary of additional choices, being:
            training_type: String of either 'VAL', 'TEST', or 'PROD' to indicate training
                mode.
            covariates_types: List containing any one, both or neither of 'past' and 'future'.
            feature_covariates : List containing any one, both or neither of 'time' and 'data'.
        saved_model_name: A string concatenation of various model attributes.
        model_args: Dictionary of model arguments used to create the arguments for creating
            the Darts model instance.
        is_torch_forecasting_model: Boolean indicating whether model is a Torch forecasting
            model (as defined by the Darts library).
        is_global_forecasting_model: Boolean indicating whether model is a Global forecasting
            model (as defined by the Darts library).
        darts_model: A Darts model object.
        training_args: A dictionary of arguments used when training the model.
        past_covariates: List of Darts timeseries representing the past covariates.
            List is ordered to correpond with the order of target series.
        future_covariates: List of Darts timeseries representing the future covariates.
            List is ordered to correpond with the order of target series.
        covariates_scalers: A dictionary of scalers with two keys 'past' and 'future,
            each containing a list of scalers used to scale the covariates timeseries.
            The lists are ordered to correspond to the lists of covariate timeseries.
    """

    MODELS_DIR_PATH = os.path.join(C.WORK_DIR, 'Models parameters')

    models_attributes = pd.read_csv(
        os.path.join(MODELS_DIR_PATH, 'Models parameters.csv'),
        index_col = 0,
    )

    # Parameters not defined in models_attributes
    MODEL_CHOICE_PARAMS = [
        'target_series_names',
        'train_val_data_start',
        'train_val_data_end',
        'early_stopper_patience',
        'forecast_horizon',
        'max_epochs',
    ]

    def __init__(
        self,
        model_name,
        model_save_path,
        target_series_names = None,
        forecast_horizon = 1,
        max_epochs = 25,
        quantiles = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95],
        probabilistic_num_samples = 100,
        early_stopper_patience = 5,
        target_series_unscaled = None,
        target_series = None,
        past_covariates = None,
        future_covariates = None,
        target_scalers = None,
        covariates_scalers = None,
        additional_model_info = {},
        name_qualifier = None,
        log_tensorboard = False,
        is_new_model = True
    ):

        self.init_params = self.get_init_params(locals())

        # Set model attributes.
        self.model_name = model_name
        self.target_series_names = target_series_names
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.probabilistic_num_samples = probabilistic_num_samples
        self.max_epochs = max_epochs
        self.early_stopper_patience = early_stopper_patience
        self.log_tensorboard = log_tensorboard

        # Set the model save path and create the directory if required.
        self.model_save_path = model_save_path
        if is_new_model:
            self.create_model_dir(model_save_path)

        # Set the target timeseries components and scalers.
        self._set_target_series_components(target_series, target_series_unscaled)
        self.target_scalers = target_scalers

        # Set start and end times of dataset.
        self.train_val_data_start = target_series['train_val'][0].start_time()
        self.train_val_data_end = target_series['train_val'][0].end_time()
        self.train_data_end = target_series['train'][0].end_time()

        # Add name qualifier to additional model info before getting the saved name.
        additional_model_info['name_qualifier'] = name_qualifier
        self.additional_model_info = additional_model_info
        self.saved_model_name = self.get_saved_model_name(additional_model_info)

        # Retrieve the model args from the model attributes data.
        self.model_args = self.get_model_args()

        # If we are using lagged future covariates then we need to remove a chunk equal to
        # forecast horizon at the end of training to make sure we don't go past our data end.
        # (We always set the lags_future_covariates to be the same as the model forecast horizon.)
        if (
            'lags_future_covariates' in self.model_args and
            self.model_args['lags_future_covariates']
        ):
            self._shorten_target_series_by_forecast_horizon()

        self.is_torch_forecasting_model = self._check_if_torch_forecasting_model(
            self.model_args['Class']
        )
        self.is_global_forecasting_model = self._check_if_global_forecasting_model()

        self.darts_model = None
        self.training_args = None

        self.covariates_scalers = covariates_scalers

        # Initialise the darts model.
        if is_new_model:
            self._init_darts_model(past_covariates, future_covariates)

            # Set covariates series. We do this after the initialisation of the darts model
            # so we can check which covariates are supported.
            self.past_covariates = None
            self.future_covariates = None
            self.set_covariates(past_covariates, future_covariates)

        self._set_training_args()


    def get_init_params(self, params):
        params.pop('self')
        return params


    def set_covariates(self, past_covariates, future_covariates):
        self.past_covariates = past_covariates if self.uses_past_covariates else None
        self.future_covariates = future_covariates if self.uses_future_covariates else None


    def create_model_dir(self, model_save_path):

        # Check if this model has already been saved.
        if os.path.exists(f"{self.model_save_path}"):
            raise ValueError(
                f"Cannot create model - it already exists at {self.model_save_path}."
            )

        os.makedirs(model_save_path)


    @staticmethod
    def load_model(dir_path):
        """Loads an instance of this class.

        Args:
            dir_path: String of path to the folder containing the saved model.

        Returns:
            The loaded model instance.
        """

        # Load the init parameters and arguments for the saved model.
        init_params_path = os.path.join(dir_path, 'model_init_params.pkl')
        if not os.path.exists(init_params_path):
            print(f"Model file not found at {init_params_path}")
            return None

        with open(init_params_path, 'rb') as file:
            init_params = pickle.load(file)

            # Set the flag so that we don't try to save again on creating a new model instance.
            init_params['is_new_model'] = False

        # Create a new model instance using the same init params as the original.
        loaded_model = Model(**init_params)

        # Load in the darts model.
        darts_model_class = globals()[loaded_model.model_args['Class']]
        loaded_model.darts_model = darts_model_class.load(os.path.join(dir_path, 'darts_model'))

        # Set covariates now that we have loaded the darts model.
        loaded_model.set_covariates(
            init_params['past_covariates'],
            init_params['future_covariates']
        )

        # Set the trainin args.
        loaded_model._set_training_args()

        # Set the model_save_path to the current location in case the folder has been moved.
        # Also reset the darts model work_dir accordingly if necessary.
        # TODO The tensorboard logs dir is still set to the old path. It is set when the
        # darts model is instantiated by creating a new logger for the pytorch lightning
        # model so we can't just reset it here.
        loaded_model.model_save_path = dir_path
        if loaded_model.is_torch_forecasting_model:
            loaded_model.darts_model.work_dir = loaded_model._get_working_dir()

        return loaded_model

    def _set_target_series_components(self, target_series, target_series_unscaled):

        # Training
        self.target_series = target_series['train']
        self.target_series_unscaled = target_series_unscaled['train']

        # Validation
        self.val_series = target_series['val']
        self.val_series_unscaled = target_series_unscaled['val']

        # Combined training and validation for historical forecasts.
        self.target_series_train_val = target_series['train_val']
        self.target_series_train_val_unscaled = target_series_unscaled['train_val']


    def _shorten_target_series_by_forecast_horizon(self):
        """Set the target series components, in each case removing a chunk at the
        end equivalent to the forecast horizon.
        """

        self.target_series = self.slice_ts_sequence(
            self.target_series, 0, -self.forecast_horizon)
        self.target_series_unscaled = self.slice_ts_sequence(
            self.target_series_unscaled, 0, -self.forecast_horizon)


    def get_model_choices(self):
        """Gets the model choices for this model (i.e., those
        parameters that define the model that aren't defined by the
        model type in the file 'Models parameters.csv').

        Returns:
            Dictionary of the model choices.
        """
        choices = {}
        for param in self.MODEL_CHOICE_PARAMS:
            choices[param] = getattr(self, param)
        return choices


    @property
    def uses_past_covariates(self):
        """Whether the Darts model uses past covariates.

        Returns:
            Boolean.
        """
        if not hasattr(self.darts_model, 'uses_past_covariates'):
            return False

        return self.darts_model.uses_past_covariates

    @property
    def uses_future_covariates(self):
        """Whether the Darts model uses future covariates.

        Returns:
            Boolean.
        """
        if not hasattr(self.darts_model, 'uses_future_covariates'):
            return False

        return self.darts_model.uses_future_covariates


    def get_past_covariates_features(self):
        """Gets a list of the past covariates features used by this model.

        Returns:
            List of strings that identify the past covariates features.
        """
        return self._get_ts_sequence_features(self.past_covariates)


    def get_future_covariates_features(self):
        """Gets a list of the future covariates features used by this model.

        Returns:
            List of strings that identify the future covariates features.
        """
        return self._get_ts_sequence_features(self.future_covariates)


    def _get_ts_sequence_features(self, ts_sequence):

        if ts_sequence is None:
            return None

        single_ts_sequence = ts_sequence[0] if isinstance(ts_sequence, list) else ts_sequence
        return single_ts_sequence.columns.to_list()

    def get_model_args(self):
        """Retrieves the specified arguments of this model.

        Returns:
            A dictionary of the model arguments.
        """

        model_args = {}

        for attribute in self.models_attributes.keys():

            # Get the attribute value
            attr_val = self.models_attributes[attribute][self.model_name]

            # Skip if NaN or False
            if pd.isna(attr_val) or not attr_val:
                continue

            # Convert floats to integers where needed.
            float_type_args = ['dropout', 'learning_rate']
            if isinstance(attr_val, np.float64) and attribute not in float_type_args:
                attr_val = int(attr_val)

            model_args[attribute] = attr_val

            # For torch models only, add in log_tensorboard and save_checkpoints if appropriate.
            if self._check_if_torch_forecasting_model(model_args['Class']):
                model_args['log_tensorboard'] = self.log_tensorboard

                # We don't save checkpoints if we aren't validating.
                if self.val_series:
                    model_args['save_checkpoints'] = True

        return model_args


    def get_saved_model_name(self, additional_model_info : dict):
        """Gets the name used to save the model.

        Returns:
            The model saved name as a string.
        """

        saved_model_name_elements = [self.model_name]

        model_choices = self.get_model_choices()

        model_info = {**additional_model_info, **model_choices}

        for param, choice in model_info.items():

            # Convert param to first letters only, so for example 'forecast_horizon'
            # becomes 'fp'
            short_param = ''.join([x[0] for x in param.split('_')])

            # If choice is a list convert to a string.
            if isinstance(choice, list):
                choice = '|'.join(choice)

            # Make sure we convert None / NaN consistently.
            if pd.isna(choice):
                choice = None

            saved_model_name_elements.append(
                f"{short_param}={choice}"
            )

        saved_model_name = '_'.join(saved_model_name_elements)

        return saved_model_name


    def _check_if_torch_forecasting_model(self, model_class):
        return issubclass(globals()[model_class], TorchForecastingModel)


    def _check_if_global_forecasting_model(self):
        model_class = globals()[self.model_args['Class']]
        return issubclass(model_class, GlobalForecastingModel)


    @property
    def is_probabilistic(self):
        """Whether the Darts model is a probabilistic model.

        Returns:
            Boolean.
        """
        model_class = globals()[self.model_args['Class']]
        return hasattr(model_class, 'likelihood')

    @property
    def has_output_chunk_length_param(self):
        """Whether the Darts class for this model has the parameter 'output_chunk_length'.

        Returns:
            Boolean.
        """
        model_class = globals()[self.model_args['Class']]
        return hasattr(model_class, 'output_chunk_length')


    def _get_working_dir(self):
        return f"{self.model_save_path}"


    def _init_darts_model(self, past_covariates, future_covariates):

        # Create model args skipping any that must be converted or used only in training.
        darts_model_args = self.model_args.copy()
        model_args_to_skip = [
            'Name',
            'Class',
            'output_chunk_length',
            'learning_rate',
            'early_stopper',
            'lags_past_covariates',
            'lags_future_covariates',
        ]
        _ = [darts_model_args.pop(key) for key in model_args_to_skip if key in darts_model_args]

        # Add in max epochs, work_dir and model_name arguments for any Torch forecasting models.
        if self.is_torch_forecasting_model:
            darts_model_args['n_epochs'] = self.max_epochs
            darts_model_args['model_name'] = self.saved_model_name
            darts_model_args['work_dir'] = self._get_working_dir()

        # Add in output chunk length if model can handle it.
        if self.has_output_chunk_length_param:
            darts_model_args['output_chunk_length'] = self.forecast_horizon

        # Convert learning rate argument to format for Darts model.
        if 'learning_rate' in self.model_args:
            darts_model_args['optimizer_kwargs'] = {'lr': self.model_args['learning_rate']}

        # Add in early stopper if one is set and we have validation data.
        if (
            self.val_series and
            self.is_torch_forecasting_model and self.model_args['early_stopper']
        ):

            monitor = "val_" + self.model_args['early_stopper']

            early_stopper_params = {
                'monitor' : monitor,
                'patience' : self.early_stopper_patience,
                'min_delta' : 0.0,
                'mode' : 'min',
            }

            early_stopper = EarlyStopping(**early_stopper_params)
            checkpointer = ModelCheckpoint(
                monitor = monitor,
                dirpath = os.path.join(self._get_working_dir(), self.saved_model_name, 'checkpoints'),
                filename= 'custom-stop-{epoch}-{val_loss:.4f}-{' + monitor + ':.4f}'
            )
            darts_model_args['pl_trainer_kwargs'] = {'callbacks' : [early_stopper, checkpointer]}
            if self.model_args['early_stopper'] != 'loss':
                darts_model_args['torch_metrics'] = globals()[self.model_args['early_stopper']]()

        # Set the probabilistic parameters as required.
        if self.is_probabilistic:
            if self.model_args['likelihood'] == 'Deterministic':
                darts_model_args['likelihood'] = None
                darts_model_args['loss_fn'] = nn.MSELoss()
            elif darts_model_args['likelihood'] == 'Probabilistic':
                darts_model_args['likelihood'] = QuantileRegression(quantiles=self.quantiles)
            else:
                raise ValueError(
                    'If specified, likelihood must be "Deterministic" or "Probabilistic"')

        # TODO. This is a really hacky way of avoiding checking whether
        # past covariates are set. The problem is that we can't set
        # self.past_covariates and self.future_covariates until after
        # the darts model has been initialised because otherwise we
        # can't check whether past or future covariates are supported by
        # the model.
        if (
            'lags_past_covariates' in self.model_args and
            self.model_args['lags_past_covariates'] and
            past_covariates
        ):
            darts_model_args['lags_past_covariates'] = (
                self.forecast_horizon
            )

        if (
            'lags_future_covariates' in self.model_args and
            self.model_args['lags_future_covariates'] and
            future_covariates
        ):
            darts_model_args['lags_future_covariates'] = (
                0,
                self.forecast_horizon
            )

        # Create the Darts model instance.
        model_class = globals()[self.model_args['Class']]
        self.darts_model = model_class(**darts_model_args)


    def _maybe_convert_ts_to_list(self, ts_sequence):
        """Converts ts_sequence to a list when ts_sequence is not a
        list but a timeseries.
        """

        if not isinstance(ts_sequence, list):
            ts_sequence = [ts_sequence.copy()]
        return ts_sequence


    def _set_training_args(self):

        # Check that the model handle can multiple series if necessary.
        if not self.is_global_forecasting_model:
            if isinstance(self.target_series, list):
                if len(self.target_series) > 1:
                    raise ValueError(
                        f"{self.model_args['Class']} model can only forecast one series."
                    )

        training_args = {'series' : self.target_series}

        # Set past_covariates and future_covariates arguments.
        if self.uses_past_covariates:
            training_args['past_covariates'] = self.past_covariates

        if self.uses_future_covariates:
            training_args['future_covariates'] = self.future_covariates

        # Add in validation series arguments if are validating and if we are using early stopper.
        if (
            self.val_series and
            self.is_torch_forecasting_model and self.model_args['early_stopper']
        ):

            training_args['val_series'] = self.val_series

            if self.uses_past_covariates:
                training_args['val_past_covariates'] = self.past_covariates

            if self.uses_future_covariates:
                training_args['val_future_covariates'] = self.future_covariates

        # Convert any timeseries sequence with only one timeseries to a single timeseries
        # as some models don't accept sequences.
        training_args = {
            k : self._maybe_convert_ts_sequence_to_ts(ts) for k, ts in training_args.items()
        }

        # Set verbose argument.
        if self.is_torch_forecasting_model:
            training_args['verbose'] = True

        self.training_args = training_args


    def _get_target_series_unscaled(self, predict_series = ''):
        pred_ts_seq_index = self.target_series_names.index(predict_series) if predict_series else 0
        return self.target_series_train_val_unscaled[pred_ts_seq_index]


    def train_and_save(self):
        """Train the Darts model and save the trained model."""

        self.darts_model.fit(
            **self.training_args
        )

        print(f"Saving the model to: {self.model_save_path}")

        # Save the init params of this model instance.
        with open(os.path.join(self.model_save_path, 'model_init_params.pkl'), 'wb') as file:
            pickle.dump(self.init_params, file)

        # Save darts model.
        self.darts_model.save(os.path.join(self.model_save_path, 'darts_model'))


    def _maybe_convert_ts_sequence_to_ts(self, ts):
        if isinstance(ts, list):
            if len(ts) == 1:
                ts = ts[0]
        return ts


    def get_best_darts_model(self, val_loss_monitor = False):
        """Gets the best saved Darts model.

        The default metric used during training is val_loss. However this
        can be changed to a custom metric in the file 'Models parameters.csv'.
        The val_loss_monitor parameter is used to determine whether to
        retrieve the best model as determined by the default metric or as
        determined by the custom metric.

        Args:
            val_loss_monitor: Boolean of whether to use the default val_loss metric.

        Returns:
            A Darts model instance.
        """
        if val_loss_monitor:
            return self._get_best_val_loss_darts_model()

        model = self.darts_model
        # If this class of model doesn't store checkpoints then just use current model as is.
        if not hasattr(model, 'load_from_checkpoint'):
            return model

        # If we haven't been using an early stopper then use current model as is.
        if 'early_stopper' not in self.model_args:
            return model

        # Now let's check if we have a "custom" saved checkpoint.
        base_path = os.path.join(self._get_working_dir(), self.saved_model_name, 'checkpoints')
        if os.path.isdir(base_path):
            filenames = os.listdir(base_path)
            checkpoint_filename = None
            for filename in filenames:
                if filename[:6] == 'custom':
                    checkpoint_filename = filename

            if checkpoint_filename:
                return model.load_from_checkpoint(
                    self.saved_model_name,
                    file_name = checkpoint_filename,
                    work_dir = self._get_working_dir()
                )

        # Since we don't have "custom" saved checkpoint we'll load the standard
        # best checkpoint (determined by val_loss)
        return self._get_best_val_loss_darts_model()


    def _get_best_val_loss_darts_model(self):

        model = self.darts_model

        # If this class of model doesn't store checkpoints then just use current model as is.
        if not hasattr(model, 'load_from_checkpoint'):
            return model

        # Equally if we don't have any checkpoints then just use current model.
        if not os.path.exists(os.path.join(self._get_working_dir(), self.saved_model_name, 'checkpoints')):
            return model

        return model.load_from_checkpoint(
            self.saved_model_name,
            work_dir = self._get_working_dir(),
            best = True
        )


    def get_tensorboard_log_dir(self):
        """Gets the path to the directory where tensorboard logs are stored.

        Returns:
            String of the path to the directory.
        """
        return os.path.join(self._get_working_dir(), self.saved_model_name, 'logs')


    def _get_validation_series_length(self):
        if not self.val_series:
            return 0

        val_series = self._maybe_convert_ts_to_list(self.val_series)
        return len(val_series[0])


    def get_predicted_series(
        self,
        darts_model = None,
        series = None,
        forecast_horizon = 1,
        past_covariates = None,
        future_covariates = None
    ):
        """Gets a forecast of the target series.

        Args:
            darts_model: A Darts model instance. If not provided, the Darts model
                saved at the end of training will be used. Usually a "best" model is
                provided instead.
            series: A Darts timeseries or a list of Darts timeseries. The forecast(s) begins
                at the end of the provided series. If none is provided, the series used for
                training is used.
            forecast_horizon: Integer number of time steps to forecast.
            past_covariates: A Darts timeseries or a list of Darts timeseries of the past
                covariates to be used in prediction.
            future_covariates: A Darts timeseries or a list of Darts timeseries of the future
                covariates to be used in prediction.

        Returns:
            Prediction(s) in the form of a Darts timeseries or a list of Darts timeseries.
        """

        predict_args = {}

        # Set covariates if needed, and only use the model's stored covariates if
        # they have not been specified as arguments.
        if self.uses_past_covariates:
            predict_args['past_covariates'] = \
                past_covariates if past_covariates is not None else self.past_covariates

        if self.uses_future_covariates:
            predict_args['future_covariates'] = \
                future_covariates if future_covariates is not None else self.future_covariates

        # Convert covariates to a single timeseries if we have a sequence with only one timeseries.
        predict_args = {
            k : self._maybe_convert_ts_sequence_to_ts(ts) for k, ts in predict_args.items()
        }

        predict_args['n'] = forecast_horizon

        # Set number of samples for probabilistic prediction if required.
        num_samples = 1
        if (
            self.is_probabilistic and
            self.model_args['likelihood'] == 'Probabilistic'
        ):
            num_samples = self.probabilistic_num_samples

        predict_args['num_samples'] = num_samples

        # Add in which target series we are predicting if we need to.
        if series:
            predict_args['series'] = series

        model = darts_model if darts_model else self.darts_model

        pred_series = model.predict(**predict_args)

        if isinstance(pred_series, list):
            for i in range(len(pred_series)):
                pred_series[i] = self.target_scalers[i].inverse_transform(pred_series[i])
        else:
            pred_series = self.target_scalers[0].inverse_transform(pred_series)

        return pred_series


    def get_historical_forecast(
        self,
        target_series = None,
        predict_series : str = '',
        darts_model = None,
        start : str = '',
        end : str = '',
        forecast_horizon = None,
        stride = None,
    ):
        """Gets historical forecast(s).

        Args:
            target_series: A Darts timeseries or a list of Darts timeseriess, representing
                the history of the target series whose future is to be predicted. If a list
                is provided, the target series is set using the predict_series parameter.
                If target_series is not specified the method returns the forecast of the
                training series.
            predict_series: A string representing which target series to predict. Only
                required if a list of timeseries is provided as target_series.
            darts_model: A Darts model instance. If not provided, the Darts model
                saved at the end of training will be used. Usually a "best" model is
                provided instead.
            start: A date as a string in the form yyyy-mm-dd, representing from when
                the historical forecasts should start. If not specified, the earliest
                possible point of the target series will be used (which depends on
                the input chunk length of the model).
            end: A date as a string in the form yyyy-mm-dd, representing when the historical
                forecasts should end. If not specified, the end of the target series
                will be used.
            forecast_horizon: Integer of number of timesteps to predict per forecast.
            stride: Integer of how many timesteps to advance from the start of one forecast
                to the next.

        Returns:
            A list of Darts timeseries.
        """

        if forecast_horizon is None:
            forecast_horizon = self.forecast_horizon

        if stride is None:
            stride = self.forecast_horizon

        # Each sequence of timeseries has the same order so we set the index to
        # use for the series we are backtesting.
        pred_ts_seq_index = self.target_series_names.index(predict_series)

        if isinstance(target_series, list):
            target_series = target_series[pred_ts_seq_index]

        if not end:
            end = target_series.end_time()
        else:
            end = pd.to_datetime(end)

        if not start:
            start_index = max(1, self.model_args['input_chunk_length'])
            start = target_series.time_index[start_index]
        else:
            start = pd.to_datetime(start)

        retrain = not self.darts_model._supports_non_retrainable_historical_forecasts()

        historical_forecast_args = {
            'series' : target_series,
            'forecast_horizon' : forecast_horizon,
            'stride' : stride,
            'last_points_only' : False,
            'retrain' : retrain,
            'start' : start,
        }

        if (
            self.is_probabilistic and
            self.model_args['likelihood'] == 'Probabilistic'
        ):
            historical_forecast_args['num_samples'] = self.probabilistic_num_samples

        # Set the historical forecast covariates.
        cov_hf_args = {}
        if self.uses_past_covariates and self.past_covariates:
            cov_hf_args['past_covariates'] = self.past_covariates[pred_ts_seq_index]
        if self.uses_future_covariates and self.future_covariates:
            cov_hf_args['future_covariates'] = self.future_covariates[pred_ts_seq_index]

        # Convert covariates to a single timeseries if we have a sequence with only one timeseries.
        cov_hf_args = {
            k : self._maybe_convert_ts_sequence_to_ts(ts) for k, ts in cov_hf_args.items()
        }

        # Combine all historical forecast args.
        historical_forecast_args = {**historical_forecast_args, **cov_hf_args}

        model = darts_model if darts_model else self.darts_model

        forecasts = model.historical_forecasts(**historical_forecast_args)

        scaler = self.target_scalers[pred_ts_seq_index]

        return [scaler.inverse_transform(ts) for ts in forecasts]


    def get_historical_forecast_window(
        self,
        historical_forecasts,
        start,
        end
    ):
        """Gets a subset of a list of the (chronologically ordered) historical forecasts.

        The first historical forecast is that in which the specified start date is found, and
        the last historical forecast is that in which the specified end date is found.

        Args:
            historical_forecasts: A list of Darts timeseries, being the chronologically
                ordered forecasts.
            start: Start date as a string in the form yyyy-mm-dd
            end: End date as a string in the form yyyy-mm-dd

        Returns:
            Forecasts as a list of Darts timeseries.
        """

        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        first_index = historical_forecasts[0]
        last_index = historical_forecasts[-1]
        for i in range(len(historical_forecasts)):
            if historical_forecasts[i].is_within_range(start):
                first_index = i
            if historical_forecasts[i].is_within_range(end):
                last_index = i

        return historical_forecasts[first_index:last_index + 1]

    def plot_forecasts(
        self,
        forecasts,
        predict_series = '',
        figsize = (16, 8),
        title = ''
    ):
        """Plots forecasts.

        Args:
            forecasts: Darts timeseries or a list of Darts timeseries, each representing
                a different item being forecasted.
            predict_series: A string indicating which item should be forecasted if several
                have been provided.
            figsize: A tuple representing figure size to be passed to Matplotlib.
            title: A string to be used as the plot title.
        """

        # If we have been given a single forecast timeseries, convert to a list.
        if not isinstance(forecasts, list):
            forecasts = [forecasts]

        # Get the actual series to compare with.
        actual_series = self._get_target_series_unscaled(predict_series)
        actual_series_slice = actual_series.slice(
            forecasts[0].start_time(),
            forecasts[-1].end_time()
        )

        self.plot_series(
            forecasts,
            actual_series = actual_series_slice,
            title = title,
            figsize = figsize
        )


    def plot_series(
        self,
        forecasts,
        actual_series = None,
        figsize = (16, 8),
        title = ''
    ):
        """Plot forecast(s) against actual if provided.

        Args:
            forecasts: A forecast or a list of forecasts as Darts timeseries.
            actual_series: A Darts timeseries representing known data.
            figsize: A tuple representing figure size to be passed to Matplotlib.
            title: A string to be used as the plot title.

        """
        plt.figure(figsize=figsize)

        if actual_series:
            actual_series.plot(label = 'Actual')

        # If we have been given a single forecast timeseries, convert to a list.
        if not isinstance(forecasts, list):
            forecasts = [forecasts]

        for forecast in forecasts:
            forecast.plot(label = f"Forecast {forecast.start_time()} to {forecast.end_time()}")
        plt.legend()
        if title:
            plt.title(title)
        plt.show()


    def get_metrics(
        self,
        forecasts,
        metrics = ['MAPE', 'MAE'],
        predict_series = '',
        output_type = 'both'
    ):
        """Gets performance metrics for forecast by comparing with the model target
        series data.

        If several chronological forecasts are provided, then the metric for each forecast is
        provided, as well as an aggregate mean result for all forecasts.

        Args:
            forecasts: A forecast or a list of forecasts as Darts timeseries.
            metrics: A list of strings, containing one, both or neither of 'MAPE'
                and 'MAE'.
            predict_series: A string indicating for which item metrics should be
                calculated if several have been provided.
            output_type: A string, one of 'both', 'individual' or 'aggregate' indicating which
                type(s) of metrics to return.

        Returns:
            A dictionary of the form {
                'mape' : {
                    'individual' : A list of values,
                    'aggregate' : The single aggregate value.
                },
                'mae' : {
                    'individual' : A list of values,
                    'aggregate' : The single aggregate value.
                }
            }
        """

        # Covert forecasts to a list if we are given a single timeseries.
        if not isinstance(forecasts, list):
            forecasts = [forecasts]

        # Get the actual series to compare with.
        target_series = self._get_target_series_unscaled(predict_series)
        target_series = [target_series] * len(forecasts)

        output = {}
        for metric in metrics:

            metric_func = globals()[metric.lower()]

            individual = []
            for i in range(len(forecasts)):
                individual.append(metric_func(target_series[i], forecasts[i]))

            aggregate = metric_func(target_series, forecasts, inter_reduction = np.mean)

            if output_type == 'individual':
                output = individual

            elif output_type == 'aggregate':
                output = aggregate

            else:
                output[metric] = {
                    'individual' : individual,
                    'aggregate' : aggregate
                }

        return output


    def get_validation_period_forecast(self, darts_model = None):
        """Gets a forecast for the whole validation period.

        Args:
            darts_model: A Darts model instance. If not provided, the Darts model
                saved at the end of training will be used. Usually a "best" model is
                provided instead.

        Returns:
            Prediction(s) in the form of a Darts timeseries or a list of Darts timeseries.

        Raises:
            ValueError if the model has no validation data (most likely because it is
            being run in production mode).
        """

        # We can't run determine the period if we don't have a validation period.
        if self.val_series is None:
            raise ValueError(
                'Validation period forecast not possible as we have no validation series. ' \
                'Are you in production mode?'
            )

        # Set forecast horizon to cover whole validation period.

        get_predicted_series_kwargs = {
            'forecast_horizon' : self._get_validation_series_length(),
        }

        # Specify the sequence of target series we are predicting if we have more
        # than one. We don't always set it because some models cannot forecast more
        # than one series and don't take series as an argument when predicting.
        if len(self.target_series_names) > 1:
            get_predicted_series_kwargs['series'] = self.target_series

        model_pred_series = self.get_predicted_series(
            darts_model if darts_model else self.darts_model,
            **get_predicted_series_kwargs
        )

        return model_pred_series


    def slice_ts_sequence(self, ts_sequence, start, end):
        """Gets a slice of each Darts timeseries in a list of Darts timeseries.

        Args:
            ts_sequence: A list of Darts timeseries.
            start: An integer representing the time step to start the slice.
            end: An integer representing the time step to end the slice.
        """
        if ts_sequence is None:
            return None

        return [ts[start:end] for ts in ts_sequence]

