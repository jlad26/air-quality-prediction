import string
from datetime import datetime, timedelta
from flask import Flask, render_template, request
from air_quality import prediction
import air_quality.constants as C
from air_quality.metrics import MetricsManager


app = Flask(__name__)


@app.route('/')
def index():

    key_times = prediction.get_key_times()

    end_day = key_times['last_forecast_start'] + timedelta(days = 3)

    # Get a default image of the current forecast for NO2.
    predictor = prediction.PollutantPredictor('NO2')
    default_img_dir_dict = predictor.get_prediction_result_dir(
        key_times['last_forecast_start'],
        end_day
    )
    default_img_dir = default_img_dir_dict['plots_dir']

    metric_manager = MetricsManager()
    metrics_plots_urls = metric_manager.get_plot_urls()

    ga_id = C.ENV_VARS['GA_ID'] if 'GA_ID' in C.ENV_VARS else 'none'

    return render_template(
        'index.html',
        default_img_dir = default_img_dir,
        ga_id = ga_id,
        **key_times,
        **metrics_plots_urls
    )


@app.route('/fetch-pollutant')
def fetch_pollutant():

    # Check that all query paramters are provided.
    required_query_params = ['pollutant', 'start', 'days']
    provided_query_params = list(request.args.keys())
    intersection = [value for value in required_query_params if value in provided_query_params]
    if set(required_query_params) != set(intersection):
        return error('Please specify query parameters of pollutant, start and end.')

    # Validate pollutants value.
    pollutant = _sanitize_string(request.args['pollutant']).upper()
    if pollutant not in _pollutants():
        return error(f"Please specify pollutant as one of {', '.join(_pollutants())}.")

    # Validate dates.
    datetimes = {}
    for date_type in ['start']:
        value = _sanitize_date(request.args[date_type])
        try:
            datetimes[date_type] = datetime.strptime(value, '%Y-%m-%d')
        except ValueError:
            return error('Please provide start and end dates in the format yyyy-mm-dd')

    # Validate days choice.
    days = int(request.args['days'])
    if not days or days not in (1, 2, 3, 4, 5):
        return error('Please provide day value of 1, 2, 3, 4 or 5')

    end_datetime = datetimes['start'] + timedelta(days = days - 1)

    predictor = prediction.PollutantPredictor(pollutant)

    response = predictor.get_prediction_result_dir(datetimes['start'], end_datetime)

    return response


@app.route('/refresh-metrics')
def refresh_metrics():
    if 'key' not in request.args:
        return 'Please provide a key.'
    key = _sanitize_string(request.args['key'])
    if key != C.ENV_VARS['CACHE_CLEAR_KEY']:
        return 'Invalid key'

    metrics_manager = MetricsManager()
    metrics_manager.update_metrics()
    return 'Metrics updated'


@app.route('/clear-plots')
def clear_plots():
    if 'key' not in request.args:
        return 'Please provide a key.'
    key = _sanitize_string(request.args['key'])
    if key != C.ENV_VARS['CACHE_CLEAR_KEY']:
        return 'Invalid key'
    predictor = prediction.PollutantPredictor('NO2')
    predictor.clear_plots_cache()
    return 'Cache cleared.'

def _sanitize_date(input):
    whitelist = string.digits + '-'
    return "".join(filter(lambda x: x in whitelist, input))


def _sanitize_string(input):
    whitelist = string.ascii_letters + string.digits + ' -_.'
    return "".join(filter(lambda x: x in whitelist, input))


def _pollutants():
    return ['NO2', 'O3', 'PM10', 'PM2.5', 'SO2']


def error(message):
    return {'error' : f"Prediction request error: {message}"}
