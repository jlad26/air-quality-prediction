# Air Quality Development and Model Selection

The outcome of this project can be seen at [airquality.sneezingtrees.com](http://airquality.sneezingtrees.com) -
an online tool for predicting air quality in Montpellier. I recommend you start there as most of the initial background and explanatory information is available there.

This is one of two repositories that go hand in hand:
- [Air Quality Development and Selection](https://github.com/jlad26/air-quality-model-selection) - contains all work done to develop and select a machine learning model for predicting the concentrations of the five pollutants used to measure European Air Quality.
- **Air Quality Prediction Application** (where you are now) - the code for the web application at [airquality.sneezingtrees.com](http://airquality.sneezingtrees.com)

# Data
All the data required for this project that is not available in this repository (and its sibling) is available on [Google Drive](https://drive.google.com/drive/folders/1oyqjshm5qBBPRnwVxDH2NI_Q4hTdpFci?usp=sharing). The "live" data that the website uses is as at 21 October 2022.
You will need to request an API key from [Geod'Air](https://www.geodair.fr/donnees/api) to update beyond that (it's free).

# Local deployment

- Make sure you have a `.env` file in the root directory i.e. the parent directory of the `app` and `update_app` folders. Use the `.env-sample` file to help you.
- Download the `app_data` folder from Google Drive and put it somewhere in your filesystem (where you as a user have access). Set the `WORK_DIR` and `BIND_MOUNT_PATH` values in your `.env` file to that path.

## Flask / Gunicorn

### Flask
Run `flask --app air_quality_prediction --debug run` from the `app` folder. (You don't have to use the `--debug` option but it can be very helpful.) The site will then be available at http://localhost:5000/.

### Gunicorn
Run `gunicorn -w 4 --bind 0.0.0.0:8000 wsgi:app`. (The `-w` option is the number of workers. Adjust to suit your computer.) The site be available at http://localhost:8000/.

## Docker
Run with the command `docker compose up` in a terminal in the parent directory of the `app` and `update_app` folders where the `docker-compose.yml` file is.

# Cloud hosted
1. Create an `app_data` folder on the host and upload all the data in the `app_data` folder on Google drive. Set the folder and all of its contents to permssions 777.
1. Upload a docker version of the `.env` file (i.e. `WORK_DIR` key-value pair is excluded) to the `app_data` folder on the host.
1. Upload the file `docker-compose cloud.yml` to the `app_data` folder on the host and rename the file to `docker-compose.yml`.
1. Run build command `docker build -t {your_docker_hub_username}/aq-prediction:latest --build-arg CACHEBUST=$(date +%s) .` in `app` folder on your local machine.
1. Run build command `docker build -t {your_docker_hub_username}/aq-update:latest --build-arg CACHEBUST=$(date +%s) .` in `update_app` folder on your local machine.
1. Push to docker hub from your local machine with `docker push {your_docker_hub_username}/aq-prediction:latest` and `docker push {your_docker_hub_username}/aq-update:latest`.
1. Log in to your host via SSH and navigate to the `app_data` folder. Run `docker compose up --detach` command.