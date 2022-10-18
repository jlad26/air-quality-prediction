"""Module for constants for the air_quality package.
"""

import os
from dotenv import dotenv_values

# .env file location is dependent on evironment. If in docker, it is stored in
# /app/app_data. Otherwise it is stored in parent of current working directory.

AZURE_ENV_DIR = '/app/app_data'
env_dir = AZURE_ENV_DIR if os.path.exists(AZURE_ENV_DIR) else os.path.dirname(os.getcwd())

env_path = os.path.join(env_dir, '.env')
if not os.path.exists(env_path):
    raise FileNotFoundError(f"Could not find .env file at {env_path}")

# Set environment variables, converting text boolean values to true booleans.
ENV_VARS = dotenv_values(env_path)
for var, value in ENV_VARS.items():
    if value.lower() in ('true', 'false'):
        ENV_VARS[var] = value.lower() == 'true'

# Set the working directory. Default is same folder as .env file.
WORK_DIR = ENV_VARS['WORK_DIR'] if 'WORK_DIR' in ENV_VARS else os.path.join(os.getcwd(), 'app_data')

LOG_DIR_PATH = os.path.join(WORK_DIR, 'logs')
