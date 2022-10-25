"""Module for constants for the air_quality package.
"""

import os
from dotenv import dotenv_values

# .env file location is dependent on evironment. If in docker, it is stored in
# /app/app_data. Otherwise we look for it in current working directory then its parent.
# This is because for prediction web app and updater share a .env in their common parent.

# First look for bind mount.
HOST_ENV_DIR = '/app/app_data'
if os.path.exists(HOST_ENV_DIR):
    env_dir = HOST_ENV_DIR

else:

    # Then check in current working directory.
    if os.path.exists(os.path.join(os.getcwd(), '.env')):
        env_dir = os.getcwd()

    # Followed by parent of current working directory.
    else:
        env_dir = os.path.dirname(os.getcwd())

env_path = os.path.join(env_dir, '.env')
if not os.path.exists(env_path):
    raise FileNotFoundError(f"Could not find .env file at {env_path}")

# Set environment variables, converting text boolean values to true booleans.
ENV_VARS = dotenv_values(env_path)
for var, value in ENV_VARS.items():
    if value.lower() in ('true', 'false'):
        ENV_VARS[var] = value.lower() == 'true'

# Set the working directory. Default is same folder as .env file.
WORK_DIR = ENV_VARS['WORK_DIR'] if 'WORK_DIR' in ENV_VARS else env_dir

LOG_DIR_PATH = os.path.join(WORK_DIR, 'logs')
