"""Script called by cron to update data.
"""

import air_quality_updater.data_management as dm

updater = dm.DataUpdater()

# Download new pollutant, historical weather and weather forecast data via APIs, then clean
# and merge with existing data.
updater.update_all()

# Clear the key times cache and then hit the home page so that: a) the key times cache is
# rebuilt taking into account the new data, and b) the new default forecast plots are generated.
updater.clear_key_times_cache()
print(updater.hit_home_page())

# Refresh the metrics based on the new data and key times.
print(updater.refresh_metrics())