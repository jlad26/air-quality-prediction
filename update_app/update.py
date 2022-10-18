import air_quality.data_management as dm

updater = dm.DataUpdater()

updater.update_all()

updater.clear_key_times_cache()

print(updater.hit_home_page())

print(updater.refresh_metrics())