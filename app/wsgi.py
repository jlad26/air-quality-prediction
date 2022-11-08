"""Module loaded on launch of container to start Gunicorn server.
"""

from air_quality_prediction import app

if __name__ == "__main__":
	app.run()