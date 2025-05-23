bind = "0.0.0.0:8080"
workers = 1  # keep it to 1 for low memory platforms like Railway
preload_app = False  # don't preload the app in master process
timeout = 120  # increase if model inference takes time