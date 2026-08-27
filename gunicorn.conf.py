import os

# Dynamically bind to Render's port, defaulting to 10000 locally
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
timeout = 120
workers = 2