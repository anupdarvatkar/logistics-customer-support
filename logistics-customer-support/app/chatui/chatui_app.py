import os
import logging
from dotenv import load_dotenv
from flask import Flask, render_template

# Import the blueprint from app_routes.py which contains all our API endpoints
from app_routes import chat_bp

# --- Basic App Setup ---
load_dotenv()
app = Flask(__name__)

# A secret key is required for Flask sessions to work.
# The session is used in app_routes.py to store a unique user_id.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_secure_default_secret_key")

# Register the blueprint. All routes defined in app_routes.py (e.g., /api/chat/...)
# will now be active in the application.
app.register_blueprint(chat_bp)

# --- Configuration ---
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "8080"))

# --- Main Page and Error Routes ---

@app.route('/')
def home():
    """
    Serves the main chat interface page.
    This assumes you have a 'chat_ui.html' in your 'templates' folder.
    """
    return render_template('chat_ui.html')

@app.errorhandler(404)
def page_not_found(e):
    """Handles 404 Not Found errors."""
    # This assumes a '404.html' template exists that likely extends chat_ui.html
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handles 500 Internal Server errors."""
    logging.error(f"Internal Server Error: {e}")
    # This assumes a '500.html' template exists that likely extends chat_ui.html
    return render_template('500.html'), 500

# --- Main Execution ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"\n--- Starting Logistics Chat UI Server on http://{APP_HOST}:{APP_PORT} ---")
    # debug=True is useful for development; it enables auto-reloading.
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    app.run(debug=True, host=APP_HOST, port=APP_PORT)