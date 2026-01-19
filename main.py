from flask import Flask
from app import app1
from app2 import app2
import os
from datetime import datetime

app = Flask(__name__, template_folder="templates", static_folder="static")

# Base folders
UPLOAD_FOLDER = "uploads"
BASE_MERGED_FOLDER = "merged_pdfs"

# Generate timestamped folder (YYYYMMDD_HHMMSS)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MERGED_FOLDER = os.path.join(BASE_MERGED_FOLDER, f"run_{timestamp}")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MERGED_FOLDER, exist_ok=True)

# Flask config
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MERGED_FOLDER"] = MERGED_FOLDER
app.config["SENDER_EMAIL"] = "rajesh@legodesk.com"
app.config["SENDER_PASSWORD"] = "kjfh uzeu dynh kkfe"
app.config["OCR_THREADS"] = 4

# Register blueprints
app.register_blueprint(app1)
app.register_blueprint(app2)

if __name__ == "__main__":
    app.run(debug=True, port=5000)


