from flask import Flask
from app import app1
from app2 import app2
from flask import current_app

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "uploads"
MERGED_FOLDER = "merged_pdfs"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["MERGED_FOLDER"] = MERGED_FOLDER
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MERGED_FOLDER"] = "merged_pdfs"
app.config["SENDER_EMAIL"] = "main id"
app.config["SENDER_PASSWORD"] = "mail password"
app.config["OCR_THREADS"] = 4

# register both backends
app.register_blueprint(app1)
app.register_blueprint(app2)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
