from flask import Flask, render_template, request, jsonify, send_file, make_response, Response
import pandas as pd
import os
import re
import time
import logging
from datetime import datetime
from googleapiclient.discovery import build
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
import smtplib
from email.message import EmailMessage
import json
import urllib.parse
import socket
import ssl
from googleapiclient.errors import HttpError
import http.client
from threading import Lock
from flask import Blueprint
import hashlib

# -------------------------------
# CONFIGURATION
# -------------------------------
app1 = Blueprint("app1", __name__, template_folder="templates", static_folder="static")
SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets.readonly"
]

# Email config (consider moving to env vars)
SENDER_EMAIL = "rajeshkumarpanda235@gmail.com"
SENDER_PASSWORD = "pmje ybac glnn wygd"

logging.basicConfig(filename="drive_search_log.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Do NOT build Drive/Sheets globally at import time on Windows/old httplib2 — build lazily instead
_drive_service = None
_sheets_service = None

CASE_TYPES_FILE = "case_types.json"
CASE_TYPES = {}
_case_types_lock = Lock()

STATE_DIR = "states"
os.makedirs(STATE_DIR, exist_ok=True)

def load_case_types():
    global CASE_TYPES
    try:
        if not os.path.exists(CASE_TYPES_FILE):
            # Create empty default if missing
            with open(CASE_TYPES_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=4)
        with open(CASE_TYPES_FILE, "r", encoding="utf-8") as f:
            CASE_TYPES = json.load(f)
    except Exception as e:
        logging.exception("Failed to load case_types.json, using empty dict")
        CASE_TYPES = {}

def save_case_types():
    global CASE_TYPES
    try:
        with _case_types_lock:
            with open(CASE_TYPES_FILE, "w", encoding="utf-8") as f:
                json.dump(CASE_TYPES, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.exception("Failed to save case_types.json")

# Load at startup
load_case_types()

def build_credentials():
    """Load service account credentials and ensure token is fresh (using requests transport)."""
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    try:
        creds.refresh(GoogleAuthRequest())  # ensures access token valid and avoids some ssl issues
    except Exception as e:
        logging.exception("Failed to refresh credentials")
        # continue — creds may still be usable (will try building services)
    return creds

def get_drive_service():
    global _drive_service
    try:
        creds = build_credentials()
        # use cache_discovery=False to avoid discovery cache issues
        _drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        logging.exception("Failed to build drive service")
        # fallback: try build without refresh
        try:
            creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            _drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
        except Exception as e2:
            logging.exception("Second attempt to build drive service failed")
            _drive_service = None
    return _drive_service

def get_sheets_service():
    global _sheets_service
    try:
        creds = build_credentials()
        _sheets_service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    except Exception as e:
        logging.exception("Failed to build sheets service")
        try:
            creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            _sheets_service = build("sheets", "v4", credentials=creds, cache_discovery=False)
        except Exception as e2:
            logging.exception("Second attempt to build sheets service failed")
            _sheets_service = None
    return _sheets_service

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# ORIGINAL CASE_TYPES moved to JSON and loaded above
# -------------------------------

# -------------------------------
# HELPER FUNCTIONS (unchanged / kept, plus state helpers)
# -------------------------------
def extract_folder_id(link):
    if not link:
        return None
    m = re.search(r'folders/([a-zA-Z0-9_-]+)', link)
    if m:
        return m.group(1)
    m = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', link)
    if m:
        return m.group(1)
    return link.strip()

def extract_sheet_id(link):
    if not link:
        return None
    m = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', link)
    if m:
        return m.group(1)
    m = re.search(r'[?&]id=([a-zA-Z0-9-_]+)', link)
    if m:
        return m.group(1)
    return link.strip()

def safe_drive_list(query, retries=4):
    """
    List Google Drive files safely with retries and exponential backoff.
    Uses get_drive_service() to ensure a requests-based transport (avoids httplib2 ssl issues on Windows).
    Returns dict with 'files' key (to keep compatibility).
    """
    backoff = 1
    all_files = []
    page_token = None
    attempt = 0

    while True:
        try:
            drive = get_drive_service()
            if drive is None:
                raise RuntimeError("Drive service not available")

            results = drive.files().list(
                q=query,
                fields="nextPageToken, files(id,name,mimeType)",
                pageToken=page_token,
            ).execute()

            all_files.extend(results.get("files", []))
            page_token = results.get("nextPageToken")

            if not page_token:
                break

        except (TimeoutError, ssl.SSLError, socket.timeout, HttpError, OSError, http.client.IncompleteRead, ValueError) as e:
            attempt += 1
            logging.exception(f"[safe_drive_list] Error: {e} (attempt {attempt}/{retries})")
            print(f"[safe_drive_list] Error: {e} (attempt {attempt}/{retries})")

            if attempt >= retries:
                print("[safe_drive_list] Max retries reached — returning partial results.")
                break

            # backoff and retry
            time.sleep(backoff)
            backoff *= 2
            continue

        except Exception as e:
            attempt += 1
            logging.exception(f"[safe_drive_list] Unexpected error: {e} (attempt {attempt}/{retries})")
            print(f"[safe_drive_list] Unexpected error: {e} (attempt {attempt}/{retries})")
            if attempt >= retries:
                break
            time.sleep(backoff)
            backoff *= 2
            continue

    return {"files": all_files}

def print_folder_structure(folder_id):
    tree = []
    query = f"'{folder_id}' in parents and trashed=false"
    results = safe_drive_list(query)
    for file in results.get("files", []):
        node = {"name": file["name"], "id": file["id"], "type": "folder" if file["mimeType"] == "application/vnd.google-apps.folder" else "file"}
        if node["type"] == "folder":
            node["children"] = print_folder_structure(file["id"])
        tree.append(node)
    return tree

def normalize_filename(filename):
    name_without_ext = filename.split('.')[0].lower()
    tokens = re.split(r'[-_\s.]+', name_without_ext)
    tokens = [token.strip() for token in tokens if token.strip()]
    return ' '.join(tokens)

def matches_pattern(normalized_filename, pattern_obj):
    if isinstance(pattern_obj, list):
        return all(re.search(rf'\b{re.escape(token)}\b', normalized_filename, re.IGNORECASE) for token in pattern_obj)
    return bool(re.search(pattern_obj, normalized_filename, re.IGNORECASE))

def search_files_with_regex(folder_id, pattern_obj):
    matched_links = []
    def recurse(fid):
        results = safe_drive_list(f"'{fid}' in parents and trashed=false")
        for item in results.get("files", []):
            if item["mimeType"] == "application/vnd.google-apps.folder":
                recurse(item["id"])
            else:
                normalized = normalize_filename(item["name"])
                if matches_pattern(normalized, pattern_obj):
                    matched_links.append(f"https://drive.google.com/file/d/{item['id']}/view")
    recurse(folder_id)
    return matched_links if matched_links else ["Not Found"]

# -------------------------------
# State helpers for checkpointing
# -------------------------------
_state_lock = Lock()

def make_state_id(folder_link, excel_path, borrower_col, doc_patterns):
    payload = {
        "folder_link": folder_link or "",
        "excel_path": excel_path or "",
        "borrower_col": borrower_col or "",
        "doc_patterns": doc_patterns or {}
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest

def state_file_for_id(state_id):
    return os.path.join(STATE_DIR, f"state_{state_id}.json")

def partial_output_for_id(state_id):
    return os.path.join(UPLOAD_FOLDER, f"partial_{state_id}.xlsx")

def save_state(state_id, state_obj):
    path = state_file_for_id(state_id)
    try:
        with _state_lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state_obj, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.exception("Failed to save state file")

def load_state(state_id):
    path = state_file_for_id(state_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.exception("Failed to load state file")
            return None
    return None

def cleanup_state(state_id):
    # remove state file and partial file if they exist
    sf = state_file_for_id(state_id)
    pf = partial_output_for_id(state_id)
    try:
        if os.path.exists(sf):
            os.remove(sf)
    except Exception:
        pass
    try:
        if os.path.exists(pf):
            os.remove(pf)
    except Exception:
        pass

# -------------------------------
# ROUTES (original + new)
# -------------------------------
@app1.route("/")
def home():
    # ensure we load fresh case types before rendering (in case of external edit)
    load_case_types()
    return render_template("index.html", CASE_TYPES=CASE_TYPES)

@app1.route("/get_all_case_types", methods=["GET"])
def get_all_case_types():
    load_case_types()
    return jsonify({"case_types": CASE_TYPES})

# Add case type
@app1.route("/add_case_type", methods=["POST"])
def add_case_type():
    try:
        data = request.get_json() or {}
        new_type = (data.get("case_type") or "").strip()
        if not new_type:
            return jsonify({"error": "Empty case type name"}), 400
        load_case_types()
        if new_type in CASE_TYPES:
            return jsonify({"error": "Case type already exists"}), 400
        CASE_TYPES[new_type] = {}
        save_case_types()
        return jsonify({"message": "Case type added", "case_types": CASE_TYPES})
    except Exception as e:
        logging.exception("Error adding case type")
        return jsonify({"error": str(e)}), 500

# Delete case type
@app1.route("/delete_case_type", methods=["POST"])
def delete_case_type():
    try:
        data = request.get_json() or {}
        case_type = (data.get("case_type") or "").strip()
        if not case_type:
            return jsonify({"error": "Missing case_type"}), 400
        load_case_types()
        if case_type not in CASE_TYPES:
            return jsonify({"error": "Case type not found"}), 404
        CASE_TYPES.pop(case_type)
        save_case_types()
        return jsonify({"message": "Case type deleted", "case_types": CASE_TYPES})
    except Exception as e:
        logging.exception("Error deleting case type")
        return jsonify({"error": str(e)}), 500

# Add document under a case type
@app1.route("/add_document", methods=["POST"])
def add_document():
    try:
        data = request.get_json() or {}
        case_type = (data.get("case_type") or "").strip()
        doc_name = (data.get("document_name") or "").strip()
        pattern = (data.get("pattern") or "").strip()
        if not case_type or not doc_name:
            return jsonify({"error": "Missing case_type or document_name"}), 400
        load_case_types()
        if case_type not in CASE_TYPES:
            return jsonify({"error": "Case type not found"}), 404
        # pattern -> list of tokens (split by comma)
        patterns = [p.strip() for p in pattern.split(",") if p.strip()] if pattern else []
        CASE_TYPES[case_type][doc_name] = patterns
        save_case_types()
        return jsonify({"message": "Document added", "case_type": case_type, "documents": CASE_TYPES[case_type]})
    except Exception as e:
        logging.exception("Error adding document")
        return jsonify({"error": str(e)}), 500

# Edit document - can rename and update pattern
@app1.route("/edit_document", methods=["POST"])
def edit_document():
    try:
        data = request.get_json() or {}
        case_type = (data.get("case_type") or "").strip()
        old_name = (data.get("old_name") or "").strip()
        new_name = (data.get("new_name") or "").strip()
        pattern = (data.get("pattern") or "").strip()
        if not case_type or not old_name:
            return jsonify({"error": "Missing case_type or old_name"}), 400
        load_case_types()
        if case_type not in CASE_TYPES or old_name not in CASE_TYPES[case_type]:
            return jsonify({"error": "Document not found"}), 404
        # Build new pattern list
        patterns = [p.strip() for p in pattern.split(",") if p.strip()] if pattern else []
        # If renaming
        if new_name and new_name != old_name:
            CASE_TYPES[case_type].pop(old_name)
            CASE_TYPES[case_type][new_name] = patterns
        else:
            CASE_TYPES[case_type][old_name] = patterns
        save_case_types()
        return jsonify({"message": "Document updated", "case_type": case_type, "documents": CASE_TYPES[case_type]})
    except Exception as e:
        logging.exception("Error editing document")
        return jsonify({"error": str(e)}), 500

# Delete document
@app1.route("/delete_document", methods=["POST"])
def delete_document():
    try:
        data = request.get_json() or {}
        case_type = (data.get("case_type") or "").strip()
        doc_name = (data.get("document_name") or "").strip()
        if not case_type or not doc_name:
            return jsonify({"error": "Missing case_type or document_name"}), 400
        load_case_types()
        if case_type not in CASE_TYPES or doc_name not in CASE_TYPES[case_type]:
            return jsonify({"error": "Document not found"}), 404
        CASE_TYPES[case_type].pop(doc_name)
        save_case_types()
        return jsonify({"message": "Document deleted", "case_type": case_type, "documents": CASE_TYPES[case_type]})
    except Exception as e:
        logging.exception("Error deleting document")
        return jsonify({"error": str(e)}), 500
    
@app1.route("/edit_case_type", methods=["POST"])
def edit_case_type():
    data = request.get_json()
    old_case_type = data.get("old_case_type")
    new_case_type = data.get("new_case_type")

    if not old_case_type or not new_case_type:
        return jsonify({"error": "Both old and new case type names are required"}), 400

    try:
        # Assuming your case types are stored in a JSON file or dictionary
        with open("case_types.json", "r") as f:
            case_types = json.load(f)

        if old_case_type not in case_types:
            return jsonify({"error": "Old case type not found"}), 404

        if new_case_type in case_types:
            return jsonify({"error": "A case type with this name already exists"}), 400

        # Rename the key
        case_types[new_case_type] = case_types.pop(old_case_type)

        # Save updated data
        with open("case_types.json", "w") as f:
            json.dump(case_types, f, indent=4)

        return jsonify({"message": "Case type renamed successfully"}), 200

    except Exception as e:
        print("Error editing case type:", e)
        return jsonify({"error": str(e)}), 500


# keep original upload_sheet_or_excel, get_documents, run, send_email, stream_scan and SSE generator
@app1.route("/upload_sheet_or_excel", methods=["POST"])
def upload_sheet_or_excel():
    if "file" in request.files:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        try:
            df = pd.read_excel(path, nrows=0)
            cols = df.columns.tolist()
        except Exception as e:
            logging.exception("Error reading uploaded excel")
            return jsonify({"error": f"Failed to read Excel: {str(e)}"}, 500)
        return jsonify({"message": "File uploaded", "columns": cols, "file_path": path})

    data = request.get_json() or {}
    sheet_link = data.get("sheet_link", "").strip()
    if sheet_link:
        spreadsheet_id = extract_sheet_id(sheet_link)
        try:
            sheets = get_sheets_service()
            if sheets is None:
                raise RuntimeError("Sheets service not available")
            meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id, fields="sheets.properties").execute()
            sheets_list = meta.get("sheets", [])
            if not sheets_list:
                return jsonify({"error": "No sheets inside spreadsheet"}), 400
            first_title = sheets_list[0]["properties"]["title"]
            rng = f"'{first_title}'!1:1"
            resp = sheets.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=rng).execute()
            values = resp.get("values", [])
            headers = values[0] if values else []
            file_path = f"gsheet:{spreadsheet_id}:{first_title}"
            return jsonify({"message": "Google Sheet loaded", "columns": headers, "file_path": file_path})
        except Exception as e:
            logging.exception("Error reading Google Sheet")
            return jsonify({"error": f"Failed to read Google Sheet: {str(e)}"}, 500)

    return jsonify({"error": "No file or sheet_link provided"}), 400

@app1.route("/get_documents", methods=["POST"])
def get_documents():
    data = request.get_json()
    case_type = data.get("caseType")
    load_case_types()
    docs = CASE_TYPES.get(case_type, {})
    doc_list = [{"documentName": k, "defaultPattern": ", ".join(v)} for k, v in docs.items()]
    return jsonify({"documents": doc_list})

@app1.route("/run", methods=["POST"])
def run_search():
    """
    Main long-running process. Supports checkpoint/resume using a deterministic state file.
    Expects JSON with:
      - folder_link
      - excel_path
      - borrower_col
      - case_type
      - doc_patterns (dict)
      - email_to, email_cc, email_subject, message (optional)
    """
    data = request.get_json() or {}
    folder_link = data.get("folder_link")
    excel_path = data.get("excel_path")
    borrower_col = data.get("borrower_col")
    case_type = data.get("case_type")
    doc_patterns = data.get("doc_patterns", {})

    # optional email fields
    email_to = data.get("email_to", "")
    email_cc = data.get("email_cc", "")
    email_subject = data.get("email_subject", "Automated Report")
    email_message = data.get("message", "Please find the attached report.")

    # build state
    state_id = make_state_id(folder_link, excel_path, borrower_col, doc_patterns)
    state_path = state_file_for_id(state_id)
    partial_output = partial_output_for_id(state_id)

    # Read Excel or Google Sheet
    try:
        if isinstance(excel_path, str) and excel_path.startswith("gsheet:"):
            parts = excel_path.split(":", 2)
            spreadsheet_id = parts[1]
            sheet_name = parts[2] if len(parts) > 2 else None
            sheets = get_sheets_service()
            if sheets is None:
                return jsonify({"error": "Sheets service not available"}), 500
            result = sheets.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id, range=sheet_name
            ).execute()
            values = result.get("values", [])
            if not values:
                df = pd.DataFrame()
            else:
                df = pd.DataFrame(values[1:], columns=values[0])
        else:
            df = pd.read_excel(excel_path)
    except Exception as e:
        logging.exception("Error reading input sheet/file")
        return jsonify({"error": str(e)}), 500

    # Normalize borrower column
    if borrower_col not in df.columns and borrower_col != "":
        for c in df.columns:
            if str(c).strip().lower() == str(borrower_col).strip().lower():
                borrower_col = c
                break

    # Load state if exists
    processed_indices = set()
    try:
        state = load_state(state_id)
        if state:
            processed_indices = set(state.get("processed_indices", []))
            if os.path.exists(state.get("partial_output", partial_output)):
                try:
                    df_partial = pd.read_excel(state.get("partial_output", partial_output))
                    if len(df_partial) == len(df):
                        df = df_partial
                except:
                    pass
    except:
        logging.exception("Failed to load existing state; starting fresh")

    parent_id = extract_folder_id(folder_link)

    # Ensure result columns exist (these should APPEND, not reorder)
    for doc in doc_patterns.keys():
        if doc not in df.columns:
            df[doc] = ""
    if "Folder Link" not in df.columns:
        df["Folder Link"] = ""

    # Main row processing
    try:
        for idx, row in df.iterrows():
            if int(idx) in processed_indices:
                continue

            borrower = str(row.get(borrower_col, "")).strip().lower() if borrower_col else ""

            # find matching folder
            query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = safe_drive_list(query)
            folder_id = None
            for folder in results.get("files", []):
                if folder["name"].strip().lower() == borrower:
                    folder_id = folder["id"]
                    break

            if folder_id:
                df.at[idx, "Folder Link"] = f"https://drive.google.com/drive/folders/{folder_id}"
            else:
                df.at[idx, "Folder Link"] = "Folder Not Found"
                processed_indices.add(int(idx))
                save_state(state_id, {"processed_indices": list(processed_indices), "partial_output": partial_output})
                try:
                    df.to_excel(partial_output, index=False)
                except:
                    logging.exception("Failed to save partial output")
                continue

            # search docs
            for doc, pattern in doc_patterns.items():
                if not pattern:
                    df.at[idx, doc] = "Skipped"
                    continue
                patterns = [p.strip() for p in pattern.split(",") if p.strip()]
                links = search_files_with_regex(folder_id, patterns)
                df.at[idx, doc] = "; ".join(links)

            processed_indices.add(int(idx))

            # checkpoint
            try:
                df.to_excel(partial_output, index=False)
            except:
                logging.exception("Failed to save partial output")
            save_state(state_id, {"processed_indices": list(processed_indices), "partial_output": partial_output})

    except Exception as e:
        logging.exception("Error while processing rows")
        save_state(state_id, {
            "processed_indices": list(processed_indices),
            "partial_output": partial_output,
            "error": str(e),
        })
        return jsonify({"error": f"Processing interrupted: {str(e)}"}), 500

    # ---------- ✅ FIXED COLUMN ORDERING HERE ----------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"Report_{timestamp}.xlsx"

    try:
        # ORIGINAL COLUMNS FIRST
        original_cols = [
            c for c in df.columns
            if c not in (["Folder Link"] + list(doc_patterns.keys()))
        ]

        # NEW COLUMNS AT THE END
        new_cols = []
        if "Folder Link" in df.columns:
            new_cols.append("Folder Link")

        for d in doc_patterns.keys():
            if d in df.columns:
                new_cols.append(d)

        ordered_cols = original_cols + new_cols

        try:
            df = df.loc[:, ordered_cols]
        except:
            pass

        df.to_excel(output_file, index=False)

    except Exception as e:
        logging.exception("Failed to write final report")
        return jsonify({"error": f"Failed to write report: {str(e)}"}), 500
    # ---------------------------------------------------

    # cleanup
    cleanup_state(state_id)

    # send email if requested
    if email_to:
        try:
            msg = EmailMessage()
            msg['From'] = SENDER_EMAIL
            msg['To'] = email_to if isinstance(email_to, str) else ", ".join(email_to)
            if email_cc:
                msg['Cc'] = email_cc if isinstance(email_cc, str) else ", ".join(email_cc)
            msg['Subject'] = email_subject
            msg.set_content(email_message)

            with open(output_file, 'rb') as f:
                file_data = f.read()
                msg.add_attachment(file_data, maintype='application',
                                   subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                   filename=os.path.basename(output_file))

            with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
                smtp.starttls()
                smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                smtp.send_message(msg)

        except Exception as e:
            logging.exception("Email send failed")
            response = make_response(send_file(output_file, as_attachment=True))
            response.headers['X-Report-Filename'] = output_file
            response.headers['X-Email-Status'] = f"failed: {str(e)}"
            return response

    # return file
    response = make_response(send_file(output_file, as_attachment=True))
    response.headers['X-Report-Filename'] = output_file
    if email_to:
        response.headers['X-Email-Status'] = "success"
    return response


@app1.route("/send_email", methods=["POST"])
def send_email():
    # This endpoint remains as a utility if called separately.
    try:
        data = request.get_json()
        to_emails = data.get("emails", "")
        cc_emails = data.get("cc", "")
        subject = data.get("subject", "Automated Report")
        message_body = data.get("message", "Please find the attached report.")
        file_path = data.get("file_path", "")

        if not to_emails or not file_path:
            return jsonify({"error": "Missing recipient emails or file path"}), 400

        to_list = [e.strip() for e in to_emails.split(",") if e.strip()]
        cc_list = [e.strip() for e in cc_emails.split(",") if e.strip()] if cc_emails else []

        if not to_list:
            return jsonify({"error": "No valid recipient email provided"}), 400

        if not os.path.exists(file_path):
            if os.path.exists(os.path.join(os.getcwd(), file_path)):
                file_path = os.path.join(os.getcwd(), file_path)
            else:
                return jsonify({"error": f"Report file not found: {file_path}"}), 404

        msg = EmailMessage()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ", ".join(to_list)
        if cc_list:
            msg['Cc'] = ", ".join(cc_list)
        msg['Subject'] = subject
        msg.set_content(message_body)

        with open(file_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(file_path)
            msg.add_attachment(file_data, maintype='application', subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=file_name)

        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)

        return jsonify({"message": "Email sent successfully!"})
    except Exception as e:
        logging.exception("Error sending email")
        return jsonify({"error": str(e)}), 500

# -------------------------------
# SSE streaming endpoint for live row-by-row folder scan
# -------------------------------
def generate_sse_scan(parent_link, excel_path, borrower_col):
    try:
        parent_id = extract_folder_id(parent_link)
        if isinstance(excel_path, str) and excel_path.startswith("gsheet:"):
            parts = excel_path.split(":", 2)
            spreadsheet_id = parts[1]
            sheet_name = parts[2] if len(parts) > 2 else None
            sheets = get_sheets_service()
            if sheets is None:
                payload = {"error": "Sheets service not available"}
                yield f"data: {json.dumps(payload)}\n\n"
                return
            result = sheets.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=sheet_name).execute()
            values = result.get("values", [])
            if not values:
                df = pd.DataFrame()
            else:
                df = pd.DataFrame(values[1:], columns=values[0])
        else:
            df = pd.read_excel(excel_path)
    except Exception as e:
        logging.exception("Error reading sheet/file for SSE scan")
        payload = {"error": f"Failed to read file/sheet: {str(e)}"}
        yield f"data: {json.dumps(payload)}\n\n"
        return

    try:
        folder_query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        parent_children = safe_drive_list(folder_query).get("files", [])
        folder_map = {f["name"].strip().lower(): f["id"] for f in parent_children}
    except Exception as e:
        logging.exception("Error listing parent children")
        payload = {"error": f"Failed to list parent folder: {str(e)}"}
        yield f"data: {json.dumps(payload)}\n\n"
        return

    for idx, row in df.iterrows():
        borrower_value = str(row.get(borrower_col, "")).strip()
        borrower_normal = borrower_value.lower()
        status = "not_found"
        folder_id = None
        structure = []

        if borrower_normal in folder_map:
            status = "found"
            folder_id = folder_map[borrower_normal]
            try:
                structure = print_folder_structure(folder_id)
            except Exception as e:
                logging.exception("Error building folder structure")
                structure = [{"error": f"Failed to enumerate folder: {str(e)}"}]

        payload = {
            "borrower": borrower_value,
            "index": int(idx),
            "status": status,
            "folder_id": folder_id,
            "structure": structure
        }
        yield f"data: {json.dumps(payload)}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"

@app1.route("/stream_scan")
def stream_scan():
    folder_link = request.args.get("folder_link", "")
    excel_path = request.args.get("excel_path", "")
    borrower_col = request.args.get("borrower_col", "")
    excel_path = urllib.parse.unquote_plus(excel_path)
    return Response(generate_sse_scan(folder_link, excel_path, borrower_col),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    # note: threaded=True is kept; if you use a production server, use gunicorn/uvicorn instead
    app1.run(debug=True, threaded=True)
