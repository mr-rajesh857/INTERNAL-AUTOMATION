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
from typing import List, Union
import shutil

# -------------------------------
# CONFIGURATION
# -------------------------------
app1 = Blueprint("app1", __name__, template_folder="templates", static_folder="static")
# app1 = Flask(__name__)
SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets.readonly"
]

# Email config (consider moving to env vars)
SENDER_EMAIL = "rajesh@legodesk.com"
SENDER_PASSWORD = "kjfh uzeu dynh kkfe"

logging.basicConfig(filename="drive_search_log.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Do NOT build Drive/Sheets globally at import time on Windows/old httplib2 – build lazily instead
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
        creds.refresh(GoogleAuthRequest())
    except Exception as e:
        logging.exception("Failed to refresh credentials")
    return creds

def get_drive_service():
    global _drive_service
    try:
        creds = build_credentials()
        _drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        logging.exception("Failed to build drive service")
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
# HELPER FUNCTIONS
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
                print("[safe_drive_list] Max retries reached – returning partial results.")
                break

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

def normalize_filename(filename: str) -> str:
    """
    Normalize filename by:
    - Converting to lowercase
    - Replacing separators (_, -, space) with single space
    - Removing file extension
    """
    try:
        name_without_ext = filename.rsplit('.', 1)[0].lower()
        normalized = re.sub(r'[-_\s.]+', ' ', name_without_ext)
        return normalized.strip()
    except Exception as e:
        logging.error(f"Error normalizing filename '{filename}': {e}")
        return filename.lower()

def parse_search_pattern(pattern: str) -> List[List[str]]:
    """
    Parse search pattern with OR and AND logic.
    """
    try:
        if not pattern or not pattern.strip():
            return [[]]
        
        or_groups = [g.strip() for g in pattern.split('|') if g.strip()]
        
        parsed_groups = []
        for group in or_groups:
            if '_' in group or '-' in group:
                and_terms = [group.lower().strip()]
            else:
                and_terms = [term.lower().strip() for term in group.split(',') if term.strip()]
            
            parsed_groups.append(and_terms)
        
        return parsed_groups if parsed_groups else [[]]
    
    except Exception as e:
        logging.error(f"Error parsing pattern '{pattern}': {e}")
        return [[pattern.lower().strip()]]

def matches_pattern(normalized_filename: str, pattern_groups: List[List[str]]) -> bool:
    """
    Check if normalized filename matches any of the OR groups.
    """
    try:
        if not pattern_groups or not pattern_groups[0]:
            return True
        
        for and_group in pattern_groups:
            all_match = True
            
            for term in and_group:
                if '_' in term or '-' in term:
                    variations = [
                        term,
                        term.replace('_', ' '),
                        term.replace('-', ' '),
                        term.replace('_', '-'),
                        term.replace('-', '_')
                    ]
                    
                    term_found = any(
                        variation in normalized_filename 
                        for variation in variations
                    )
                else:
                    term_found = bool(re.search(rf'\b{re.escape(term)}\b', normalized_filename, re.IGNORECASE))
                
                if not term_found:
                    all_match = False
                    break
            
            if all_match:
                return True
        
        return False
    
    except Exception as e:
        logging.error(f"Error matching pattern for '{normalized_filename}': {e}")
        return False

def search_files_with_regex(folder_id: str, pattern: Union[str, List[str]], drive_service=None, max_depth: int = 10) -> List[str]:
    """
    Enhanced recursive search with better error handling and depth limiting.
    """
    if drive_service is None:
        try:
            drive_service = get_drive_service()
        except Exception as e:
            logging.error(f"Failed to get Drive service: {e}")
            return [f"Error: {str(e)}"]
    
    matched_links = []
    visited_folders = set()
    
    if isinstance(pattern, list):
        pattern = ','.join(pattern)
    
    pattern_groups = parse_search_pattern(pattern)
    
    def recurse(fid: str, depth: int = 0):
        nonlocal matched_links, visited_folders
        
        if depth > max_depth:
            logging.warning(f"Max recursion depth {max_depth} reached for folder {fid}")
            return
        
        if fid in visited_folders:
            return
        visited_folders.add(fid)
        
        try:
            results = safe_drive_list(f"'{fid}' in parents and trashed=false")
            
            if not results or 'files' not in results:
                logging.warning(f"No results returned for folder {fid}")
                return
            
            files = results.get("files", [])
            
            for item in files:
                try:
                    item_name = item.get("name", "")
                    item_id = item.get("id", "")
                    mime_type = item.get("mimeType", "")
                    
                    if not item_name or not item_id:
                        continue
                    
                    if mime_type == "application/vnd.google-apps.folder":
                        recurse(item_id, depth + 1)
                    else:
                        normalized = normalize_filename(item_name)
                        
                        if matches_pattern(normalized, pattern_groups):
                            file_link = f"https://drive.google.com/file/d/{item_id}/view"
                            matched_links.append(file_link)
                            logging.info(f"Match found: {item_name} -> {file_link}")
                
                except Exception as item_error:
                    logging.error(f"Error processing item in folder {fid}: {item_error}")
                    continue
        
        except Exception as folder_error:
            logging.error(f"Error accessing folder {fid}: {folder_error}")
            return
    
    try:
        recurse(folder_id)
    except Exception as e:
        logging.error(f"Critical error in search_files_with_regex: {e}")
        return [f"Error: {str(e)}"]
    
    return matched_links if matched_links else ["Not Found"]

# -------------------------------
# ✅ NEW: ENHANCED DUPLICATE FOLDER DETECTION
# -------------------------------
def is_descendant_of(folder_id, ancestor_id, drive_service):
    """
    Check if folder_id is a descendant (child/grandchild/etc) of ancestor_id.
    """
    try:
        current_id = folder_id
        max_depth = 50
        
        for _ in range(max_depth):
            file_meta = drive_service.files().get(
                fileId=current_id,
                fields="parents"
            ).execute()
            
            parents = file_meta.get("parents", [])
            
            if not parents:
                return False
            
            if ancestor_id in parents:
                return True
            
            current_id = parents[0]
        
        return False
        
    except Exception as e:
        logging.error(f"Error checking ancestry for {folder_id}: {e}")
        return False

def get_all_direct_child_folders(parent_id, borrower_name, drive_service=None):
    """
    Find ALL direct child folders matching borrower name (case-insensitive).
    Simpler than global search - just searches direct children.
    
    Returns:
        List[dict]: All matching folders with {id, name}
    """
    if drive_service is None:
        drive_service = get_drive_service()
    
    if not borrower_name or not borrower_name.strip():
        return []
    
    borrower_lower = borrower_name.strip().lower()
    matching = []
    
    # Query direct children only
    query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    
    try:
        results = safe_drive_list(query)
        
        # Find ALL folders with matching name (case-insensitive)
        for folder in results.get("files", []):
            folder_name_lower = folder["name"].strip().lower()
            
            if folder_name_lower == borrower_lower:
                matching.append({
                    "id": folder["id"],
                    "name": folder["name"]  # Keep original case
                })
                logging.info(f"  ✅ Found matching folder: '{folder['name']}' ({folder['id']})")
        
        if matching:
            logging.info(f"🔍 Total {len(matching)} folder(s) matching '{borrower_name}'")
        else:
            logging.info(f"❌ No folders found matching '{borrower_name}'")
        
        return matching
        
    except Exception as e:
        logging.error(f"Error finding folders for '{borrower_name}': {e}")
        return []
    
def get_all_matching_folders(parent_id, borrower_name, drive_service=None):
    """
    Find ALL folders matching borrower name, including duplicates.
    Uses Drive API search to find all folders with exact name match,
    then verifies they are descendants of parent_id.
    
    Returns:
        List[dict]: All matching folders with {id, name}
    """
    if drive_service is None:
        drive_service = get_drive_service()
    
    if not borrower_name or not borrower_name.strip():
        return []
    
    borrower_clean = borrower_name.strip()
    matching = []
    
    # Escape single quotes in folder name for Drive API query
    escaped_name = borrower_clean.replace("'", "\\'")
    
    # Search for ALL folders with this exact name (case-sensitive in Drive API)
    query = f"mimeType='application/vnd.google-apps.folder' and name='{escaped_name}' and trashed=false"
    
    try:
        results = safe_drive_list(query)
        
        for folder in results.get("files", []):
            folder_id = folder["id"]
            folder_name = folder["name"]
            
            # Verify it's a descendant of parent_id
            if is_descendant_of(folder_id, parent_id, drive_service):
                matching.append({
                    "id": folder_id,
                    "name": folder_name
                })
                logging.info(f"  ✅ Found matching folder: '{folder_name}' ({folder_id})")
        
        logging.info(f"🔍 Total {len(matching)} folder(s) matching '{borrower_clean}' under parent")
        return matching
        
    except Exception as e:
        logging.error(f"Error finding folders for '{borrower_clean}': {e}")
        return []


def extract_folder_ids_from_links(folder_link_cell):
    """
    Parse "Folder Link" column to extract folder IDs.
    
    Input: "https://drive.google.com/drive/folders/ID1; https://..."
    Output: ["ID1", "ID2", ...]
    """
    if not folder_link_cell or folder_link_cell == "Folder Not Found":
        return []
    
    links = [link.strip() for link in str(folder_link_cell).split(";")]
    folder_ids = []
    
    for link in links:
        fid = extract_folder_id(link)
        if fid:
            folder_ids.append(fid)
    
    return folder_ids

# -------------------------------
# State helpers for checkpointing
# -------------------------------
_state_lock = Lock()

def make_state_id(folder_link, excel_path, borrower_col, doc_patterns):
    """Generate deterministic state ID from inputs"""
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
    """Get state file path"""
    return os.path.join(STATE_DIR, f"state_{state_id}.json")

def partial_output_for_id(state_id):
    """Get partial output file path"""
    return os.path.join("uploads", f"partial_{state_id}.xlsx")

def load_state_safe(state_id, max_retries=3):
    """Load state with retry mechanism for corrupted files"""
    path = state_file_for_id(state_id)
    
    for attempt in range(max_retries):
        if not os.path.exists(path):
            logging.info(f"No existing state found for {state_id}")
            return None
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
                
            if not isinstance(state, dict):
                raise ValueError("Invalid state structure")
            
            if "processed_indices" not in state:
                state["processed_indices"] = []
            
            if "total_rows" not in state:
                state["total_rows"] = 0
            
            logging.info(f"✅ State loaded: {len(state['processed_indices'])}/{state.get('total_rows', 0)} rows processed")
            return state
            
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"⚠️ State file corrupted (attempt {attempt+1}/{max_retries}): {e}")
            
            backup_path = path + ".backup"
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, path)
                    logging.info("Restored state from backup")
                    continue
                except Exception as backup_err:
                    logging.error(f"Failed to restore from backup: {backup_err}")
            
            if attempt == max_retries - 1:
                logging.error("❌ All state load attempts failed, starting fresh")
                return None
            
            time.sleep(0.5)
    
    return None

def save_state_atomic(state_id, state_obj):
    """Save state atomically with backup"""
    path = state_file_for_id(state_id)
    temp_path = path + ".tmp"
    backup_path = path + ".backup"
    
    try:
        with _state_lock:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state_obj, f, indent=2, ensure_ascii=False)
            
            if os.path.exists(path):
                try:
                    shutil.copy2(path, backup_path)
                except Exception as backup_err:
                    logging.warning(f"Failed to create backup: {backup_err}")
            
            if os.name == 'nt':
                if os.path.exists(path):
                    os.remove(path)
            os.rename(temp_path, path)
            
            processed = len(state_obj.get('processed_indices', []))
            total = state_obj.get('total_rows', 0)
            logging.info(f"💾 State saved: {processed}/{total} rows ({state_obj.get('progress_percent', 0):.1f}%)")
            
    except Exception as e:
        logging.exception("Failed to save state atomically")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def save_state_enhanced(state_id, state_obj, current_row=None, error=None):
    """Enhanced state save with recovery metadata and progress tracking"""
    state_obj["last_save"] = datetime.now().isoformat()
    state_obj["total_rows"] = state_obj.get("total_rows", 0)
    
    if current_row is not None:
        state_obj["last_processed_row"] = current_row
        
    processed = len(state_obj.get("processed_indices", []))
    total = state_obj.get("total_rows", 0)
    if total > 0:
        state_obj["progress_percent"] = round((processed / total) * 100, 2)
    else:
        state_obj["progress_percent"] = 0
    
    if error:
        state_obj["last_error"] = str(error)
        state_obj["error_timestamp"] = datetime.now().isoformat()
    
    save_state_atomic(state_id, state_obj)
    
def is_processing_complete(state_id, state_obj):
    """Check if processing is complete"""
    processed = len(state_obj.get("processed_indices", []))
    total = state_obj.get("total_rows", 0)
    
    if total == 0:
        return False
    
    is_complete = processed >= total
    
    if is_complete:
        logging.info(f"✅ Processing complete: {processed}/{total} rows")
    
    return is_complete

def cleanup_state(state_id, reason="completed"):
    """Remove state file and partial file"""
    sf = state_file_for_id(state_id)
    pf = partial_output_for_id(state_id)
    backup = sf + ".backup"
    
    files_removed = []
    
    try:
        if os.path.exists(sf):
            os.remove(sf)
            files_removed.append("state")
    except Exception as e:
        logging.warning(f"Failed to remove state file: {e}")
    
    try:
        if os.path.exists(backup):
            os.remove(backup)
            files_removed.append("backup")
    except Exception as e:
        logging.warning(f"Failed to remove backup file: {e}")
    
    try:
        if os.path.exists(pf):
            os.remove(pf)
            files_removed.append("partial")
    except Exception as e:
        logging.warning(f"Failed to remove partial file: {e}")
    
    if files_removed:
        logging.info(f"🗑️ Cleanup ({reason}): Removed {', '.join(files_removed)} files")

def cleanup_old_states(max_age_hours=24):
    """Clean up state files older than max_age_hours"""
    if not os.path.exists(STATE_DIR):
        return
    
    now = datetime.now()
    cleaned = 0
    
    for filename in os.listdir(STATE_DIR):
        if not filename.startswith("state_") or not filename.endswith(".json"):
            continue
        
        filepath = os.path.join(STATE_DIR, filename)
        
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            age_hours = (now - mtime).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                state_id = filename.replace("state_", "").replace(".json", "")
                cleanup_state(state_id, reason=f"stale ({age_hours:.1f}h old)")
                cleaned += 1
        
        except Exception as e:
            logging.error(f"Error checking state file {filename}: {e}")
    
    if cleaned > 0:
        logging.info(f"🧹 Cleaned up {cleaned} stale state files")

def save_checkpoint(state_id, processed_indices, partial_output, df, total_rows, current_row):
    """Save checkpoint during processing"""
    try:
        df.to_excel(partial_output, index=False)
        save_state_enhanced(state_id, {
            "processed_indices": list(processed_indices),
            "partial_output": partial_output,
            "total_rows": total_rows
        }, current_row=current_row)
    except Exception as e:
        logging.error(f"Checkpoint save failed at row {current_row}: {e}")

# -------------------------------
# ROUTES
# -------------------------------
@app1.route("/")
def home():
    load_case_types()
    return render_template("index.html", CASE_TYPES=CASE_TYPES)

@app1.route("/get_all_case_types", methods=["GET"])
def get_all_case_types():
    load_case_types()
    return jsonify({"case_types": CASE_TYPES})

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
        CASE_TYPES[case_type][doc_name] = pattern
        save_case_types()
        return jsonify({"message": "Document added", "case_type": case_type, "documents": CASE_TYPES[case_type]})
    except Exception as e:
        logging.exception("Error adding document")
        return jsonify({"error": str(e)}), 500

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
        if new_name and new_name != old_name:
            CASE_TYPES[case_type].pop(old_name)
            CASE_TYPES[case_type][new_name] = pattern
        else:
            CASE_TYPES[case_type][old_name] = pattern
        save_case_types()
        return jsonify({"message": "Document updated", "case_type": case_type, "documents": CASE_TYPES[case_type]})
    except Exception as e:
        logging.exception("Error editing document")
        return jsonify({"error": str(e)}), 500

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
        with open("case_types.json", "r") as f:
            case_types = json.load(f)

        if old_case_type not in case_types:
            return jsonify({"error": "Old case type not found"}), 404

        if new_case_type in case_types:
            return jsonify({"error": "A case type with this name already exists"}), 400

        case_types[new_case_type] = case_types.pop(old_case_type)

        with open("case_types.json", "w") as f:
            json.dump(case_types, f, indent=4)

        return jsonify({"message": "Case type renamed successfully"}), 200

    except Exception as e:
        print("Error editing case type:", e)
        return jsonify({"error": str(e)}), 500

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
            return jsonify({"error": f"Failed to read Excel: {str(e)}"}), 500
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
            return jsonify({"error": f"Failed to read Google Sheet: {str(e)}"}), 500

    return jsonify({"error": "No file or sheet_link provided"}), 400

@app1.route("/get_documents", methods=["POST"])
def get_documents():
    data = request.get_json()
    case_type = data.get("caseType")
    load_case_types()
    docs = CASE_TYPES.get(case_type, {})
    doc_list = [{"documentName": k, "defaultPattern": v if isinstance(v, str) else ", ".join(v)} for k, v in docs.items()]
    return jsonify({"documents": doc_list})

@app1.route("/run", methods=["POST"])
def run_search_enhanced():
    """
    ✅ FIXED: Enhanced run with duplicate folder support
    """
    data = request.get_json() or {}
    folder_link = data.get("folder_link")
    excel_path = data.get("excel_path")
    borrower_col = data.get("borrower_col")
    case_type = data.get("case_type")
    doc_patterns = data.get("doc_patterns", {})
    
    email_to = data.get("email_to", "")
    email_cc = data.get("email_cc", "")
    email_subject = data.get("email_subject", "Automated Report")
    email_message = data.get("message", "Please find the attached report.")
    
    state_id = make_state_id(folder_link, excel_path, borrower_col, doc_patterns)
    partial_output = partial_output_for_id(state_id)
    
    logging.info(f"🚀 Starting process with state_id: {state_id}")
    
    # Load input data
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
            df = pd.DataFrame(values[1:], columns=values[0]) if values else pd.DataFrame()
        else:
            df = pd.read_excel(excel_path)
    except Exception as e:
        logging.exception("Error reading input")
        return jsonify({"error": f"Failed to read input: {str(e)}"}), 500
    
    if df.empty:
        return jsonify({"error": "Input file is empty"}), 400
    
    if borrower_col not in df.columns and borrower_col != "":
        for c in df.columns:
            if str(c).strip().lower() == str(borrower_col).strip().lower():
                borrower_col = c
                break
    
    # Load or initialize state
    processed_indices = set()
    resume_mode = False
    
    state = load_state_safe(state_id)
    if state:
        processed_indices = set(state.get("processed_indices", []))
        resume_mode = True
        
        if is_processing_complete(state_id, state):
            logging.info("⚠️ Process already completed, cleaning up old state")
            cleanup_state(state_id, reason="already_completed")
            processed_indices = set()
            resume_mode = False
        else:
            logging.info(f"🔄 RESUMING: {len(processed_indices)}/{len(df)} rows already processed")
        
        if resume_mode and os.path.exists(partial_output):
            try:
                df_partial = pd.read_excel(partial_output)
                if len(df_partial) == len(df):
                    df = df_partial
                    logging.info("✅ Loaded partial results successfully")
            except Exception as e:
                logging.warning(f"Could not load partial results: {e}")
    
    # Initialize output columns
    parent_id = extract_folder_id(folder_link)
    
    for doc in doc_patterns.keys():
        if doc not in df.columns:
            df[doc] = ""
    if "Folder Link" not in df.columns:
        df["Folder Link"] = ""
    
    initial_state = {
        "processed_indices": list(processed_indices),
        "partial_output": partial_output,
        "total_rows": len(df),
        "started_at": datetime.now().isoformat(),
        "resume_mode": resume_mode
    }
    save_state_enhanced(state_id, initial_state)
    
    # ============================================
    # ✅ FIXED: Process rows with duplicate folder support
    # ============================================
    errors = []
    checkpoint_frequency = 5
    drive_service = get_drive_service()
    
    try:
        for idx, row in df.iterrows():
            if int(idx) in processed_indices:
                continue
            
            try:
                borrower = str(row.get(borrower_col, "")).strip()
                
                # ✅ FIX: Find ALL matching direct child folders (case-insensitive)
                matching_folders_info = get_all_direct_child_folders(parent_id, borrower, drive_service)
                
                if not matching_folders_info:
                    df.at[idx, "Folder Link"] = "Folder Not Found"
                    processed_indices.add(int(idx))
                    
                    if int(idx) % checkpoint_frequency == 0:
                        save_checkpoint(state_id, processed_indices, partial_output, df, len(df), int(idx))
                    
                    continue
                
                # ✅ FIX: Store ALL folder links
                matching_folders = [f["id"] for f in matching_folders_info]
                folder_links = [
                    f"https://drive.google.com/drive/folders/{fid}" 
                    for fid in matching_folders
                ]
                df.at[idx, "Folder Link"] = "; ".join(folder_links)
                
                logging.info(f"Row {idx}: Found {len(matching_folders)} folder(s) for '{borrower}'")
                
                # ✅ FIX: Search documents in ALL matching folders
                for doc, pattern in doc_patterns.items():
                    if not pattern:
                        df.at[idx, doc] = "Skipped"
                        continue
                    
                    all_doc_links = []
                    
                    for folder_id in matching_folders:
                        try:
                            links = search_files_with_regex(folder_id, pattern, drive_service)
                            
                            if links and links != ["Not Found"]:
                                all_doc_links.extend(links)
                        except Exception as search_err:
                            logging.error(f"Search error in folder {folder_id} for doc '{doc}': {search_err}")
                            errors.append(f"Row {idx}, Folder {folder_id}, Doc '{doc}': {str(search_err)}")
                    
                    # Store combined results
                    if all_doc_links:
                        df.at[idx, doc] = "; ".join(all_doc_links)
                        logging.info(f"Row {idx}, Doc '{doc}': Found {len(all_doc_links)} file(s)")
                    else:
                        df.at[idx, doc] = "Not Found"
                
                processed_indices.add(int(idx))
                
                if int(idx) % checkpoint_frequency == 0 or int(idx) == len(df) - 1:
                    save_checkpoint(state_id, processed_indices, partial_output, df, len(df), int(idx))
            
            except Exception as row_err:
                logging.exception(f"Error processing row {idx}")
                errors.append(f"Row {idx}: {str(row_err)}")
                processed_indices.add(int(idx))
                continue
    
    except Exception as e:
        logging.exception("Critical error during processing")
        save_state_enhanced(state_id, {
            "processed_indices": list(processed_indices),
            "partial_output": partial_output,
            "total_rows": len(df),
            "critical_error": True
        }, error=str(e))
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    # Check completion
    final_state = {
        "processed_indices": list(processed_indices),
        "partial_output": partial_output,
        "total_rows": len(df)
    }
    
    if is_processing_complete(state_id, final_state):
        logging.info("✅ Processing 100% complete - cleaning up state files")
    
    # Generate final report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"Report_{timestamp}.xlsx"
    
    try:
        original_cols = [c for c in df.columns if c not in (["Folder Link"] + list(doc_patterns.keys()))]
        new_cols = []
        if "Folder Link" in df.columns:
            new_cols.append("Folder Link")
        for d in doc_patterns.keys():
            if d in df.columns:
                new_cols.append(d)
        
        ordered_cols = original_cols + new_cols
        df = df.loc[:, ordered_cols]
        df.to_excel(output_file, index=False)
        
        logging.info(f"📊 Final report generated: {output_file}")
        
    except Exception as e:
        logging.exception("Failed to generate final report")
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500
    
    cleanup_state(state_id, reason="completed")
    
    # Send email
    if email_to:
        try:
            msg = EmailMessage()
            msg['From'] = SENDER_EMAIL
            msg['To'] = email_to if isinstance(email_to, str) else ", ".join(email_to)
            if email_cc:
                msg['Cc'] = email_cc if isinstance(email_cc, str) else ", ".join(email_cc)
            msg['Subject'] = email_subject
            
            body = email_message
            if errors:
                body += f"\n\n⚠️ Note: {len(errors)} errors occurred during processing:\n"
                body += "\n".join(errors[:10])
                if len(errors) > 10:
                    body += f"\n... and {len(errors) - 10} more errors (see log)"
            
            msg.set_content(body)
            
            with open(output_file, 'rb') as f:
                file_data = f.read()
                msg.add_attachment(file_data, maintype='application',
                                 subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                 filename=os.path.basename(output_file))
            
            with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
                smtp.starttls()
                smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                smtp.send_message(msg)
            
            logging.info("📧 Email sent successfully")
            
        except Exception as e:
            logging.exception("Email send failed")
            response = make_response(send_file(output_file, as_attachment=True))
            response.headers['X-Report-Filename'] = output_file
            response.headers['X-Email-Status'] = f"failed: {str(e)}"
            return response
    
    response = make_response(send_file(output_file, as_attachment=True))
    response.headers['X-Report-Filename'] = output_file
    if email_to:
        response.headers['X-Email-Status'] = "success"
    if errors:
        response.headers['X-Processing-Errors'] = str(len(errors))
    
    return response

@app1.route("/send_email", methods=["POST"])
def send_email():
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

# ============================================
# ✅ FIXED: Enhanced SSE Scan with duplicate support
# ============================================
def generate_sse_scan(parent_link, excel_path, borrower_col):
    """
    Enhanced SSE scan with duplicate support (direct children only).
    """
    try:
        parent_id = extract_folder_id(parent_link)
        
        if isinstance(excel_path, str) and excel_path.startswith("gsheet:"):
            parts = excel_path.split(":", 2)
            spreadsheet_id = parts[1]
            sheet_name = parts[2] if len(parts) > 2 else None
            sheets = get_sheets_service()
            if sheets is None:
                yield f"data: {json.dumps({'error': 'Sheets service not available'})}\n\n"
                return
            result = sheets.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id, range=sheet_name
            ).execute()
            values = result.get("values", [])
            df = pd.DataFrame(values[1:], columns=values[0]) if values else pd.DataFrame()
        else:
            df = pd.read_excel(excel_path)
            
    except Exception as e:
        logging.exception("Error reading file for SSE scan")
        yield f"data: {json.dumps({'error': f'Failed to read file: {str(e)}'})}\n\n"
        return

    drive_service = get_drive_service()
    
    logging.info(f"📁 SSE Scan: Starting scan of parent folder {parent_id}")

    # Process each borrower
    for idx, row in df.iterrows():
        borrower_value = str(row.get(borrower_col, "")).strip()
        
        # ✅ FIX: Use enhanced search to find ALL matching direct child folders
        matching_folders = get_all_direct_child_folders(parent_id, borrower_value, drive_service)
        
        if not matching_folders:
            payload = {
                "borrower": borrower_value,
                "index": int(idx),
                "status": "not_found",
                "folder_count": 0,
                "folders": []
            }
            yield f"data: {json.dumps(payload)}\n\n"
            continue
        
        # Scan structure for each folder
        folder_data = []
        for folder_info in matching_folders:
            folder_id = folder_info["id"]
            folder_name = folder_info["name"]
            
            try:
                structure = print_folder_structure(folder_id)
                folder_data.append({
                    "folder_id": folder_id,
                    "folder_name": folder_name,
                    "structure": structure
                })
                logging.info(f"  ✅ SSE Scan: Scanned '{folder_name}' ({folder_id})")
            except Exception as e:
                logging.exception(f"Error scanning folder {folder_id}")
                folder_data.append({
                    "folder_id": folder_id,
                    "folder_name": folder_name,
                    "error": str(e)
                })
        
        logging.info(f"📁 SSE Scan Row {idx}: Found {len(folder_data)} folder(s) for '{borrower_value}'")
        
        payload = {
            "borrower": borrower_value,
            "index": int(idx),
            "status": "found",
            "folder_count": len(folder_data),
            "folders": folder_data
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


@app1.route("/debug_list_folders", methods=["POST"])
def debug_list_folders():
    """
    Debug endpoint to show ALL folders in parent with duplicates highlighted.
    """
    data = request.get_json() or {}
    folder_link = data.get("folder_link", "").strip()
    
    if not folder_link:
        return jsonify({"error": "No folder link provided"}), 400
    
    parent_id = extract_folder_id(folder_link)
    
    try:
        drive_service = get_drive_service()
        query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = safe_drive_list(query)
        
        folders = results.get("files", [])
        
        # Group by name to find duplicates
        from collections import defaultdict
        folder_groups = defaultdict(list)
        
        for f in folders:
            folder_groups[f["name"].strip().lower()].append({
                "id": f["id"],
                "name": f["name"],
                "created": "unknown"  # Drive API doesn't return creation date in list
            })
        
        # Format response
        duplicates = {}
        singles = {}
        
        for name, folder_list in folder_groups.items():
            if len(folder_list) > 1:
                duplicates[name] = folder_list
            else:
                singles[name] = folder_list[0]
        
        return jsonify({
            "parent_id": parent_id,
            "total_folders": len(folders),
            "unique_names": len(folder_groups),
            "duplicates_count": len(duplicates),
            "duplicates": duplicates,
            "singles": singles
        })
        
    except Exception as e:
        logging.exception("Error listing folders")
        return jsonify({"error": str(e)}), 500
    

def init_state_cleanup():
    """Run once at startup to clean old states"""
    try:
        cleanup_old_states(max_age_hours=24)
    except Exception as e:
        logging.error(f"Startup cleanup failed: {e}")

init_state_cleanup()
# Add this NEW route to app.py (before if __name__ == "__main__":)


if __name__ == "__main__":
    app1.run(debug=True, threaded=True)
