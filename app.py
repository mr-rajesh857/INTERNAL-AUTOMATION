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
from typing import List, Union  # ✅ ADDED THIS IMPORT
import shutil

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
SENDER_EMAIL = ""
SENDER_PASSWORD = ""

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

def normalize_filename(filename: str) -> str:
    """
    Normalize filename by:
    - Converting to lowercase
    - Replacing separators (_, -, space) with single space
    - Removing file extension
    """
    try:
        name_without_ext = filename.rsplit('.', 1)[0].lower()
        # Replace common separators with space
        normalized = re.sub(r'[-_\s.]+', ' ', name_without_ext)
        return normalized.strip()
    except Exception as e:
        logging.error(f"Error normalizing filename '{filename}': {e}")
        return filename.lower()

def parse_search_pattern(pattern: str) -> List[List[str]]:
    """
    Parse search pattern with OR and AND logic.
    
    Format examples:
    - "loan,statement" → AND condition: both "loan" AND "statement" must be present
    - "loan_statement|loan statement" → OR condition: either "loan_statement" OR "loan statement"
    - "bank,statement|bank_statement" → "bank" AND ("statement" OR "bank_statement")
    
    Syntax:
    - Comma (,) = AND operator (all tokens must be present)
    - Pipe (|) = OR operator (at least one group must match)
    - Underscore/hyphen in quoted phrases treated as single term
    
    Returns:
    List of OR groups, where each group is a list of AND terms
    
    Example:
    "loan_statement|loan statement,copy" returns:
    [['loan_statement'], ['loan', 'statement', 'copy']]
    """
    try:
        if not pattern or not pattern.strip():
            return [[]]
        
        # Split by OR operator (|)
        or_groups = [g.strip() for g in pattern.split('|') if g.strip()]
        
        parsed_groups = []
        for group in or_groups:
            # Check if group contains underscores/hyphens (treat as single term)
            if '_' in group or '-' in group:
                # This is a single compound term
                and_terms = [group.lower().strip()]
            else:
                # Split by comma for AND terms
                and_terms = [term.lower().strip() for term in group.split(',') if term.strip()]
            
            parsed_groups.append(and_terms)
        
        return parsed_groups if parsed_groups else [[]]
    
    except Exception as e:
        logging.error(f"Error parsing pattern '{pattern}': {e}")
        return [[pattern.lower().strip()]]

def matches_pattern(normalized_filename: str, pattern_groups: List[List[str]]) -> bool:
    """
    Check if normalized filename matches any of the OR groups.
    Each OR group must have all its AND terms present.
    
    Args:
        normalized_filename: Normalized file name
        pattern_groups: List of OR groups from parse_search_pattern()
    
    Returns:
        True if any OR group fully matches
    """
    try:
        if not pattern_groups or not pattern_groups[0]:
            return True  # Empty pattern matches everything
        
        # Check each OR group
        for and_group in pattern_groups:
            # All terms in this AND group must match
            all_match = True
            
            for term in and_group:
                # Handle compound terms (with underscore/hyphen)
                if '_' in term or '-' in term:
                    # Try matching the exact compound term
                    # Also try with spaces, hyphens, underscores
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
                    # Simple word boundary match
                    term_found = bool(re.search(rf'\b{re.escape(term)}\b', normalized_filename, re.IGNORECASE))
                
                if not term_found:
                    all_match = False
                    break
            
            # If all AND terms matched in this OR group, return True
            if all_match:
                return True
        
        return False
    
    except Exception as e:
        logging.error(f"Error matching pattern for '{normalized_filename}': {e}")
        return False

def search_files_with_regex(folder_id: str, pattern: Union[str, List[str]], drive_service=None, max_depth: int = 10) -> List[str]:
    """
    Enhanced recursive search with better error handling and depth limiting.
    
    Args:
        folder_id: Google Drive folder ID to search in
        pattern: Search pattern (string or list of strings for backward compatibility)
        drive_service: Google Drive service instance
        max_depth: Maximum recursion depth to prevent infinite loops
    
    Returns:
        List of Google Drive file links or ["Not Found"]
    """
    if drive_service is None:
        try:
            drive_service = get_drive_service()
        except Exception as e:
            logging.error(f"Failed to get Drive service: {e}")
            return [f"Error: {str(e)}"]
    
    matched_links = []
    visited_folders = set()  # Prevent circular references
    
    # Convert old list format to new pattern format for backward compatibility
    if isinstance(pattern, list):
        pattern = ','.join(pattern)
    
    # Parse the pattern
    pattern_groups = parse_search_pattern(pattern)
    
    def recurse(fid: str, depth: int = 0):
        """Recursive helper with depth limiting and error handling"""
        nonlocal matched_links, visited_folders
        
        # Prevent infinite recursion
        if depth > max_depth:
            logging.warning(f"Max recursion depth {max_depth} reached for folder {fid}")
            return
        
        # Prevent circular references
        if fid in visited_folders:
            return
        visited_folders.add(fid)
        
        try:
            # Use the safe_drive_list with retries
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
                    
                    # Recurse into folders
                    if mime_type == "application/vnd.google-apps.folder":
                        recurse(item_id, depth + 1)
                    else:
                        # Check if file matches pattern
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
                
            # Validate state structure
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
            
            # Try backup if exists
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
            # Write to temp file first
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state_obj, f, indent=2, ensure_ascii=False)
            
            # Backup existing state if it exists
            if os.path.exists(path):
                try:
                    shutil.copy2(path, backup_path)
                except Exception as backup_err:
                    logging.warning(f"Failed to create backup: {backup_err}")
            
            # Atomic replace
            if os.name == 'nt':  # Windows
                if os.path.exists(path):
                    os.remove(path)
            os.rename(temp_path, path)
            
            processed = len(state_obj.get('processed_indices', []))
            total = state_obj.get('total_rows', 0)
            logging.info(f"💾 State saved: {processed}/{total} rows ({state_obj.get('progress_percent', 0):.1f}%)")
            
    except Exception as e:
        logging.exception("Failed to save state atomically")
        # Clean up temp file if exists
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
        
    # Calculate progress percentage
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
    """
    Remove state file and partial file
    
    Args:
        state_id: The state identifier
        reason: Why cleanup is happening (completed, manual, error)
    """
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
    """
    Clean up state files older than max_age_hours
    Useful for cleaning up abandoned/crashed processes
    """
    if not os.path.exists(STATE_DIR):
        return
    
    now = datetime.now()
    cleaned = 0
    
    for filename in os.listdir(STATE_DIR):
        if not filename.startswith("state_") or not filename.endswith(".json"):
            continue
        
        filepath = os.path.join(STATE_DIR, filename)
        
        try:
            # Check file age
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            age_hours = (now - mtime).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                # Extract state_id from filename
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
        # Store the pattern as-is (supports new OR/AND syntax)
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
        # Store pattern as-is
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
    # Return pattern as-is (now supports OR/AND syntax)
    doc_list = [{"documentName": k, "defaultPattern": v if isinstance(v, str) else ", ".join(v)} for k, v in docs.items()]
    return jsonify({"documents": doc_list})

@app1.route("/run", methods=["POST"])
def run_search_enhanced():
    """
    Enhanced run with:
    - Automatic resume from crashes
    - Automatic cleanup when complete
    - Better error handling
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
    
    # Clean up old abandoned states (optional - run once per day)
    # cleanup_old_states(max_age_hours=24)
    
    # Generate deterministic state ID
    state_id = make_state_id(folder_link, excel_path, borrower_col, doc_patterns)
    partial_output = partial_output_for_id(state_id)
    
    logging.info(f"🚀 Starting process with state_id: {state_id}")
    
    # ============================================
    # LOAD INPUT DATA
    # ============================================
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
    
    # Normalize borrower column name
    if borrower_col not in df.columns and borrower_col != "":
        for c in df.columns:
            if str(c).strip().lower() == str(borrower_col).strip().lower():
                borrower_col = c
                break
    
    # ============================================
    # LOAD OR INITIALIZE STATE
    # ============================================
    processed_indices = set()
    resume_mode = False
    
    state = load_state_safe(state_id)
    if state:
        processed_indices = set(state.get("processed_indices", []))
        resume_mode = True
        
        # Check if already complete
        if is_processing_complete(state_id, state):
            logging.info("⚠️ Process already completed, cleaning up old state")
            cleanup_state(state_id, reason="already_completed")
            processed_indices = set()  # Start fresh
            resume_mode = False
        else:
            logging.info(f"🔄 RESUMING: {len(processed_indices)}/{len(df)} rows already processed")
        
        # Try to load partial results
        if resume_mode and os.path.exists(partial_output):
            try:
                df_partial = pd.read_excel(partial_output)
                if len(df_partial) == len(df):
                    df = df_partial
                    logging.info("✅ Loaded partial results successfully")
            except Exception as e:
                logging.warning(f"Could not load partial results: {e}")
    
    # ============================================
    # INITIALIZE OUTPUT COLUMNS
    # ============================================
    parent_id = extract_folder_id(folder_link)
    
    for doc in doc_patterns.keys():
        if doc not in df.columns:
            df[doc] = ""
    if "Folder Link" not in df.columns:
        df["Folder Link"] = ""
    
    # Save initial state
    initial_state = {
        "processed_indices": list(processed_indices),
        "partial_output": partial_output,
        "total_rows": len(df),
        "started_at": datetime.now().isoformat(),
        "resume_mode": resume_mode
    }
    save_state_enhanced(state_id, initial_state)
    
    # ============================================
    # PROCESS ROWS
    # ============================================
    errors = []
    checkpoint_frequency = 5  # Save every 5 rows (tune this for performance)
    
    try:
        for idx, row in df.iterrows():
            # Skip already processed rows
            if int(idx) in processed_indices:
                continue
            
            try:
                borrower = str(row.get(borrower_col, "")).strip().lower() if borrower_col else ""
                
                # Find borrower folder
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
                    
                    # Checkpoint
                    if int(idx) % checkpoint_frequency == 0:
                        save_checkpoint(state_id, processed_indices, partial_output, df, len(df), int(idx))
                    
                    continue
                
                # Search for documents
                for doc, pattern in doc_patterns.items():
                    if not pattern:
                        df.at[idx, doc] = "Skipped"
                        continue
                    
                    try:
                        links = search_files_with_regex(folder_id, pattern)
                        df.at[idx, doc] = "; ".join(links)
                    except Exception as search_err:
                        logging.error(f"Search error for {doc}: {search_err}")
                        df.at[idx, doc] = f"Error: {str(search_err)}"
                        errors.append(f"Row {idx}, Doc '{doc}': {str(search_err)}")
                
                # Mark as processed
                processed_indices.add(int(idx))
                
                # Checkpoint (save every N rows or at the end)
                if int(idx) % checkpoint_frequency == 0 or int(idx) == len(df) - 1:
                    save_checkpoint(state_id, processed_indices, partial_output, df, len(df), int(idx))
            
            except Exception as row_err:
                logging.exception(f"Error processing row {idx}")
                errors.append(f"Row {idx}: {str(row_err)}")
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
    
    # ============================================
    # CHECK COMPLETION & AUTO-CLEANUP
    # ============================================
    final_state = {
        "processed_indices": list(processed_indices),
        "partial_output": partial_output,
        "total_rows": len(df)
    }
    
    if is_processing_complete(state_id, final_state):
        logging.info("✅ Processing 100% complete - cleaning up state files")
        # We'll cleanup after generating final report
    
    # ============================================
    # GENERATE FINAL REPORT
    # ============================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"Report_{timestamp}.xlsx"
    
    try:
        # Reorder columns
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
    
    # ============================================
    # AUTO-CLEANUP STATE (Complete!)
    # ============================================
    cleanup_state(state_id, reason="completed")
    
    # ============================================
    # SEND EMAIL
    # ============================================
    if email_to:
        try:
            msg = EmailMessage()
            msg['From'] = SENDER_EMAIL
            msg['To'] = email_to if isinstance(email_to, str) else ", ".join(email_to)
            if email_cc:
                msg['Cc'] = email_cc if isinstance(email_cc, str) else ", ".join(email_cc)
            msg['Subject'] = email_subject
            
            # Add error summary if any
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
    
    # ============================================
    # RETURN RESPONSE
    # ============================================
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
# Add before if __name__ == "__main__":
def init_state_cleanup():
    """Run once at startup to clean old states"""
    try:
        cleanup_old_states(max_age_hours=24)
    except Exception as e:
        logging.error(f"Startup cleanup failed: {e}")

# Call at startup
init_state_cleanup()

if __name__ == "__main__":
    app1.run(debug=True, threaded=True)
