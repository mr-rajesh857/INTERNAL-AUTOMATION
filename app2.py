import os
import re
import time
import platform
import requests
import sys
from io import BytesIO
from PIL import Image, ImageOps
import pandas as pd
import pytesseract
import pdf2image
from PyPDF2 import PdfReader, PdfMerger
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Blueprint, request, jsonify, render_template, current_app
from werkzeug.utils import secure_filename
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import threading
import mimetypes
import ssl
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from playwright.sync_api import sync_playwright
import json
# Google Drive API libs (optional)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except Exception:
    service_account = None
    build = None
    MediaFileUpload = None

# Optional libs
try:
    import camelot
except Exception:
    camelot = None
try:
    import tabula
except Exception:
    tabula = None
try:
    import easyocr
except Exception:
    easyocr = None

# Tesseract path
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
# SCOPES = ['https://www.googleapis.com/auth/drive.file']
# Defaults (will prefer current_app.config values at runtime)
DEFAULT_UPLOAD_FOLDER = "uploads"
DEFAULT_MERGED_FOLDER = "merged_pdfs"
DEFAULT_DRIVE_FOLDER_ID = "1f0Es9z4ZiZapZ8NldcC3mbuGOxukLUrk"  # your provided id as fallback
DEFAULT_SERVICE_ACCOUNT_BASENAME = "service_account.json"

# Ensure local folders exist (these are safe defaults)
os.makedirs(DEFAULT_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEFAULT_MERGED_FOLDER, exist_ok=True)

# Threading safety + caches
FILE_CACHE = {}
FILE_CACHE_LOCK = threading.Lock()

# Session for downloads
session = requests.Session()

# POPPLER (optional) path (Windows). Set env var POPPLER_PATH if needed.
POPPLER_PATH = os.environ.get("POPPLER_PATH")
# Thread counts (may be overridden by config)
DOWNLOAD_THREADS = 4
VERIFY_THREADS = 4
OCR_PAGE_THREADS = 4

# OCR settings
USE_EASYOCR = False
EASYOCR_MODEL_LANGS = ["en"]
EASYOCR_GPU_IF_AVAILABLE = True
DEFAULT_DPI = 180
EARLY_EXIT_KEYWORDS = ["loan", "loan id", "loan_id", "loanid", "mobile", "phone", "aadhaar", "pan", "id", "name"]
PLAYWRIGHT_LOCK = threading.Lock()

# ----------- State Management Functions -----------
def get_state_file_path(excel_path):
    """Generate hidden state file path based on excel filename"""
    base_dir = os.path.dirname(excel_path) or "."
    excel_filename = os.path.basename(excel_path)
    state_filename = f".state_{excel_filename}.json"
    return os.path.join(base_dir, state_filename)

def load_state(excel_path):
    """Load processing state from file if it exists"""
    state_path = get_state_file_path(excel_path)
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
            print(f"📂 Loaded state from {state_path}")
            return state
    except Exception as e:
        print(f"Failed to load state file: {e}")
        return None

def save_state(excel_path, last_index, ver_log_rows, merge_log_rows, print_log, summary_stats):
    """Save current processing state to file"""
    state_path = get_state_file_path(excel_path)
    state_data = {
        'last_processed_index': last_index,
        'verification_log_rows': ver_log_rows,
        'merge_log_rows': merge_log_rows,
        'textual_print_log': print_log,
        'summary_stats': summary_stats,
        'timestamp': datetime.now().isoformat()
    }
    try:
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save state file: {e}")

def delete_state(excel_path):
    """Delete state file after successful completion"""
    state_path = get_state_file_path(excel_path)
    if os.path.exists(state_path):
        try:
            os.remove(state_path)
            print(f"🗑️ Deleted state file: {state_path}")
        except Exception as e:
            print(f"Failed to delete state file: {e}")

def check_has_state(excel_path):
    """Check if a state file exists for the given excel file"""
    state_path = get_state_file_path(excel_path)
    return os.path.exists(state_path)

# ----------- Helpers -----------
def download_pdf_via_playwright(url, timeout=30000):
    try:
        with PLAYWRIGHT_LOCK:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(accept_downloads=True)
                page = context.new_page()

                pdf_bytes = None

                def handle_response(response):
                    nonlocal pdf_bytes
                    try:
                        ct = response.headers.get("content-type", "")
                        if "application/pdf" in ct.lower():
                            pdf_bytes = response.body()
                    except Exception:
                        pass

                page.on("response", handle_response)

                page.goto(url, timeout=timeout, wait_until="networkidle")
                page.wait_for_timeout(5000)

                browser.close()
                return pdf_bytes

    except Exception as e:
        print("Playwright PDF download failed:", e)
        return None

def extract_drive_id(link):
    if not isinstance(link, str):
        return None
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", link)
    if m:
        return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", link)
    return m.group(1) if m else None

def download_drive_file(file_id, session=None, timeout=30):
    try:
        s = session or requests.Session()
        base = "https://drive.google.com/uc?export=download"
        resp = s.get(base, params={"id": file_id}, stream=True, timeout=timeout)
        ct = resp.headers.get("content-type", "").lower()
        if "html" in ct:
            for k, v in resp.cookies.items():
                if k.startswith("download_warning"):
                    resp = s.get(base, params={"id": file_id, "confirm": v}, stream=True, timeout=timeout)
                    break
        if resp.status_code == 200:
            return resp.content
        else:
            print(f"Drive download returned status {resp.status_code} for id {file_id}")
            return None
    except Exception as e:
        print("Drive download error:", e)
        return None
def is_html_bytes(b):
    if not b:
        return False
    head = b[:200].lower()
    return b"<html" in head or b"<!doctype html" in head
# ----------------------------
# Generic (non-Drive) downloader
# ----------------------------
def download_generic_file(url, session=None, timeout=30):
    """
    Download PDFs/images from any public clickable URL
    (e.g., https://s.legodesk.com/xxxx)
    """
    try:
        s = session or requests.Session()
        resp = s.get(url, stream=True, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200:
            return resp.content
        else:
            print(f"Generic download failed [{resp.status_code}]: {url}")
            return None
    except Exception as e:
        print("Generic download error:", e)
        return None


# ----------------------------
# Unified downloader (Drive + Any URL)
# ----------------------------
def cached_download_drive_file(file_id, session=None, timeout=30):
    if not file_id:
        return None
    with FILE_CACHE_LOCK:
        if file_id in FILE_CACHE:
            return FILE_CACHE[file_id]
    data = download_drive_file(file_id, session=session, timeout=timeout)
    if data:
        with FILE_CACHE_LOCK:
            FILE_CACHE[file_id] = data
    return data
def download_file_from_link(link):
    if not link or not isinstance(link, str):
        return None

    # 1️⃣ Google Drive
    file_id = extract_drive_id(link)
    if file_id:
        return cached_download_drive_file(file_id, session=session)

    # 2️⃣ Try normal HTTP download
    raw = download_generic_file(link, session=session)
    if raw:
        # PDF or Image → OK
        if is_pdf_bytes(raw):
            return raw

        # HTML detected → JS-protected
        head = raw[:200].lower()
        if b"<html" not in head:
            return raw

    # 3️⃣ Playwright fallback (Legodesk etc.)
    print("⚙ Using Playwright for:", link)
    return download_pdf_via_playwright(link)


def is_pdf_bytes(b: bytes):
    return isinstance(b, (bytes, bytearray)) and b[:4] == b"%PDF"

def normalize_for_match(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).lower().strip())

def verify_text_simple(row, text):
    name = normalize_for_match(row.get("Name", ""))
    loan_id = normalize_for_match(row.get("Loan ID", ""))
    phone = normalize_for_match(row.get("Phone", ""))
    t = normalize_for_match(text or "")
    if name and name in t:
        return "Name Matched"
    elif loan_id and loan_id in t:
        return "Loan ID Matched"
    elif phone and phone in t:
        return "Phone Matched"
    else:
        return "No Match"

# ----------------------------
# FAST PDF TEXT (PDFMiner first, PyPDF2 fallback)
# ----------------------------
def fast_pdf_text(file_bytes):
    # 1) Try PDFMiner (best for structured/table PDFs)
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(BytesIO(file_bytes))
        if text and text.strip():
            return text.lower()
    except Exception:
        pass

    # 2) PyPDF2 fallback (strict=False helps many malformed PDFs)
    try:
        file_like = BytesIO(file_bytes)
        file_like.seek(0)
        reader = PdfReader(file_like, strict=False)
        parts = []
        for p in reader.pages:
            try:
                t = p.extract_text()
                if t and t.strip():
                    parts.append(t)
            except Exception:
                continue
        if parts:
            return "\n".join(parts).lower()
    except Exception:
        pass

    return ""

# ----------------------------
# Image cropping + heuristics
# ----------------------------
def autocrop_image(img, border_pct=0.02):
    try:
        gray = img.convert("L")
        cont = ImageOps.autocontrast(gray)
        bw = cont.point(lambda x: 0 if x < 200 else 255, "1")
        bbox = bw.getbbox()
        if not bbox:
            return img
        w, h = img.size
        left = max(0, bbox[0] - int(w * border_pct))
        upper = max(0, bbox[1] - int(h * border_pct))
        right = min(w, bbox[2] + int(w * border_pct))
        lower = min(h, bbox[3] + int(h * border_pct))
        return img.crop((left, upper, right, lower))
    except Exception:
        return img

def looks_like_table_image(img):
    try:
        gray = img.convert("L")
        w, h = gray.size
        small = gray.resize((max(1, w//4), max(1, h//4)))
        pixels = small.load()
        vertical_lines = 0
        horizontal_lines = 0
        for y in range(0, small.size[1], max(1, small.size[1]//20)):
            row_black = sum(1 for x in range(small.size[0]) if pixels[x, y] < 60)
            if row_black > small.size[0] * 0.4:
                horizontal_lines += 1
        for x in range(0, small.size[0], max(1, small.size[0]//20)):
            col_black = sum(1 for y in range(small.size[1]) if pixels[x, y] < 60)
            if col_black > small.size[1] * 0.4:
                vertical_lines += 1
        return (horizontal_lines >= 1 and vertical_lines >= 1)
    except Exception:
        return False

# ----------------------------
# Tesseract OCR with confidences
# ----------------------------
def tesseract_ocr_with_confidence(img, psm=6, oem=1):
    try:
        config = f"--oem {oem} --psm {psm}"
        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
        texts = []
        confs = []
        n = len(data.get("text", []))
        for i in range(n):
            txt = data["text"][i]
            conf = data["conf"][i]
            if txt and str(txt).strip():
                texts.append(txt)
                try:
                    c = float(conf)
                except Exception:
                    c = -1.0
                confs.append(c)
        avg_conf = float(sum([c for c in confs if c >= 0]) / len([c for c in confs if c >= 0])) if any(c >= 0 for c in confs) else -1.0
        return (" ".join(texts), avg_conf, confs)
    except Exception:
        try:
            txt = pytesseract.image_to_string(img, config=f"--oem {oem} --psm {psm}")
            return (txt, -1.0, [])
        except Exception:
            return ("", -1.0, [])

# ----------------------------
# EasyOCR wrapper
# ----------------------------
EASYOCR_READER = None
def init_easyocr_if_needed():
    global EASYOCR_READER
    if not USE_EASYOCR:
        return
    if easyocr is None:
        print("EasyOCR not installed or import failed.")
        return
    if EASYOCR_READER is None:
        gpu_flag = False
        try:
            import torch
            gpu_flag = torch.cuda.is_available() and EASYOCR_GPU_IF_AVAILABLE
        except Exception:
            gpu_flag = False
        EASYOCR_READER = easyocr.Reader(EASYOCR_MODEL_LANGS, gpu=gpu_flag)

def easyocr_image_with_confidence(img):
    try:
        init_easyocr_if_needed()
        if EASYOCR_READER is None:
            return ("", -1.0, [])
        import numpy as np
        arr = np.array(img.convert("RGB"))
        results = EASYOCR_READER.readtext(arr)
        texts = []
        confs = []
        for bbox, text, conf in results:
            texts.append(text)
            confs.append(conf)
        avg = float(sum(confs) / len(confs)) if confs else -1.0
        return (" ".join(texts), avg, confs)
    except Exception:
        return ("", -1.0, [])

# ----------------------------
# Smart OCR per page (table-aware)
# ----------------------------
def smart_ocr_page(img, use_easyocr=False):
    try:
        img_cropped = autocrop_image(img)
        w, h = img_cropped.size
        if min(w, h) < 800:
            target = (min(1600, w*2), min(1600, h*2))
            img_cropped = img_cropped.resize(target, Image.LANCZOS)

        is_table = looks_like_table_image(img_cropped)
        if is_table:
            if use_easyocr:
                text, avg_conf, confs = easyocr_image_with_confidence(img_cropped)
                used = "easyocr_table_psm4"
                return (text.lower(), avg_conf, confs, used)
            else:
                text, avg_conf, confs = tesseract_ocr_with_confidence(img_cropped, psm=4, oem=1)
                used = "tesseract_table_psm4"
                return (text.lower(), avg_conf, confs, used)
        else:
            if use_easyocr:
                text, avg_conf, confs = easyocr_image_with_confidence(img_cropped)
                used = "easyocr_psm6"
                return (text.lower(), avg_conf, confs, used)
            else:
                text, avg_conf, confs = tesseract_ocr_with_confidence(img_cropped, psm=6, oem=1)
                used = "tesseract_psm6"
                return (text.lower(), avg_conf, confs, used)
    except Exception:
        return ("", -1.0, [], "error")

# ----------------------------
# OCR PDF pages in parallel
# ----------------------------
def ocr_pdf_pages_parallel(pdf_bytes, dpi=DEFAULT_DPI, use_easyocr=False, max_workers=OCR_PAGE_THREADS, max_pages_scan=None):
    try:
        convert_kwargs = {"dpi": dpi}
        if POPPLER_PATH:
            convert_kwargs["poppler_path"] = POPPLER_PATH
        pages = pdf2image.convert_from_bytes(pdf_bytes, **convert_kwargs)
    except Exception as e:
        print("pdf->images conversion error:", e)
        return ("", -1.0, [], [])
    # optionally limit scanning to first N pages
    if max_pages_scan is not None and isinstance(max_pages_scan, int) and max_pages_scan > 0:
        pages = pages[:max_pages_scan]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(smart_ocr_page, page, use_easyocr): i for i, page in enumerate(pages)}
        ordered_results = [None] * len(pages)
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                text, avg_conf, confs, used = fut.result()
            except Exception:
                text, avg_conf, confs, used = ("", -1.0, [], "error")
            ordered_results[i] = (text, avg_conf, confs, used)
            # don't attempt to cancel other futures; we still break collection early here
            if any(k in (text or "") for k in EARLY_EXIT_KEYWORDS):
                break
    collected_text = []
    collected_confs = []
    collected_used = []
    for res in ordered_results:
        if res is None:
            continue
        text, avg_conf, confs, used = res
        collected_text.append(text)
        collected_confs.append(avg_conf)
        collected_used.append(used)
        if any(k in (text or "") for k in EARLY_EXIT_KEYWORDS):
            break
    full_text = "\n".join(collected_text)
    valid = [c for c in collected_confs if c is not None and c >= 0]
    overall_conf = float(sum(valid) / len(valid)) if valid else -1.0
    return (full_text, overall_conf, collected_confs, collected_used)

def ocr_from_image_bytes(image_bytes, use_easyocr=False):
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        txt, avg_conf, confs, used = smart_ocr_page(img, use_easyocr=use_easyocr)
        return txt, avg_conf, confs, used
    except Exception:
        return "", -1.0, [], "error"

def ocr_from_pdf_bytes(pdf_bytes, dpi=DEFAULT_DPI, use_easyocr=False):
    # limit pages scanned by default to avoid expensive conversion for huge docs (None => full)
    return ocr_pdf_pages_parallel(pdf_bytes, dpi=dpi, use_easyocr=use_easyocr, max_workers=OCR_PAGE_THREADS, max_pages_scan=None)

# ----------------------------
# Table extraction (Camelot / Tabula / PDFMiner fallback)
# ----------------------------
def extract_table_text_from_pdf_with_camelot(pdf_bytes, pages="1-end"):
    if camelot is None:
        return None
    try:
        tmp = "/tmp/_camelot_tmp.pdf"
        with open(tmp, "wb") as f:
            f.write(pdf_bytes)
        try:
            tables = camelot.read_pdf(tmp, pages=pages, flavor='lattice')
            if len(tables) == 0:
                tables = camelot.read_pdf(tmp, pages=pages, flavor='stream')
        except Exception:
            tables = camelot.read_pdf(tmp, pages=pages, flavor='stream')
        if len(tables) == 0:
            return None
        parts = []
        for t in tables:
            df = t.df
            parts.append("\n".join([" | ".join(row) for row in df.values.tolist()]))
        return "\n\n".join(parts)
    except Exception:
        return None

def extract_table_text_from_pdf_with_tabula(pdf_bytes, pages="all"):
    if tabula is None:
        return None
    try:
        tmp = "/tmp/_tabula_tmp.pdf"
        with open(tmp, "wb") as f:
            f.write(pdf_bytes)
        dfs = tabula.read_pdf(tmp, pages=pages, multiple_tables=True)
        if not dfs:
            return None
        parts = []
        for df in dfs:
            parts.append("\n".join([" | ".join(map(str, row)) for row in df.values.tolist()]))
        return "\n\n".join(parts)
    except Exception:
        return None

def try_table_extraction(pdf_bytes):
    if camelot is not None:
        try:
            t = extract_table_text_from_pdf_with_camelot(pdf_bytes)
            if t:
                return t.lower()
        except Exception:
            pass
    if tabula is not None:
        try:
            t = extract_table_text_from_pdf_with_tabula(pdf_bytes)
            if t:
                return t.lower()
        except Exception:
            pass
    try:
        from pdfminer.high_level import extract_text
        txt = extract_text(BytesIO(pdf_bytes))
        if txt and txt.strip():
            lines = txt.splitlines()
            table_lines = [l for l in lines if "|" in l or re.search(r"\s{2,}", l)]
            if table_lines:
                return "\n".join(table_lines).lower()
    except Exception:
        pass
    return None

# ----------------------------
# Image -> PDF converter
# ----------------------------
def image_bytes_to_pdf_bytes(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    out = BytesIO()
    img.save(out, format="PDF")
    return out.getvalue()

# ----------------------------
# High-level extraction wrapper
# ----------------------------
def smart_extract_text_from_bytes(raw_bytes, use_easyocr=False):
    if raw_bytes is None:
        return ("", -1.0, [], "none")
    try:
        if is_pdf_bytes(raw_bytes):
            pdfminer_text = fast_pdf_text(raw_bytes)
            if pdfminer_text and pdfminer_text.strip():
                return (pdfminer_text.lower(), 100.0, [], "pdf_text")
            table_text = try_table_extraction(raw_bytes)
            if table_text:
                return (table_text.lower(), 90.0, [], "table_extraction")
            return ocr_from_pdf_bytes(raw_bytes, dpi=DEFAULT_DPI, use_easyocr=use_easyocr)
        else:
            return ocr_from_image_bytes(raw_bytes, use_easyocr=use_easyocr)
    except Exception:
        return ("", -1.0, [], "error")

# ----------------------------
# Single document verification
# ----------------------------
def process_single_verification(row, doc_col, link, use_easyocr=False):
    loan_id = row.get("Loan ID", "")
    name = row.get("Name", "")
    start_time = time.time()
    method_used = ""
    file_type = ""
    verification_result = "No Match"
    reason = ""
    notes = ""
    try:
        file_id = extract_drive_id(link)
        if not file_id:
            method_used = ""
            verification_result = "Failed"
            reason = "Invalid Drive Link"
            elapsed = time.time() - start_time
            return ({
                "Loan ID": loan_id,
                "Name": name,
                "Document": doc_col,
                "Method Used": method_used,
                "Verification Result": verification_result,
                "Reason": reason,
                "Time Taken (s)": round(elapsed, 2),
                "File Type": file_type,
                "Confidence": -1.0,
                "Notes": notes
            }, False)

        raw = cached_download_drive_file(file_id, session=session)
        if not raw:
            verification_result = "Failed"
            reason = "File Not Accessible"
            elapsed = time.time() - start_time
            return ({
                "Loan ID": loan_id,
                "Name": name,
                "Document": doc_col,
                "Method Used": method_used,
                "Verification Result": verification_result,
                "Reason": reason,
                "Time Taken (s)": round(elapsed, 2),
                "File Type": file_type,
                "Confidence": -1.0,
                "Notes": notes
            }, False)

        if is_pdf_bytes(raw):
            file_type = "PDF"
            pdf_text = fast_pdf_text(raw)
            if pdf_text and pdf_text.strip():
                method_used = "PDF Text Read"
                verification_result = verify_text_simple(row, pdf_text)
                confidence = 100.0
                if verification_result != "No Match":
                    elapsed = time.time() - start_time
                    return ({
                        "Loan ID": loan_id,
                        "Name": name,
                        "Document": doc_col,
                        "Method Used": method_used,
                        "Verification Result": verification_result,
                        "Reason": "",
                        "Time Taken (s)": round(elapsed, 2),
                        "File Type": file_type,
                        "Confidence": confidence,
                        "Notes": ""
                    }, True)
                notes = "Selectable text present but didn't match; trying table extraction/OCR"
            table_text = try_table_extraction(raw)
            if table_text:
                method_used = "Table Extract (Camelot/Tabula/PDFMiner)"
                verification_result = verify_text_simple(row, table_text)
                confidence = 90.0
                if verification_result != "No Match":
                    elapsed = time.time() - start_time
                    return ({
                        "Loan ID": loan_id,
                        "Name": name,
                        "Document": doc_col,
                        "Method Used": method_used,
                        "Verification Result": verification_result,
                        "Reason": "",
                        "Time Taken (s)": round(elapsed, 2),
                        "File Type": file_type,
                        "Confidence": confidence,
                        "Notes": ""
                    }, True)
                else:
                    notes = "Table extraction performed but no match"
            method_used = "OCR (Table-aware pages)"
            ocr_text, avg_conf, confs, used_modes = smart_extract_text_from_bytes(raw, use_easyocr=use_easyocr)
            if ocr_text:
                verification_result = verify_text_simple(row, ocr_text)
                if verification_result != "No Match":
                    elapsed = time.time() - start_time
                    used_str = used_modes if isinstance(used_modes, str) else ", ".join(used_modes) if isinstance(used_modes, (list, tuple)) else str(used_modes)
                    return ({
                        "Loan ID": loan_id,
                        "Name": name,
                        "Document": doc_col,
                        "Method Used": method_used + f" [{used_str}]",
                        "Verification Result": verification_result,
                        "Reason": "",
                        "Time Taken (s)": round(elapsed, 2),
                        "File Type": file_type,
                        "Confidence": avg_conf,
                        "Notes": notes
                    }, True)
                else:
                    reason = "No match after OCR"
                    elapsed = time.time() - start_time
                    used_str = used_modes if isinstance(used_modes, str) else ", ".join(used_modes) if isinstance(used_modes, (list, tuple)) else str(used_modes)
                    return ({
                        "Loan ID": loan_id,
                        "Name": name,
                        "Document": doc_col,
                        "Method Used": method_used + f" [{used_str}]",
                        "Verification Result": "No Match",
                        "Reason": reason,
                        "Time Taken (s)": round(elapsed, 2),
                        "File Type": file_type,
                        "Confidence": avg_conf,
                        "Notes": notes
                    }, False)
            else:
                reason = "OCR produced no text"
                elapsed = time.time() - start_time
                return ({
                    "Loan ID": loan_id,
                    "Name": name,
                    "Document": doc_col,
                    "Method Used": method_used,
                    "Verification Result": "No Text Extracted",
                    "Reason": reason,
                    "Time Taken (s)": round(elapsed, 2),
                    "File Type": file_type,
                    "Confidence": -1.0,
                    "Notes": notes
                }, False)
        else:
            file_type = "Image"
            method_used = "OCR (Image)"
            ocr_text, avg_conf, confs, used = ocr_from_image_bytes(raw, use_easyocr=use_easyocr)
            if ocr_text:
                verification_result = verify_text_simple(row, ocr_text)
                elapsed = time.time() - start_time
                if verification_result != "No Match":
                    return ({
                        "Loan ID": loan_id,
                        "Name": name,
                        "Document": doc_col,
                        "Method Used": method_used + f" [{used}]",
                        "Verification Result": verification_result,
                        "Reason": "",
                        "Time Taken (s)": round(elapsed, 2),
                        "File Type": file_type,
                        "Confidence": avg_conf,
                        "Notes": ""
                    }, True)
                else:
                    return ({
                        "Loan ID": loan_id,
                        "Name": name,
                        "Document": doc_col,
                        "Method Used": method_used + f" [{used}]",
                        "Verification Result": "No Match",
                        "Reason": "No match after OCR",
                        "Time Taken (s)": round(elapsed, 2),
                        "File Type": file_type,
                        "Confidence": avg_conf,
                        "Notes": ""
                    }, False)
            else:
                elapsed = time.time() - start_time
                return ({
                    "Loan ID": loan_id,
                    "Name": name,
                    "Document": doc_col,
                    "Method Used": method_used,
                    "Verification Result": "No Text Extracted",
                    "Reason": "OCR produced no text",
                    "Time Taken (s)": round(elapsed, 2),
                    "File Type": file_type,
                    "Confidence": -1.0,
                    "Notes": ""
                }, False)
    except Exception as e:
        elapsed = time.time() - start_time
        return ({
            "Loan ID": loan_id,
            "Name": name,
            "Document": doc_col,
            "Method Used": method_used,
            "Verification Result": "Failed",
            "Reason": f"Exception: {e}",
            "Time Taken (s)": round(elapsed, 2),
            "File Type": file_type,
            "Confidence": -1.0,
            "Notes": ""
        }, False)
# ----------------------------
# Google Drive helpers
# ----------------------------
def get_drive_service(service_account_file):
    if service_account is None or build is None:
        raise RuntimeError("Google API libraries not installed.")
    if not os.path.exists(service_account_file):
        raise RuntimeError(f"Service account file not found at: {service_account_file}")

    scopes = ["https://www.googleapis.com/auth/drive"]  # ✅ correct
    creds = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=scopes
    )

    service = build(
        "drive",
        "v3",
        credentials=creds,
        cache_discovery=False
    )
    return service

def upload_file_to_drive(local_path, folder_id, service_account_file):
    try:
        service = get_drive_service(service_account_file)
        filename = os.path.basename(local_path)
        media = MediaFileUpload(local_path, resumable=True)

        file_metadata = {
            "name": filename,
            "parents": [folder_id]
        }

        created = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink",
            supportsAllDrives=True   # ✅ REQUIRED
        ).execute()

        file_id = created["id"]

        # Optional public read (Shared Drive safe)
        try:
            service.permissions().create(
                fileId=file_id,
                body={"type": "anyone", "role": "reader"},
                supportsAllDrives=True   # ✅ REQUIRED
            ).execute()
        except Exception as e:
            print("Permission warning:", e)

        return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"

    except Exception as e:
        print("Drive upload error:", e)
        return None

# ----------------------------
# Email sending
# ----------------------------
def send_email_with_attachment(sender_email, sender_password, to_email, cc_email=None, subject=None, body=None, file_paths=None):
    if not to_email:
        print("No recipient provided for email; skipping send.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject or "Log from verifier"
        recipients = [to_email]
        if cc_email:
            cc_list = [c.strip() for c in cc_email.split(",") if c.strip()]
            msg["Cc"] = ", ".join(cc_list)
            recipients += cc_list
        msg.attach(MIMEText(body or "Please find attached log.", "plain"))
        for file_path in file_paths or []:
            if not file_path or not os.path.exists(file_path):
                print(f"Attachment not found (skipping): {file_path}")
                continue
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = "application/octet-stream"
            main_type, sub_type = mime_type.split("/", 1)
            with open(file_path, "rb") as f:
                part = MIMEBase(main_type, sub_type)
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
            msg.attach(part)
        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())
        print(f"📧 Email sent to {to_email}" + (f" (CC: {cc_email})" if cc_email else ""))
        return True
    except Exception as e:
        print("❌ Email sending failed:", e)
        return False


def process_documents(excel_path, ver_docs, merge_seq_docs, use_easyocr=False,
                      ver_to=None, ver_cc=None, ver_subject=None,
                      merge_to=None, merge_cc=None, merge_subject=None):

    global USE_EASYOCR
    USE_EASYOCR = use_easyocr

    # Read runtime config values from current_app where available
    try:
        upload_folder = current_app.config.get("UPLOAD_FOLDER", DEFAULT_UPLOAD_FOLDER)
        merged_folder = current_app.config.get("MERGED_FOLDER", DEFAULT_MERGED_FOLDER)
        drive_folder_id = current_app.config.get("DRIVE_FOLDER_ID", DEFAULT_DRIVE_FOLDER_ID)
        service_account_file = current_app.config.get("SERVICE_ACCOUNT_FILE",
                                                      os.path.join(os.getcwd(), DEFAULT_SERVICE_ACCOUNT_BASENAME))
        # ensure local folders exist
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(merged_folder, exist_ok=True)
    except RuntimeError:
        # In case called outside flask app context, fall back to defaults
        upload_folder = DEFAULT_UPLOAD_FOLDER
        merged_folder = DEFAULT_MERGED_FOLDER
        drive_folder_id = DEFAULT_DRIVE_FOLDER_ID
        service_account_file = os.path.join(os.getcwd(), DEFAULT_SERVICE_ACCOUNT_BASENAME)
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(merged_folder, exist_ok=True)

    existing_state = load_state(excel_path)
    
    if existing_state:
        verification_log_rows = existing_state.get('verification_log_rows', [])
        merge_log_rows = existing_state.get('merge_log_rows', [])
        textual_print_log = existing_state.get('textual_print_log', [])
        start_index = existing_state.get('last_processed_index', -1) + 1
        summary_stats = existing_state.get('summary_stats', {
            'rows_skipped_missing': 0,
            'rows_failed_verification': 0,
            'rows_merged': 0
        })
        rows_skipped_missing = summary_stats.get('rows_skipped_missing', 0)
        rows_failed_verification = summary_stats.get('rows_failed_verification', 0)
        rows_merged = summary_stats.get('rows_merged', 0)
        textual_print_log.append(f"\n{'='*60}")
        textual_print_log.append(f"🔄 RESUMING FROM ROW {start_index + 1}")
        textual_print_log.append(f"{'='*60}\n")
    else:
        verification_log_rows = []
        merge_log_rows = []
        textual_print_log = []
        start_index = 0
        rows_skipped_missing = 0
        rows_failed_verification = 0
        rows_merged = 0

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        return {"error": f"Failed to read excel: {e}"}

    if "Folder Link" not in df.columns:
        return {"error": "No 'Folder Link' column found in the Excel. Rename or provide correct file."}

    folder_index = df.columns.get_loc("Folder Link")
    doc_columns = df.columns[folder_index+1:].tolist()
    if not doc_columns:
        return {"error": "No document columns detected after 'Folder Link'."}

    for d in ver_docs:
        if d not in doc_columns:
            return {"error": f"Verification document '{d}' not found in detected doc columns."}
    for d in merge_seq_docs:
        if d not in doc_columns:
            return {"error": f"Merge document '{d}' not found in detected doc columns."}

    if USE_EASYOCR:
        try:
            # init if you have this function defined in your full code
            init_easyocr_if_needed()
        except Exception:
            pass

    total_rows = len(df)

    for idx, row in df.iterrows():
        # Skip already processed rows
        if idx < start_index:
            continue
            
        included_docs = []
        loan_id = row.get("Loan ID", "")
        name = row.get("Name", "")
        textual_print_log.append(f"\nProcessing row {idx+1}/{total_rows} — Loan ID: {loan_id} — Name: {name}")

        # Gather links
        links = {}
        missing_any = False
        for col in doc_columns:
            val = row.get(col)
            if pd.isna(val) or str(val).strip() == "" or not str(val).strip().lower().startswith("http"):
                missing_any = True
                links[col] = None
            else:
                links[col] = str(val).strip()

        if missing_any:
            textual_print_log.append("  ⚠ Row skipped: not all document links present.")
            rows_skipped_missing += 1
            verification_log_rows.append({
                "Loan ID": loan_id,
                "Name": name,
                "Document": "ALL",
                "Method Used": "",
                "Verification Result": "Skipped – Missing Document(s)",
                "Reason": "One or more document links missing",
                "Time Taken (s)": 0,
                "File Type": "",
                "Confidence": -1.0,
                "Notes": ""
            })
            merge_log_rows.append({
                "Loan ID": loan_id,
                "Name": name,
                "Merged": "NO",
                "Documents Included": "",
                "Output Path": "",
                "Drive Link": "",
                "Reason": "Missing document(s)",
                "Time Taken (s)": 0
            })
            save_state(excel_path, idx, verification_log_rows, merge_log_rows, textual_print_log, {
                'rows_skipped_missing': rows_skipped_missing,
                'rows_failed_verification': rows_failed_verification,
                'rows_merged': rows_merged
            })
            continue

        # Pre-warm downloads
        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as dl_exec:
            futures = []
            for col in doc_columns:
                link = links[col]
                futures.append(dl_exec.submit(download_file_from_link, link))
            for f in as_completed(futures):
                try:
                    _ = f.result()
                except Exception:
                    pass

        # Verification (parallel)
        ver_all_pass = True
        per_row_ver_results = {}

        # 🔒 Sequential verification (Playwright-safe)
        for doc_col in ver_docs:
            link = links.get(doc_col)

            try:
                log_row, passed = process_single_verification(
                    row, doc_col, link, USE_EASYOCR
                )
            except Exception as e:
                log_row = {
                    "Loan ID": loan_id,
                    "Name": name,
                    "Document": doc_col,
                    "Method Used": "",
                    "Verification Result": "Failed",
                    "Reason": f"Exception during verification: {e}",
                    "Time Taken (s)": 0,
                    "File Type": "",
                    "Confidence": -1.0,
                    "Notes": ""
                }
                passed = False

            verification_log_rows.append(log_row)
            per_row_ver_results[doc_col] = passed

            textual_print_log.append(
                f"  • {doc_col} => {log_row.get('Verification Result')} "
                f"({log_row.get('Method Used')}) "
                f"[{log_row.get('File Type')}] conf={log_row.get('Confidence')}"
            )

            if not passed:
                ver_all_pass = False

        if not ver_all_pass:
            rows_failed_verification += 1
            textual_print_log.append("  ❌ Verification failed — skipping merge.")
            merge_log_rows.append({
                "Loan ID": loan_id,
                "Name": name,
                "Merged": "NO",
                "Documents Included": ", ".join(included_docs),
                "Output Path": "",
                "Drive Link": "",
                "Reason": "Verification failed for one or more selected documents",
                "Time Taken (s)": 0
            })
            save_state(excel_path, idx, verification_log_rows, merge_log_rows, textual_print_log, {
                'rows_skipped_missing': rows_skipped_missing,
                'rows_failed_verification': rows_failed_verification,
                'rows_merged': rows_merged
            })
            continue

        # Merge sequence
        textual_print_log.append("  ✅ All selected verification documents passed — merging.")
        merger = PdfMerger()
        merge_start = time.time()
        included_docs = []
        merge_success = True
        for doc_col in merge_seq_docs:
            link = links.get(doc_col)
            raw = download_file_from_link(link)
            if not raw:
                textual_print_log.append(f"   ⚠ Could not download {doc_col}")
                merge_success = False
                break
 
            try:
                if is_pdf_bytes(raw):
                    merger.append(BytesIO(raw))

                else:
                    # assume image
                    img = Image.open(BytesIO(raw)).convert("RGB")
                    out = BytesIO()
                    img.save(out, format="PDF")
                    out.seek(0)
                    merger.append(out)
                included_docs.append(doc_col)
            except Exception as e:
                textual_print_log.append("   Merge append error for " + doc_col + ": " + str(e))
                merge_success = False
                break

        if merge_success:
            try:
                out_name = (str(loan_id).strip() or f"loan_{idx+1}").replace("/", "_")
                merged_path = os.path.join(merged_folder, f"{out_name}.pdf")
                os.makedirs(os.path.dirname(merged_path), exist_ok=True)
                merger.write(merged_path)
                merger.close()
                merge_elapsed = time.time() - merge_start
                rows_merged += 1

                # Upload to Drive (best-effort)
                drive_link = ""
                try:
                    if service_account is not None and build is not None and MediaFileUpload is not None:
                        drive_link = upload_file_to_drive(merged_path, drive_folder_id, service_account_file) or ""
                        if drive_link:
                            textual_print_log.append(f"   🔗 Uploaded to Drive: {drive_link}")
                        else:
                            textual_print_log.append("   ⚠ Upload to Drive returned no link.")
                    else:
                        textual_print_log.append("   ⚠ Google API libs not installed; skipped Drive upload.")
                except Exception as e:
                    textual_print_log.append("   ⚠ Exception during Drive upload: " + str(e))
                    drive_link = ""

                textual_print_log.append(f"   🎉 Merged PDF saved: {merged_path} (included: {included_docs})")
                merge_log_rows.append({
                    "Loan ID": loan_id,
                    "Name": name,
                    "Merged": "YES",
                    "Documents Included": ", ".join(included_docs),
                    "Output Path": merged_path,
                    "Drive Link": drive_link or "",
                    "Reason": "All verification passed",
                    "Time Taken (s)": round(merge_elapsed, 2)
                })
            except Exception as e:
                textual_print_log.append("   ❌ Failed to write merged PDF: " + str(e))
                merge_log_rows.append({
                    "Loan ID": loan_id,
                    "Name": name,
                    "Merged": "NO",
                    "Documents Included": ", ".join(included_docs),
                    "Output Path": "",
                    "Drive Link": "",
                    "Reason": f"Merge write failed: {e}",
                    "Time Taken (s)": 0
                })
        else:
            textual_print_log.append("   ❌ Merge failed due to earlier errors.")
            merge_log_rows.append({
                "Loan ID": loan_id,
                "Name": name,
                "Merged": "NO",
                "Documents Included": ", ".join(included_docs) if included_docs else "",
                "Output Path": "",
                "Drive Link": "",
                "Reason": "Merge append/download error",
                "Time Taken (s)": 0
            })
        
        save_state(excel_path, idx, verification_log_rows, merge_log_rows, textual_print_log, {
            'rows_skipped_missing': rows_skipped_missing,
            'rows_failed_verification': rows_failed_verification,
            'rows_merged': rows_merged
        })

    delete_state(excel_path)
    textual_print_log.append(f"\n{'='*60}")
    textual_print_log.append("✅ ALL ROWS PROCESSED SUCCESSFULLY")
    textual_print_log.append(f"{'='*60}\n")

    # Save logs
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_dir = os.path.dirname(excel_path) or "."

    verification_log_path = os.path.join(base_dir, f"verification_log_{timestamp}.xlsx")
    merge_log_path = os.path.join(base_dir, f"merge_log_{timestamp}.xlsx")

    try:
        pd.DataFrame(verification_log_rows).to_excel(verification_log_path, index=False)
    except Exception as e:
        print("Failed to save verification log:", e)
        verification_log_path = None

    try:
        pd.DataFrame(merge_log_rows).to_excel(merge_log_path, index=False)
    except Exception as e:
        print("Failed to save merge log:", e)
        merge_log_path = None

    summary = {
        "Total Rows": total_rows,
        "Rows Skipped (Missing)": rows_skipped_missing,
        "Rows Failed Verification": rows_failed_verification,
        "Rows Successfully Merged": rows_merged,
        "Verification Log": verification_log_path,
        "Merge Log": merge_log_path,
        "Merged PDFs location": merged_folder
    }

    return {
        "verification_log_rows": verification_log_rows,
        "merge_log_rows": merge_log_rows,
        "print_log": "\n".join(textual_print_log),
        "summary": summary,
        "verification_log_path": verification_log_path,
        "merge_log_path": merge_log_path
    }

# ----------------------------
# Flask blueprint and routes
# ----------------------------
app2 = Blueprint("app2", __name__, template_folder="templates", static_folder="static")

@app2.route("/tool")
def index():
    return render_template("index2.html")

@app2.route("/upload", methods=["POST"])
def upload():
    if "excel" not in request.files:
        return jsonify({"error": "No file part 'excel'"}), 400
    f = request.files["excel"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    filename = secure_filename(f.filename)
    upload_folder = current_app.config.get("UPLOAD_FOLDER", DEFAULT_UPLOAD_FOLDER)
    save_path = os.path.join(upload_folder, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    f.save(save_path)
    try:
        df = pd.read_excel(save_path)
        cols = list(df.columns)
        if "Folder Link" not in cols:
            return jsonify({"error": "No 'Folder Link' column found in Excel."}), 400
        idx = cols.index("Folder Link") + 1
        doc_columns = cols[idx:]
        has_resume_state = check_has_state(save_path)
        return jsonify({
            "excel_path": save_path, 
            "document_columns": doc_columns,
            "has_resume_state": has_resume_state
        })
    except Exception as e:
        return jsonify({"error": f"Failed to read excel: {e}"}), 500

@app2.route("/process", methods=["POST"])
def process_route():
    data = request.get_json()
    excel_path = data.get("excel_path")
    verify_docs = data.get("verify_docs")  # expected list
    merge_sequence = data.get("merge_sequence")  # expected list
    use_easyocr = data.get("use_easyocr", False)

    # Email params may be provided by frontend (frontend will call /send-email to send actual mails)
    ver_to = data.get("ver_to")
    ver_cc = data.get("ver_cc")
    ver_subject = data.get("ver_subject")

    merge_to = data.get("merge_to")
    merge_cc = data.get("merge_cc")
    merge_subject = data.get("merge_subject")

    if not excel_path or not verify_docs or not merge_sequence:
        return jsonify({"error": "Missing parameters. excel_path, verify_docs, merge_sequence are required."}), 400

    # Normalize + safety check: only allow files saved in UPLOAD_FOLDER
    safe_base = os.path.abspath(current_app.config.get("UPLOAD_FOLDER", DEFAULT_UPLOAD_FOLDER))
    excel_abs = os.path.abspath(excel_path)
    if not excel_abs.startswith(safe_base + os.sep):
        return jsonify({"error": "Invalid excel_path. Not allowed."}), 400

    if not os.path.exists(excel_abs):
        return jsonify({"error": "Excel file not found on server."}), 400

    result = process_documents(
        excel_abs,
        ver_docs=verify_docs,
        merge_seq_docs=merge_sequence,
        use_easyocr=use_easyocr,
        ver_to=ver_to,
        ver_cc=ver_cc,
        ver_subject=ver_subject,
        merge_to=merge_to,
        merge_cc=merge_cc,
        merge_subject=merge_subject
    )
    return jsonify(result)

@app2.route("/send-email", methods=["POST"])
def send_email_route():
    data = request.json
    to_email = data.get("to")
    cc_email = data.get("cc")
    subject = data.get("subject")
    body = data.get("body")
    attachments = data.get("attachments", [])

    ok = send_email_with_attachment(
        current_app.config.get("SENDER_EMAIL"),
        current_app.config.get("SENDER_PASSWORD"),
        to_email, cc_email, subject, body, attachments
    )
    return jsonify({"success": ok})
