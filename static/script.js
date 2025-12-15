// NOTE: Do NOT re-declare GENERATED_REPORT_FILENAME here — it's declared in index.html
// Assume GENERATED_REPORT_FILENAME exists as a global variable.

let excelPath = "";
let columns = [];
let currentCaseType = "";
let editingDocOldName = ""; // for edit flow
let scanEventSource = null;

// toast helper
function showToast(msg, type = "info") {
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.style.display = "block";
    toast.style.background = type === "error" ? "#c0392b" : "#27ae60";
    toast.textContent = msg;
    setTimeout(() => (toast.style.display = "none"), 3000);
}

// Make uploadSheetOrExcel global (function exists in your original code)
window.uploadSheetOrExcel = async function () {
    const file = document.getElementById("excel_file").files[0];
    const sheetLink = document.getElementById("gsheet_link").value.trim();

    if (file) {
        const fd = new FormData();
        fd.append("file", file);
        const res = await fetch("/upload_sheet_or_excel", { method: "POST", body: fd });
        const json = await res.json();
        if (res.ok) {
            excelPath = json.file_path;
            columns = json.columns || [];
            populateColumnSelect(columns);
            document.getElementById("section2").style.display = "block";
            showToast("Excel uploaded successfully!");
        } else {
            showToast(json.error || "Upload failed", "error");
        }
        return;
    }

    if (sheetLink) {
        const res = await fetch("/upload_sheet_or_excel", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sheet_link: sheetLink })
        });
        const json = await res.json();
        if (res.ok) {
            excelPath = json.file_path;
            columns = json.columns || [];
            populateColumnSelect(columns);
            document.getElementById("section2").style.display = "block";
            showToast("Google Sheet loaded successfully!");
        } else {
            showToast(json.error || "Failed to load Google Sheet", "error");
        }
        return;
    }

    showToast("Please upload an .xlsx file or provide a Google Sheet link", "error");
};

function populateColumnSelect(cols) {
    const colSel = document.getElementById("borrower_col");
    if (!colSel) return;
    colSel.innerHTML = cols.map(c => `<option value="${c}">${c}</option>`).join("");
}

// generateTree (folder structure)
function generateTree(struct) {
    if (!struct || !struct.length) return "<ul><li>No files found</li></ul>";
    let html = "<ul class='tree'>";
    struct.forEach(item => {
        const isFolder = item.type === "folder";
        const icon = isFolder ? "📁" : "📄";
        const toggle = isFolder ? `<span class='toggle'>▶</span>` : "";
        const copyIcon = isFolder
            ? `<span class='copy' title='Copy folder name' onclick='copyText("${escapeHtml(item.name)}")'>📋</span>`
            : `<span class='copy' title='Copy file link' onclick='copyText("https://drive.google.com/file/d/${item.id}/view")'>📋</span>`;

        html += `<li>${toggle} ${icon} <span class='item-name'>${escapeHtml(item.name)}</span> ${copyIcon}`;

        if (isFolder && item.children && item.children.length) {
            html += `<div class='nested' style='display:none;'>${generateTree(item.children)}</div>`;
        }
        html += `</li>`;
    });
    html += "</ul>";
    return html;
}

document.addEventListener("click", e => {
    if (e.target.classList.contains("toggle")) {
        const nested = e.target.parentElement.querySelector(".nested");
        if (nested) {
            const expanded = nested.style.display === "block";
            nested.style.display = expanded ? "none" : "block";
            e.target.textContent = expanded ? "▶" : "▼";
        }
    }
});

function escapeHtml(unsafe) {
    return String(unsafe)
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function copyText(text) {
    navigator.clipboard.writeText(text);
    showToast(`Copied: ${text}`, "info");
}

// when case type changes, populate doc table & management area
document.getElementById("case_type")?.addEventListener("change", async e => {
    const caseType = e.target.value;
    currentCaseType = caseType;
    if (!caseType) {
        document.getElementById("doc_table").style.display = "none";
        document.getElementById("case_docs_area").style.display = "none";
        return;
    }

    // fetch documents for both table and doc management area
    const res = await fetch("/get_documents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ caseType })
    });
    const json = await res.json();
    const tbody = document.querySelector("#doc_table tbody");
    tbody.innerHTML = "";
    json.documents.forEach(d => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><input type="checkbox" class="doc_chk" data-doc="${escapeHtml(d.documentName)}" checked></td>
            <td>${escapeHtml(d.documentName)}</td>
            <td><input type="text" class="doc_pattern" value="${escapeHtml(d.defaultPattern)}"></td>
        `;
        tbody.appendChild(tr);
    });
    document.getElementById("doc_table").style.display = "table";

    // populate management area (readable list with edit/delete)
    populateDocsManagementArea(caseType, json.documents);
    document.getElementById("section4").style.display = "block";
});

function populateDocsManagementArea(caseType, docsArray) {
    const area = document.getElementById("case_docs_area");
    if (!area) return;
    area.style.display = "block";
    const div = document.getElementById("docs_list");
    div.innerHTML = "";
    if (!docsArray.length) {
        div.innerHTML = "<div>No documents defined for this case type.</div>";
        return;
    }
    docsArray.forEach(d => {
        const docName = d.documentName;
        const pattern = d.defaultPattern;
        const wrapper = document.createElement("div");
        wrapper.style.display = "flex";
        wrapper.style.justifyContent = "space-between";
        wrapper.style.alignItems = "center";
        wrapper.style.marginBottom = "8px";
        wrapper.innerHTML = `
            <div><strong>${escapeHtml(docName)}</strong><div style='font-size:0.9rem;color:#bdbdbd;'>${escapeHtml(pattern)}</div></div>
            <div>
                <button onclick='openEditDocModal("${escapeHtml(docName)}","${escapeHtml(pattern)}")'>Edit</button>
            </div>
        `;
        div.appendChild(wrapper);
    });
}

// START FOLDER SCAN (SSE)
function startFolderScan() {
    const folderLink = document.getElementById("folder_link").value.trim();
    const borrowerCol = document.getElementById("borrower_col").value;
    if (!folderLink) return showToast("Enter parent folder link first", "error");
    if (!borrowerCol) return showToast("Select borrower column", "error");
    if (!excelPath) return showToast("Upload an Excel or supply a Google Sheet first", "error");

    document.getElementById("scan_results").innerHTML = "";
    document.getElementById("folder_structure").innerHTML = "";

    if (scanEventSource) {
        scanEventSource.close();
        scanEventSource = null;
    }

    const q = new URLSearchParams({
        folder_link: folderLink,
        borrower_col: borrowerCol,
        excel_path: encodeURIComponent(excelPath)
    });

    const url = `/stream_scan?${q.toString()}`;
    scanEventSource = new EventSource(url);

    scanEventSource.onmessage = function (ev) {
        try {
            const data = JSON.parse(ev.data);
            if (data.error) {
                showToast(data.error, "error");
                appendScanRow({ borrower: "Error", status: "error", message: data.error });
                return;
            }
            if (data.done) {
                showToast("Folder scan completed! Please select a Case Type and click Run.");
                scanEventSource.close();
                scanEventSource = null;
                document.getElementById("section3").style.display = "block";
                return;
            }
            appendScanRow(data);
        } catch (err) {
            console.error("Invalid SSE data", err, ev.data);
        }
    };

    scanEventSource.onerror = function (err) {
        console.error("SSE error", err);
        showToast("Scan connection interrupted", "error");
        if (scanEventSource) {
            scanEventSource.close();
            scanEventSource = null;
        }
    };

    showToast("Started live folder scan. Results will appear below...");
}

function appendScanRow(payload) {
    const container = document.getElementById("scan_results");
    const row = document.createElement("div");
    row.className = "scan-row";

    const borrowerDiv = document.createElement("div");
    borrowerDiv.className = "borrower";
    borrowerDiv.textContent = payload.borrower || "(blank)";

    const statusDiv = document.createElement("div");
    statusDiv.className = "status " + (payload.status === "found" ? "found" : (payload.status === "not_found" ? "not_found" : ""));
    statusDiv.textContent = payload.status === "found" ? "Found" : (payload.status === "not_found" ? "Not Found" : payload.status);

    row.appendChild(borrowerDiv);
    row.appendChild(statusDiv);

    if (payload.status === "found" && payload.folder_id) {
        const linkDiv = document.createElement("div");
        linkDiv.style.flex = "1";
        linkDiv.innerHTML = `<div style="margin-bottom:6px;"><a href="https://drive.google.com/drive/folders/${payload.folder_id}" target="_blank">Open Folder</a> &nbsp; <span class='copy' style='cursor:pointer;' onclick='copyText("https://drive.google.com/drive/folders/${payload.folder_id}")'>📋</span></div>`;
        const structHtml = generateTree(payload.structure || []);
        const wrapper = document.createElement("div");
        wrapper.innerHTML = structHtml;
        linkDiv.appendChild(wrapper);
        row.appendChild(linkDiv);
    } else {
        const spacer = document.createElement("div");
        spacer.style.flex = "1";
        row.appendChild(spacer);
    }

    container.appendChild(row);
}

// prepareDocsThenRun (from previous)
async function prepareDocsThenRun() {
    document.getElementById("section3").style.display = "block";
    const caseTypeSelect = document.getElementById("case_type");
    const caseType = caseTypeSelect.value;

    if (!caseType) {
        showToast("Please select a Case Type before running", "error");
        return;
    }

    const tbody = document.querySelector("#doc_table tbody");
    if (!tbody.children.length) {
        const res = await fetch("/get_documents", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ caseType })
        });
        const json = await res.json();
        tbody.innerHTML = "";
        json.documents.forEach(d => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><input type="checkbox" class="doc_chk" data-doc="${escapeHtml(d.documentName)}" checked></td>
                <td>${escapeHtml(d.documentName)}</td>
                <td><input type="text" class="doc_pattern" value="${escapeHtml(d.defaultPattern)}"></td>
            `;
            tbody.appendChild(tr);
        });
        document.getElementById("section4").style.display = "block";
    }

    // Open email modal BEFORE starting heavy processing
    openEmailModalBeforeRun();
}

// ---------------------------
// EMAIL-FIRST RUN FLOW (new)
// ---------------------------

// Open email modal and disable Run button until process starts
function openEmailModalBeforeRun() {
    const modal = document.getElementById("emailModal");
    if (modal) modal.style.display = "flex";
    const runBtn = document.getElementById("runBtn");
    if (runBtn) runBtn.disabled = true;
}

// Close modal and re-enable run button (used when user cancels)
function closeEmailModal() {
    const modal = document.getElementById("emailModal");
    if (modal) modal.style.display = "none";
    const runBtn = document.getElementById("runBtn");
    if (runBtn) runBtn.disabled = false;
}

// Called when user clicks "Start Process" inside the modal
async function startProcessAfterEmail() {
    const emailTo = document.getElementById("email_input").value.trim();
    const cc = document.getElementById("cc_input").value.trim();
    const subject = document.getElementById("subject_input").value.trim();
    const message = document.getElementById("message_input").value.trim();

    if (!emailTo) {
        showToast("Please enter at least one recipient email", "error");
        return;
    }

    // hide modal and start process
    closeEmailModal();
    showToast("Starting process...", "info");

    // Call runProcess passing email details
    await runProcess(emailTo, cc, subject, message);
}

// Run Process (unchanged behavior internally but now accepts email fields)
async function runProcess(emailTo = "", emailCc = "", emailSubject = "", emailMessage = "") {
    const bar = document.getElementById("progress_bar");
    if (bar) bar.style.width = "10%";
    showToast("Running... please wait");

    const folderLink = document.getElementById("folder_link").value;
    const borrowerCol = document.getElementById("borrower_col").value;
    const caseType = document.getElementById("case_type").value || "";

    const patterns = {};
    document.querySelectorAll(".doc_chk").forEach(chk => {
        if (chk.checked) {
            const doc = chk.dataset.doc;
            const pattern = chk.closest("tr").querySelector(".doc_pattern").value;
            patterns[doc] = pattern;
        }
    });

    const reqBody = {
        folder_link: folderLink,
        excel_path: excelPath,
        borrower_col: borrowerCol,
        case_type: caseType,
        doc_patterns: patterns,
        // include email fields so backend can send automatically once done
        email_to: emailTo,
        email_cc: emailCc,
        email_subject: emailSubject || "Automated Report",
        message: emailMessage || "Please find attached the report."
    };

    try {
        const res = await fetch("/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(reqBody)
        });

        if (!res.ok) {
            const json = await res.json().catch(() => ({}));
            showToast(json.error || "Process failed", "error");
            const runBtn = document.getElementById("runBtn");
            if (runBtn) runBtn.disabled = false;
            if (bar) bar.style.width = "100%";
            return;
        }

        if (bar) bar.style.width = "50%";

        const reportFilenameHeader = res.headers.get("X-Report-Filename") || "";
        const emailStatus = res.headers.get("X-Email-Status") || "";
        const blob = await res.blob();

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        const suggestedName = reportFilenameHeader ? osPathBasename(reportFilenameHeader) : "Result_Report.xlsx";
        a.href = url;
        a.download = suggestedName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);

        if (bar) bar.style.width = "90%";
        showToast("Process completed and file downloaded!");

        // Show OCR button only if email status is success
        if (emailStatus && emailStatus.toLowerCase().startsWith("success")) {
            const ocrBtn = document.getElementById("ocr_btn");
            if (ocrBtn) ocrBtn.style.display = "inline-block";
            showToast("Email sent successfully!", "info");
        } else if (emailStatus) {
            showToast("Process completed but email failed.", "error");
        } else {
            // No email requested; do not show OCR automatically
        }

    } catch (err) {
        console.error("runProcess error", err);
        showToast("Unexpected error during process", "error");
    } finally {
        const runBtn = document.getElementById("runBtn");
        if (runBtn) runBtn.disabled = false;
        if (bar) bar.style.width = "100%";
    }
}

// small helper to get basename without depending on Node
function osPathBasename(p) {
    if (!p) return "";
    const parts = p.split(/[\\/]/);
    return parts[parts.length - 1];
}

//
// --- CASE-TYPE MANAGEMENT UI actions
//

async function promptAddCaseType() {
    const name = prompt("Enter new Case Type name:");
    if (!name) return;
    const res = await fetch("/add_case_type", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_type: name })
    });
    const json = await res.json();
    if (res.ok) {
        // add to dropdown
        const sel = document.getElementById("case_type");
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
        sel.value = name;
        // trigger change to display docs (none)
        sel.dispatchEvent(new Event('change'));
        showToast("Case type added");
    } else {
        showToast(json.error || "Failed to add case type", "error");
    }
}

async function promptDeleteCaseType() {
    const caseType = document.getElementById("case_type").value;
    if (!caseType) return showToast("Select a case type to delete", "error");
    if (!confirm(`Delete case type "${caseType}"? This will remove all its documents.`)) return;
    const res = await fetch("/delete_case_type", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_type: caseType })
    });
    const json = await res.json();
    if (res.ok) {
        // remove from dropdown
        const sel = document.getElementById("case_type");
        const opt = sel.querySelector(`option[value="${caseType}"]`);
        if (opt) opt.remove();
        sel.value = "";
        document.getElementById("doc_table").style.display = "none";
        document.getElementById("case_docs_area").style.display = "none";
        showToast("Case type deleted");
    } else {
        showToast(json.error || "Failed to delete case type", "error");
    }
}

function openAddDocumentModal() {
    if (!currentCaseType) return showToast("Select a case type first", "error");
    editingDocOldName = "";
    document.getElementById("docModalTitle").textContent = `Add Document to "${currentCaseType}"`;
    document.getElementById("doc_name_input").value = "";
    document.getElementById("doc_pattern_input").value = "";
    document.getElementById("docModal").style.display = "flex";
}

function openEditDocModal(docName, pattern) {
    if (!currentCaseType) return showToast("Select a case type first", "error");
    editingDocOldName = docName;
    document.getElementById("docModalTitle").textContent = `Edit Document in "${currentCaseType}"`;
    document.getElementById("doc_name_input").value = docName;
    document.getElementById("doc_pattern_input").value = pattern;
    document.getElementById("docModal").style.display = "flex";
}

function closeDocModal() {
    document.getElementById("docModal").style.display = "none";
    editingDocOldName = "";
}

async function saveDocument() {
    const name = document.getElementById("doc_name_input").value.trim();
    const pattern = document.getElementById("doc_pattern_input").value.trim();
    if (!name) return showToast("Enter document name", "error");
    if (!currentCaseType) return showToast("Select a case type", "error");

    if (editingDocOldName) {
        // edit
        const res = await fetch("/edit_document", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                case_type: currentCaseType,
                old_name: editingDocOldName,
                new_name: name,
                pattern: pattern
            })
        });
        const json = await res.json();
        if (res.ok) {
            // refresh view
            await refreshCaseTypeDocs(currentCaseType);
            showToast("Document updated");
            closeDocModal();
        } else {
            showToast(json.error || "Failed to update document", "error");
        }
    } else {
        // add
        const res = await fetch("/add_document", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                case_type: currentCaseType,
                document_name: name,
                pattern: pattern
            })
        });
        const json = await res.json();
        if (res.ok) {
            await refreshCaseTypeDocs(currentCaseType);
            showToast("Document added");
            closeDocModal();
        } else {
            showToast(json.error || "Failed to add document", "error");
        }
    }
}

async function deleteDocument(docName) {
    if (!currentCaseType) return showToast("Select a case type", "error");
    if (!confirm(`Delete document "${docName}" from "${currentCaseType}"?`)) return;
    const res = await fetch("/delete_document", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_type: currentCaseType, document_name: docName })
    });
    const json = await res.json();
    if (res.ok) {
        await refreshCaseTypeDocs(currentCaseType);
        showToast("Document deleted");
    } else {
        showToast(json.error || "Failed to delete document", "error");
    }
}

async function refreshCaseTypeDocs(caseType) {
    // re-fetch documents & update both doc table and management area
    const res = await fetch("/get_documents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ caseType })
    });
    const json = await res.json();
    // update doc table
    const tbody = document.querySelector("#doc_table tbody");
    tbody.innerHTML = "";
    json.documents.forEach(d => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><input type="checkbox" class="doc_chk" data-doc="${escapeHtml(d.documentName)}" checked></td>
            <td>${escapeHtml(d.documentName)}</td>
            <td><input type="text" class="doc_pattern" value="${escapeHtml(d.defaultPattern)}"></td>
        `;
        tbody.appendChild(tr);
    });
    document.getElementById("doc_table").style.display = "table";
    populateDocsManagementArea(caseType, json.documents);
}

//
// Initialize — ensure UI reflects server case types (in case page reloaded)
(async function initCaseTypeDropdown() {
    try {
        const res = await fetch("/get_all_case_types");
        const json = await res.json();
        const sel = document.getElementById("case_type");
        // clear existing server-rendered options (except placeholder)
        // preserve the first placeholder option (index 0)
        for (let i = sel.options.length - 1; i >= 1; i--) sel.remove(i);
        Object.keys(json.case_types || {}).forEach(k => {
            const opt = document.createElement("option");
            opt.value = k;
            opt.textContent = k;
            sel.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to init case type dropdown", err);
    }
})();

async function promptEditCaseType() {
    const sel = document.getElementById("case_type");
    const oldName = sel.value;

    if (!oldName) return showToast("Select a case type to edit", "error");

    const newName = prompt(`Enter new name for "${oldName}":`, oldName);
    if (!newName || newName.trim() === oldName) return;

    const res = await fetch("/edit_case_type", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old_case_type: oldName, new_case_type: newName.trim() })
    });

    const json = await res.json();

    if (res.ok) {
        // Update dropdown
        const opt = sel.querySelector(`option[value="${oldName}"]`);
        if (opt) {
            opt.value = newName;
            opt.textContent = newName;
        }
        sel.value = newName;

        // Update global case type reference
        if (typeof currentCaseType !== "undefined") {
            currentCaseType = newName;
        }

        showToast("Case type renamed successfully");
        await refreshCaseTypeDocs(newName);
    } else {
        showToast(json.error || "Failed to rename case type", "error");
    }
}
