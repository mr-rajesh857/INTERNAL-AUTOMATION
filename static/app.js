// // app.js
// let uploadedExcelPath = null;
// let detectedColumns = [];
// let lastProcessType = "verification"; // track last action for email subject

// // -------------------- UPLOAD EXCEL --------------------
// document.getElementById("uploadBtn").addEventListener("click", async () => {
//   const fileInput = document.getElementById("excelInput");
//   if (!fileInput.files.length) {
//     alert("Please select an Excel file (.xlsx)");
//     return;
//   }
//   const file = fileInput.files[0];
//   const fd = new FormData();
//   fd.append("excel", file);

//   document.getElementById("uploadMsg").innerText = "Uploading...";
//   const res = await fetch("/upload", { method: "POST", body: fd });
//   const j = await res.json();
//   if (j.error) {
//     document.getElementById("uploadMsg").innerText = "Error: " + j.error;
//     return;
//   }
//   uploadedExcelPath = j.excel_path;
//   detectedColumns = j.document_columns || [];
//   document.getElementById("uploadMsg").innerText = "Detected " + detectedColumns.length + " columns after 'Folder Link'.";
//   renderDetected(detectedColumns);
// });

// // -------------------- RENDER DETECTED --------------------
// function renderDetected(cols) {
//   const container = document.getElementById("columnsContainer");
//   container.innerHTML = "";
//   const verifyList = document.getElementById("verifyList");
//   verifyList.innerHTML = "";
//   const mergeList = document.getElementById("mergeList");
//   mergeList.innerHTML = "";

//   const info = document.createElement("div");
//   info.innerHTML = `<strong>Columns:</strong> ${cols.join(", ")}`;
//   container.appendChild(info);

//   // verify checkboxes
//   cols.forEach(c => {
//     const chip = document.createElement("label");
//     chip.className = "chip";
//     chip.innerHTML = `<input type="checkbox" value="${c}"> ${c}`;
//     verifyList.appendChild(chip);
//   });

//   // merge sequence list (draggable)
//   cols.forEach((c, i) => {
//     const li = document.createElement("li");
//     li.draggable = true;
//     li.innerText = c;
//     li.id = "merge-" + i;
//     li.addEventListener("dragstart", dragStart);
//     li.addEventListener("dragover", dragOver);
//     li.addEventListener("drop", dropItem);
//     mergeList.appendChild(li);
//   });

//   document.getElementById("chooseSection").style.display = "block";
//   document.getElementById("emailSection").style.display = "block";
// }

// // -------------------- DRAG & DROP --------------------
// let dragged = null;
// function dragStart(e) { dragged = this; e.dataTransfer.effectAllowed = "move"; }
// function dragOver(e) { e.preventDefault(); this.classList.add("drag-over"); }
// function dropItem(e) {
//   e.preventDefault();
//   this.classList.remove("drag-over");
//   const list = this.parentNode;
//   if (dragged === this) return;
//   list.insertBefore(dragged, this);
// }

// // -------------------- START PROCESSING --------------------
// document.getElementById("startBtn").addEventListener("click", async () => {
//   if (!uploadedExcelPath) { alert("Upload excel first."); return; }

//   const checks = Array.from(document.querySelectorAll("#verifyList input[type=checkbox]"))
//                       .filter(i => i.checked).map(i => i.value);
//   if (!checks.length) { alert("Select at least one document to VERIFY."); return; }

//   const mergeItems = Array.from(document.querySelectorAll("#mergeList li")).map(li => li.innerText.trim());
//   if (!mergeItems.length) { alert("No merge columns present."); return; }

// const payload = {
//   excel_path: uploadedExcelPath,
//   verify_docs: checks,
//   merge_sequence: mergeItems,
//   use_easyocr: document.getElementById("useEasyOCR").checked,

//   // Email fields
//   ver_to: document.getElementById("ver_to").value,
//   ver_cc: document.getElementById("ver_cc").value,
//   ver_subject: document.getElementById("ver_subject").value,

//   // merge_to: document.getElementById("merge_to").value,
//   // merge_cc: document.getElementById("merge_cc").value,
//   // merge_subject: document.getElementById("merge_subject").value
// };


//   document.getElementById("logsSection").style.display = "block";
//   document.getElementById("logs").innerText = "Processing. This may take time depending on files...";

//   const res = await fetch("/process", { method: "POST", headers: { "Content-Type":"application/json" }, body: JSON.stringify(payload) });
//   const j = await res.json();
//   if (j.error) {
//     document.getElementById("logs").innerText = "Error: " + j.error;
//     return;
//   }

//   document.getElementById("logs").innerText = j.print_log || "";
//   document.getElementById("summary").innerText = JSON.stringify(j.summary || {}, null, 2);
//   lastProcessType = "verification";

//   if (j.summary && j.summary["Merged PDFs location"]) {
//     const href = j.summary["Merged PDFs location"];
//     const a = document.getElementById("downloadMerged");
//     a.href = "#";
//     a.style.display = "inline-block";
//     a.onclick = () => { alert("Merged PDFs saved server-side at: " + href); };
//   }
// });

// // -------------------- EMAIL MODAL --------------------
// const emailModal = document.getElementById("emailModal");
// const emailTo = document.getElementById("emailTo");
// const emailCc = document.getElementById("emailCc");
// const emailSubject = document.getElementById("emailSubject");
// const emailBody = document.getElementById("emailBody");
// const emailStatus = document.getElementById("emailStatus");

// document.getElementById("sendEmailBtn").addEventListener("click", openEmailModal);
// document.getElementById("cancelEmail").addEventListener("click", closeEmailModal);
// document.getElementById("emailModalClose").addEventListener("click", closeEmailModal);

// // Open modal
// function openEmailModal() {
//   emailTo.value = "";
//   emailCc.value = "";
//   emailSubject.value = lastProcessType === "merge" ? "Merged Log File" : "Verification Log File";
//   emailBody.value = "Please find the attached log file.";
//   emailStatus.innerText = "";
//   emailModal.style.display = "flex";
// }

// // Close modal
// function closeEmailModal() {
//   emailModal.style.display = "none";
// }

// // Submit email form
// document.getElementById("emailForm").addEventListener("submit", async (e) => {
//   e.preventDefault();
//   const to = emailTo.value.trim();
//   if (!to) { alert("TO email required."); return; }

//   const payload = {
//     to: to,
//     cc: emailCc.value.trim(),
//     subject: emailSubject.value.trim(),
//     body: emailBody.value.trim(),
//     type: lastProcessType
//   };

//   emailStatus.innerText = "Sending...";
//   try {
//     const res = await fetch("/send-email", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify(payload)
//     });
//     const j = await res.json();
//     if (j.error) {
//       emailStatus.innerText = "Error: " + j.error;
//     } else {
//       emailStatus.innerText = "Email sent successfully!";
//       setTimeout(closeEmailModal, 1500);
//     }
//   } catch (err) {
//     emailStatus.innerText = "Error: " + err.message;
//   }
// });

// app.js (corrected)

let uploadedExcelPath = null;
let detectedColumns = [];
let lastProcessType = "verification"; // track last action for email subject
let lastProcessResponse = null; // store server response for later (paths)

// -------------------- UPLOAD EXCEL --------------------
document.getElementById("uploadBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("excelInput");
  if (!fileInput.files.length) {
    alert("Please select an Excel file (.xlsx)");
    return;
  }
  const file = fileInput.files[0];
  const fd = new FormData();
  fd.append("excel", file);

  document.getElementById("uploadMsg").innerText = "Uploading...";
  const res = await fetch("/upload", { method: "POST", body: fd });
  const j = await res.json();
  if (j.error) {
    document.getElementById("uploadMsg").innerText = "Error: " + j.error;
    return;
  }
  uploadedExcelPath = j.excel_path;
  detectedColumns = j.document_columns || [];
  document.getElementById("uploadMsg").innerText = "Detected " + detectedColumns.length + " columns after 'Folder Link'.";
  renderDetected(detectedColumns);
});

// -------------------- RENDER DETECTED --------------------
function renderDetected(cols) {
  const container = document.getElementById("columnsContainer");
  container.innerHTML = "";
  const verifyList = document.getElementById("verifyList");
  verifyList.innerHTML = "";
  const mergeList = document.getElementById("mergeList");
  mergeList.innerHTML = "";

  const info = document.createElement("div");
  info.innerHTML = `<strong>Columns:</strong> ${cols.join(", ")}`;
  container.appendChild(info);

  // verify checkboxes
  cols.forEach(c => {
    const chip = document.createElement("label");
    chip.className = "chip";
    chip.innerHTML = `<input type="checkbox" value="${c}"> ${c}`;
    verifyList.appendChild(chip);
  });

  // merge sequence list (draggable)
  cols.forEach((c, i) => {
    const li = document.createElement("li");
    li.draggable = true;
    li.innerText = c;
    li.id = "merge-" + i;
    li.addEventListener("dragstart", dragStart);
    li.addEventListener("dragover", dragOver);
    li.addEventListener("drop", dropItem);
    mergeList.appendChild(li);
  });

  document.getElementById("chooseSection").style.display = "block";
  document.getElementById("emailSection").style.display = "block";
}

// -------------------- DRAG & DROP --------------------
let dragged = null;
function dragStart(e) { dragged = this; e.dataTransfer.effectAllowed = "move"; }
function dragOver(e) { e.preventDefault(); this.classList.add("drag-over"); }
function dropItem(e) {
  e.preventDefault();
  this.classList.remove("drag-over");
  const list = this.parentNode;
  if (dragged === this) return;
  list.insertBefore(dragged, this);
}

// -------------------- START PROCESSING --------------------
document.getElementById("startBtn").addEventListener("click", async () => {
  if (!uploadedExcelPath) { alert("Upload excel first."); return; }

  const checks = Array.from(document.querySelectorAll("#verifyList input[type=checkbox]"))
                      .filter(i => i.checked).map(i => i.value);
  if (!checks.length) { alert("Select at least one document to VERIFY."); return; }

  const mergeItems = Array.from(document.querySelectorAll("#mergeList li")).map(li => li.innerText.trim());
  if (!mergeItems.length) { alert("No merge columns present."); return; }

  const payload = {
    excel_path: uploadedExcelPath,
    verify_docs: checks,
    merge_sequence: mergeItems,
    use_easyocr: document.getElementById("useEasyOCR").checked,

    // Email fields for verification (frontend will trigger verification email after process completes)
    ver_to: document.getElementById("ver_to").value,
    ver_cc: document.getElementById("ver_cc").value,
    ver_subject: document.getElementById("ver_subject").value
  };

  document.getElementById("logsSection").style.display = "block";
  document.getElementById("logs").innerText = "Processing. This may take time depending on files...";

  try {
    const res = await fetch("/process", { method: "POST", headers: { "Content-Type":"application/json" }, body: JSON.stringify(payload) });
    const j = await res.json();
    if (j.error) {
      document.getElementById("logs").innerText = "Error: " + j.error;
      return;
    }

    // store for later (merge email)
    lastProcessResponse = j;

    // show logs & summary
    document.getElementById("logs").innerText = j.print_log || "";
    document.getElementById("summary").innerText = JSON.stringify(j.summary || {}, null, 2);
    lastProcessType = "verification";

    // show merged folder link if present
    if (j.summary && j.summary["Merged PDFs location"]) {
      const href = j.summary["Merged PDFs location"];
      const a = document.getElementById("downloadMerged");
      a.href = "#";
      a.style.display = "inline-block";
      a.onclick = () => { alert("Merged PDFs saved server-side at: " + href); };
    }

    // --- Immediately send verification email (frontend triggers it) ---
    // Only do this if frontend provided a ver_to; otherwise skip and let user know
    if (j.verification_log_path) {
      const verTo = payload.ver_to && payload.ver_to.trim();
      if (verTo) {
        const verPayload = {
          to: verTo,
          cc: payload.ver_cc && payload.ver_cc.trim(),
          subject: payload.ver_subject && payload.ver_subject.trim(),
          body: "Attached verification log from the verification process.",
          attachments: [ j.verification_log_path ] // server-side path
        };
        // best-effort send
        try {
          const sendRes = await fetch("/send-email", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(verPayload)
          });
          const sendJson = await sendRes.json();
          if (sendJson && sendJson.success) {
            // show small success in logs
            const logsEl = document.getElementById("logs");
            logsEl.innerText = (logsEl.innerText || "") + "\n\n✅ Verification email sent to " + verTo;
          } else {
            const logsEl = document.getElementById("logs");
            logsEl.innerText = (logsEl.innerText || "") + `\n\n⚠ Verification email not sent (server returned failure).`;
          }
        } catch (err) {
          const logsEl = document.getElementById("logs");
          logsEl.innerText = (logsEl.innerText || "") + `\n\n⚠ Verification email failed: ${err.message}`;
        }
      } else {
        const logsEl = document.getElementById("logs");
        logsEl.innerText = (logsEl.innerText || "") + `\n\nℹ Verification email not sent — no recipient entered in "Verification Log - To Email".`;
      }
    }

    // --- If merge log exists, open merge email modal for user to provide merge email details ---
    if (j.merge_log_path) {
      openMergeModal();
    } else {
      // no merge log present
      // nothing to do
    }

  } catch (err) {
    document.getElementById("logs").innerText = "Error: " + err.message;
  }
});

// -------------------- MERGE EMAIL MODAL HANDLING --------------------
const mergeModal = document.getElementById("mergeEmailModal");
const mergeTo = document.getElementById("merge_to");
const mergeCc = document.getElementById("merge_cc");
const mergeSubject = document.getElementById("merge_subject");
const mergeStatus = document.getElementById("mergeEmailStatus");
const sendMergeBtn = document.getElementById("sendMergeEmail");
const cancelMergeBtn = document.getElementById("cancelMergeEmail");
const closeMergeBtn = document.getElementById("mergeModalClose");

function openMergeModal() {
  // prefill subject with default (user can edit)
  mergeTo.value = "";
  mergeCc.value = "";
  mergeSubject.value = "";
  mergeStatus.innerText = "";
  mergeModal.style.display = "flex";
  mergeModal.setAttribute("aria-hidden", "false");
}

function closeMergeModal() {
  mergeModal.style.display = "none";
  mergeModal.setAttribute("aria-hidden", "true");
}

// bind buttons
cancelMergeBtn.addEventListener("click", (e) => { e.preventDefault(); closeMergeModal(); });
closeMergeBtn.addEventListener("click", (e) => { e.preventDefault(); closeMergeModal(); });

sendMergeBtn.addEventListener("click", async (e) => {
  e.preventDefault();
  mergeStatus.style.color = "#333";

  const to = mergeTo.value && mergeTo.value.trim();
  if (!to) {
    mergeStatus.style.color = "red";
    mergeStatus.innerText = "Recipient email (To) is required.";
    return;
  }

  // determine subject fallback
  let subj = mergeSubject.value && mergeSubject.value.trim();
  if (!subj) {
    const now = new Date();
    const ts = `${now.getFullYear()}_${(now.getMonth()+1).toString().padStart(2,"0")}_${now.getDate().toString().padStart(2,"0")}_${now.getHours().toString().padStart(2,"0")}${now.getMinutes().toString().padStart(2,"0")}${now.getSeconds().toString().padStart(2,"0")}`;
    subj = `Merge Log Report - ${ts}`;
  }

  // attachments: take merge_log_path from lastProcessResponse
  if (!lastProcessResponse || !lastProcessResponse.merge_log_path) {
    mergeStatus.style.color = "red";
    mergeStatus.innerText = "No merge log available on server to attach.";
    return;
  }

  const payload = {
    to: to,
    cc: mergeCc.value && mergeCc.value.trim(),
    subject: subj,
    body: "Attached merge log from the merge process.",
    attachments: [ lastProcessResponse.merge_log_path ]
  };

  mergeStatus.innerText = "Sending...";
  try {
    const res = await fetch("/send-email", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const j = await res.json();
    if (j && j.success) {
      mergeStatus.style.color = "green";
      mergeStatus.innerText = "Merge email sent successfully!";
      setTimeout(closeMergeModal, 1200);
    } else {
      mergeStatus.style.color = "red";
      mergeStatus.innerText = "Merge email failed (server returned failure).";
    }
  } catch (err) {
    mergeStatus.style.color = "red";
    mergeStatus.innerText = "Merge email failed: " + err.message;
  }
});
