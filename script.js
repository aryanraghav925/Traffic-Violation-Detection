/* ============================================================
   VigileAI — script.js
   ============================================================ */

const API = "http://127.0.0.1:5000";

/* ---- DOM REFERENCES ---- */
const fileInput    = document.getElementById("fileInput");
const dropZone     = document.getElementById("dropZone");   // now a <label>
const previewImg   = document.getElementById("previewImg");
const filenameTag  = document.getElementById("filenameTag");
const filenameText = document.getElementById("filenameText");
const detectBtn    = document.getElementById("detectBtn");
const loader       = document.getElementById("loader");
const resultsEl    = document.getElementById("results");
const originalImg  = document.getElementById("originalImg");
const outputImg    = document.getElementById("outputImg");
const errorMsg     = document.getElementById("errorMsg");

let selectedFile = null;

/* ============================================================
   FILE INPUT — change event
   Fires when user picks a file via the native dialog
   ============================================================ */
fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) {
    setFile(fileInput.files[0]);
  }
});

/* ============================================================
   DRAG & DROP
   Attached to the <label> drop zone.
   We call e.preventDefault() on dragover so the browser
   doesn't open the file, and stop propagation so the label's
   default "open file dialog" doesn't fire on drop.
   ============================================================ */
dropZone.addEventListener("dragenter", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  e.stopPropagation();           // prevents label from opening file dialog
  dropZone.classList.remove("dragover");

  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    setFile(file);
  } else {
    showError("Please drop a valid image file (JPG, PNG, WEBP).");
  }
});

/* ============================================================
   SET FILE — shared by input change + drag & drop
   ============================================================ */
function setFile(file) {
  selectedFile = file;

  // Show preview inside the drop zone
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.classList.add("visible");

  // Keep original for side-by-side comparison
  originalImg.src = url;

  // Show filename tag below drop zone
  filenameText.textContent = file.name;
  filenameTag.classList.add("visible");

  // Enable detect button
  detectBtn.disabled = false;

  // Clear old state
  resultsEl.classList.remove("visible");
  hideError();
}

/* ============================================================
   DETECTION — called by the Run Detection button
   ============================================================ */
async function runDetection() {
  if (!selectedFile) return;

  detectBtn.disabled = true;
  loader.classList.add("visible");
  resultsEl.classList.remove("visible");
  hideError();

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const res  = await fetch(`${API}/detect`, { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || "Detection failed.");

    outputImg.src = `${API}${data.output_image}`;
    renderViolations(data.violations);

    loader.classList.remove("visible");
    resultsEl.classList.add("visible");
    loadLog();

  } catch (err) {
    loader.classList.remove("visible");
    showError(`⚠ ${err.message}`);
  } finally {
    detectBtn.disabled = false;
  }
}

/* ============================================================
   RENDER VIOLATIONS
   ============================================================ */
function renderViolations(violations) {
  const list  = document.getElementById("violationsList");
  const count = document.getElementById("violationCount");

  count.textContent = violations.length;

  if (!violations.length) {
    list.innerHTML = `
      <div class="no-violations">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        No violations detected
      </div>`;
    return;
  }

  list.innerHTML = violations.map((v) => `
    <div class="violation-row">
      <div>
        <div class="plate-text">${v.plate || "UNREAD"}</div>
        <div class="riders-text">${v.riders} rider${v.riders !== 1 ? "s" : ""} detected</div>
      </div>
      <div class="pills">
        ${v.no_helmet ? '<span class="pill pill-red">NO HELMET</span>'   : ""}
        ${v.tripling  ? '<span class="pill pill-orange">TRIPLING</span>' : ""}
      </div>
    </div>`
  ).join("");
}

/* ============================================================
   VIOLATIONS LOG TABLE
   ============================================================ */
async function loadLog() {
  const body = document.getElementById("logBody");

  try {
    const res  = await fetch(`${API}/violations`);
    const rows = await res.json();

    if (!rows.length) {
      body.innerHTML = `
        <tr><td class="td-empty" colspan="4">No violations logged yet.</td></tr>`;
      return;
    }

    body.innerHTML = [...rows].reverse().map((r) => `
      <tr>
        <td>${r.Time}</td>
        <td class="td-plate">${r["License Plate"]}</td>
        <td class="${r.No_Helmet === "True" ? "td-true" : "td-false"}">
          ${r.No_Helmet === "True" ? "YES" : "NO"}
        </td>
        <td class="${r.Tripling === "True" ? "td-true" : "td-false"}">
          ${r.Tripling === "True" ? "YES" : "NO"}
        </td>
      </tr>`
    ).join("");

  } catch {
    // Flask not running yet — silently skip on page load
  }
}

/* ============================================================
   HELPERS
   ============================================================ */
function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add("visible");
}

function hideError() {
  errorMsg.classList.remove("visible");
  errorMsg.textContent = "";
}

/* ============================================================
   INIT
   ============================================================ */
loadLog();
