// ==========================================================================
// TELCO CHURN INTELLIGENCE - INTERACTIVE JAVASCRIPT APP
// ==========================================================================

const CIRCUMFERENCE = 2 * Math.PI * 90; // 565.48px for r=90

let currentBatchData = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  loadModelInfo();
  setupDynamicFormInteractions();
  predictSingleCustomer(); // Initial prediction with default values
  setupDragAndDrop();
});

// Setup dynamic reactive interactions on form elements
function setupDynamicFormInteractions() {
  const form = document.getElementById("churn-form");
  if (!form) return;

  // Internet service change
  const internetSelect = document.getElementById("InternetService");
  if (internetSelect) {
    internetSelect.addEventListener("change", handleInternetServiceChange);
  }

  // Phone service change
  const phoneSelect = document.getElementById("PhoneService");
  if (phoneSelect) {
    phoneSelect.addEventListener("change", handlePhoneServiceChange);
  }

  // Auto-predict on change of ANY select
  form.querySelectorAll("select").forEach(sel => {
    sel.addEventListener("change", () => {
      predictSingleCustomer();
    });
  });
}

function handleInternetServiceChange() {
  const inetVal = document.getElementById("InternetService").value;
  const internetAddons = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
  ];

  if (inetVal === "No") {
    internetAddons.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.value = "No internet service";
        el.disabled = true;
        el.style.opacity = "0.6";
      }
    });
  } else {
    internetAddons.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.disabled = false;
        el.style.opacity = "1";
        if (el.value === "No internet service") {
          el.value = "No";
        }
      }
    });
  }

  predictSingleCustomer();
}

function handlePhoneServiceChange() {
  const phoneVal = document.getElementById("PhoneService").value;
  const multiLines = document.getElementById("MultipleLines");

  if (phoneVal === "No") {
    if (multiLines) {
      multiLines.value = "No phone service";
      multiLines.disabled = true;
      multiLines.style.opacity = "0.6";
    }
  } else {
    if (multiLines) {
      multiLines.disabled = false;
      multiLines.style.opacity = "1";
      if (multiLines.value === "No phone service") {
        multiLines.value = "No";
      }
    }
  }

  predictSingleCustomer();
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".nav-tab-btn").forEach(el => el.classList.remove("active"));

  const targetTab = document.getElementById(`tab-${tabId}`);
  const targetBtn = document.getElementById(`tab-btn-${tabId}`);

  if (targetTab && targetBtn) {
    targetTab.classList.add("active");
    targetBtn.classList.add("active");
  }
}

// Slider helper
function updateSlider(id, displayValue) {
  const badge = document.getElementById(`${id}-val`);
  if (badge) badge.textContent = displayValue;
  predictSingleCustomer();
}

// Auto-estimate Total Charges if tenure changes
function autoCalculateTotal() {
  const tenure = parseFloat(document.getElementById("tenure").value) || 0;
  const monthly = parseFloat(document.getElementById("MonthlyCharges").value) || 0;
  const estimatedTotal = tenure === 0 ? monthly : Math.round(tenure * monthly * 0.98);
  
  const totalInput = document.getElementById("TotalCharges");
  totalInput.value = estimatedTotal;
  const badge = document.getElementById("TotalCharges-val");
  if (badge) badge.textContent = "$" + estimatedTotal.toFixed(2);
}

// Load Pre-configured Personas
async function loadPersona(personaKey) {
  try {
    const res = await fetch(`/api/sample-customer/${personaKey}`);
    if (!res.ok) throw new Error("Failed to load persona");
    const data = await res.json();

    // Populate form fields
    for (const [key, val] of Object.entries(data)) {
      const el = document.getElementById(key);
      if (el) {
        el.value = val;
        if (key === "tenure") {
          const badge = document.getElementById("tenure-val");
          if (badge) badge.textContent = val + " months";
        }
        if (key === "MonthlyCharges") {
          const badge = document.getElementById("MonthlyCharges-val");
          if (badge) badge.textContent = "$" + parseFloat(val).toFixed(2);
        }
        if (key === "TotalCharges") {
          const badge = document.getElementById("TotalCharges-val");
          if (badge) badge.textContent = "$" + parseFloat(val).toFixed(2);
        }
      }
    }

    // Sync internet and phone dependent state
    handleInternetServiceChange();
    handlePhoneServiceChange();

    // Trigger prediction
    predictSingleCustomer();
  } catch (err) {
    console.error("Error loading persona:", err);
  }
}

// Collect form data as JSON
function getFormData() {
  return {
    gender: document.getElementById("gender").value,
    SeniorCitizen: document.getElementById("SeniorCitizen").value,
    Partner: document.getElementById("Partner").value,
    Dependents: document.getElementById("Dependents").value,
    tenure: parseFloat(document.getElementById("tenure").value),
    PhoneService: document.getElementById("PhoneService").value,
    MultipleLines: document.getElementById("MultipleLines").value,
    InternetService: document.getElementById("InternetService").value,
    OnlineSecurity: document.getElementById("OnlineSecurity").value,
    OnlineBackup: document.getElementById("OnlineBackup").value,
    DeviceProtection: document.getElementById("DeviceProtection").value,
    TechSupport: document.getElementById("TechSupport").value,
    StreamingTV: document.getElementById("StreamingTV").value,
    StreamingMovies: document.getElementById("StreamingMovies").value,
    Contract: document.getElementById("Contract").value,
    PaperlessBilling: document.getElementById("PaperlessBilling").value,
    PaymentMethod: document.getElementById("PaymentMethod").value,
    MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
    TotalCharges: parseFloat(document.getElementById("TotalCharges").value),
  };
}

// Predict Single Customer
async function predictSingleCustomer() {
  const payload = getFormData();
  const circle = document.getElementById("gauge-circle");
  const probText = document.getElementById("churn-prob-num");
  const badge = document.getElementById("risk-badge");
  const badgeText = document.getElementById("risk-badge-text");
  const factorsContainer = document.getElementById("risk-factors-container");
  const recsContainer = document.getElementById("retention-actions-container");

  if (!probText || !circle) return;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error("Prediction request failed");
    const result = await res.json();

    const pct = result.churn_percentage;
    probText.textContent = `${pct}%`;

    // Gauge circle animation
    const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE;
    circle.style.strokeDasharray = `${CIRCUMFERENCE}`;
    circle.style.strokeDashoffset = `${offset}`;
    circle.style.stroke = result.risk_color;

    // Risk badge styling
    badge.className = `risk-badge ${result.risk_level.toLowerCase()}`;
    badgeText.textContent = `${result.risk_level} Risk Tier (${result.churn_prediction === "Yes" ? "Likely Churner" : "Likely Retained"})`;

    // Render Risk Factors
    factorsContainer.innerHTML = "";
    if (result.risk_factors && result.risk_factors.length > 0) {
      result.risk_factors.forEach(f => {
        const item = document.createElement("div");
        item.className = "factor-item";
        const isAnchor = f.includes("Strongest retention") || f.includes("Solid retention") || f.includes("established loyalty") || f.includes("very low") || f.includes("reduces friction");
        item.innerHTML = `<span class="factor-icon">${isAnchor ? '🛡️' : '⚠️'}</span><span>${f}</span>`;
        factorsContainer.appendChild(item);
      });
    }

    // Render Retention Playbook
    recsContainer.innerHTML = "";
    if (result.retention_actions && result.retention_actions.length > 0) {
      result.retention_actions.forEach(rec => {
        const card = document.createElement("div");
        card.className = "retention-card";
        card.innerHTML = `
          <div class="rec-title">${rec.title}</div>
          <div class="rec-action">${rec.action}</div>
          <div class="rec-impact">⚡ Impact: ${rec.impact}</div>
        `;
        recsContainer.appendChild(card);
      });
    }

  } catch (err) {
    console.error("Prediction error:", err);
    probText.textContent = "--%";
  }
}

// Setup Drag & Drop for CSV Upload
function setupDragAndDrop() {
  const dropzone = document.getElementById("dropzone");
  if (!dropzone) return;

  ["dragenter", "dragover"].forEach(event => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(event => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleCsvUpload(e.dataTransfer.files);
    }
  });
}

// Handle CSV File Upload
async function handleCsvUpload(files) {
  if (!files || files.length === 0) return;
  const file = files[0];

  const formData = new FormData();
  formData.append("file", file);

  const dropzone = document.getElementById("dropzone");
  dropzone.innerHTML = `<div class="dropzone-icon">⏳</div><div class="dropzone-title">Processing & Scoring ${file.name}...</div>`;

  try {
    const res = await fetch("/api/predict-batch", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Batch processing failed");
    }

    const data = await res.json();
    currentBatchData = data;

    // Show stats
    document.getElementById("batch-stats").style.display = "grid";
    document.getElementById("batch-total").textContent = data.total_records.toLocaleString();
    document.getElementById("batch-churners").textContent = data.predicted_churners.toLocaleString();
    document.getElementById("batch-rate").textContent = `${data.churn_rate_pct}%`;
    document.getElementById("batch-high").textContent = data.risk_breakdown.high.toLocaleString();

    // Render table
    renderBatchTable(data.columns, data.preview);

    // Reset dropzone
    dropzone.innerHTML = `
      <div class="dropzone-icon">✅</div>
      <div class="dropzone-title">Scored ${data.total_records} Customers Successfully!</div>
      <div class="dropzone-sub">Drop another CSV file to rescore.</div>
    `;

  } catch (err) {
    console.error("Batch error:", err);
    alert(`Upload Error: ${err.message}`);
    dropzone.innerHTML = `
      <div class="dropzone-icon">❌</div>
      <div class="dropzone-title">Error Scoring File</div>
      <div class="dropzone-sub">${err.message}</div>
    `;
  }
}

// Render Scored Data Table
function renderBatchTable(columns, rows) {
  const container = document.getElementById("batch-results-container");
  const thead = document.getElementById("batch-table-head");
  const tbody = document.getElementById("batch-table-body");

  container.style.display = "block";

  // Priority display columns
  const displayCols = [
    "customerID", "Contract", "tenure", "MonthlyCharges", "PaymentMethod", "Churn_Risk_Pct", "Risk_Tier", "Churn_Prediction"
  ].filter(c => columns.includes(c));

  // Build Head
  thead.innerHTML = `<tr>${displayCols.map(c => `<th>${c.replace(/_/g, " ")}</th>`).join("")}</tr>`;

  // Build Body
  tbody.innerHTML = rows.map(r => {
    return `<tr>${displayCols.map(c => {
      let val = r[c] !== undefined ? r[c] : "-";
      if (c === "Churn_Risk_Pct") val = `<strong>${val}%</strong>`;
      if (c === "Risk_Tier") {
        const tier = String(val).toLowerCase();
        val = `<span class="tag-pill ${tier}">${val}</span>`;
      }
      if (c === "Churn_Prediction") {
        const isChurn = val === "Yes";
        val = `<span class="tag-pill ${isChurn ? 'high' : 'low'}">${val}</span>`;
      }
      return `<td>${val}</td>`;
    }).join("")}</tr>`;
  }).join("");
}

// Export preview table as CSV download
function exportScoredTableToCsv() {
  if (!currentBatchData || !currentBatchData.preview) return;

  const rows = currentBatchData.preview;
  if (rows.length === 0) return;

  const keys = Object.keys(rows[0]);
  const csvContent = [
    keys.join(","),
    ...rows.map(row => keys.map(k => `"${String(row[k] ?? '').replace(/"/g, '""')}"`).join(","))
  ].join("\n");

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", "telco_customer_churn_scored.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Load Model Metadata and Benchmark Scoreboard
async function loadModelInfo() {
  try {
    const res = await fetch("/api/model-info");
    if (!res.ok) return;
    const meta = await res.json();

    // Populate Overview Stats
    document.getElementById("meta-model-name").textContent = meta.model_name || "Champion Model";
    if (meta.test_metrics) {
      document.getElementById("meta-roc-auc").textContent = meta.test_metrics.roc_auc.toFixed(4);
      document.getElementById("meta-recall").textContent = `${(meta.test_metrics.recall * 100).toFixed(1)}%`;
      document.getElementById("meta-thresh").textContent = meta.optimal_threshold.toFixed(3);
    }

    // Populate Benchmark Table
    const tbody = document.getElementById("benchmark-tbody");
    if (tbody && meta.benchmark_summary) {
      tbody.innerHTML = meta.benchmark_summary.map((b, idx) => {
        const isBest = idx === 0;
        return `
          <tr style="${isBest ? 'background: rgba(99, 102, 241, 0.12); font-weight: 600;' : ''}">
            <td>${b.model_name} ${isBest ? '👑' : ''}</td>
            <td>${b.roc_auc.toFixed(4)} ± ${b.std_roc_auc ? b.std_roc_auc.toFixed(4) : '0.008'}</td>
            <td>${(b.f1 * 100).toFixed(1)}%</td>
            <td>${(b.recall * 100).toFixed(1)}%</td>
            <td>${(b.accuracy * 100).toFixed(1)}%</td>
            <td><span class="tag-pill ${isBest ? 'high' : 'low'}">${isBest ? 'CHAMPION' : 'CANDIDATE'}</span></td>
          </tr>
        `;
      }).join("");
    }
  } catch (err) {
    console.error("Error loading model info:", err);
  }
}
