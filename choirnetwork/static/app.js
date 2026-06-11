const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsSection = document.getElementById("results-section");
const resultsTitle = document.getElementById("results-title");
const resultsMeta = document.getElementById("results-meta");
const resultsList = document.getElementById("results-list");
const topKControl = document.getElementById("top-k-control");
const thresholdControl = document.getElementById("threshold-control");
const topKInput = document.getElementById("top-k");
const topKValue = document.getElementById("top-k-value");
const minScoreInput = document.getElementById("min-score");
const minScoreValue = document.getElementById("min-score-value");
const modeInputs = document.querySelectorAll('input[name="mode"]');
const llmExpandInput = document.getElementById("llm-expand");
const llmExpandLabel = document.getElementById("llm-expand-label");
const llmExpandHint = document.getElementById("llm-expand-hint");

function getSelectedMode() {
  return document.querySelector('input[name="mode"]:checked')?.value ?? "top_k";
}

function updateModeUI() {
  const isThreshold = getSelectedMode() === "threshold";
  topKControl.classList.toggle("hidden", isThreshold);
  thresholdControl.classList.toggle("hidden", !isThreshold);
}

function setStatus(message, type = "info") {
  statusEl.textContent = message;
  statusEl.className = `status ${type}`;
  statusEl.classList.remove("hidden");
}

function clearStatus() {
  statusEl.classList.add("hidden");
  statusEl.textContent = "";
}

function formatPercent(score) {
  return `${Math.round(score * 100)}%`;
}

function expansionLabel(source) {
  if (source === "curated+llm") return "curated themes validated by AI";
  if (source === "curated") return "curated topic expansion";
  if (source === "llm") return "AI topic expansion";
  return null;
}

function renderResults(data) {
  resultsSection.classList.remove("hidden");
  resultsTitle.textContent = "Recommended hymns";

  const expansion = expansionLabel(data.expansion_source);
  let meta =
    data.count === 0
      ? `No hymns matched for "${data.query}". Try lowering the threshold or broadening the title.`
      : `Showing ${data.count} hymn${data.count === 1 ? "" : "s"} for "${data.query}".`;

  if (expansion) {
    meta += ` Search used ${expansion}.`;
  }

  resultsMeta.textContent = meta;

  resultsList.innerHTML = "";
  data.results.forEach((hymn, index) => {
    const item = document.createElement("li");
    item.className = "result-card";
    item.innerHTML = `
      <div class="result-rank" aria-hidden="true">${index + 1}</div>
      <div class="result-body">
        <h3>Hymn ${hymn.label} — ${hymn.title}</h3>
        <p>${formatPercent(hymn.score)} thematic match</p>
        <div class="score-bar" aria-hidden="true">
          <div class="score-fill" style="width: ${Math.max(hymn.score * 100, 4)}%"></div>
        </div>
      </div>
      <a class="result-link" href="${hymn.url}" target="_blank" rel="noopener noreferrer">
        Open hymnal
      </a>
    `;
    resultsList.appendChild(item);
  });
}

async function runSearch(event) {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Please enter a sermon title.", "error");
    return;
  }

  const mode = getSelectedMode();
  const payload = {
    query,
    mode,
    top_k: Number(topKInput.value),
    min_score: Number(minScoreInput.value) / 100,
    use_llm_expansion: llmExpandInput.checked,
  };

  submitBtn.disabled = true;
  submitBtn.textContent = "Searching...";
  clearStatus();

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Search failed. Please try again.");
    }

    const data = await response.json();
    renderResults(data);
  } catch (error) {
    resultsSection.classList.add("hidden");
    setStatus(error.message, "error");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Find hymns";
  }
}

topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});

minScoreInput.addEventListener("input", () => {
  minScoreValue.textContent = `${minScoreInput.value}%`;
});

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) return;

    const data = await response.json();
    if (data.llm_available) {
      llmExpandInput.checked = Boolean(data.default_llm_expand);
      return;
    }

    llmExpandInput.checked = false;
    llmExpandInput.disabled = true;
    llmExpandLabel.classList.add("disabled");
    llmExpandHint.textContent =
      "Add OPENAI_API_KEY to .env and restart the server to enable AI refinement.";
  } catch {
    // Health check is best-effort; search still works without LLM.
  }
}

modeInputs.forEach((input) => input.addEventListener("change", updateModeUI));
form.addEventListener("submit", runSearch);
updateModeUI();
loadHealth();
