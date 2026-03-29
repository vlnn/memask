/* ============================================================
   FILE: gui/web/suggest-dropdown.md
   
   Three blocks to insert into gui/web/index.html.
   Each block has a placement marker showing WHERE to paste it.
   ============================================================ */


/* ── BLOCK 1: CSS ────────────────────────────────────────────
   Insert inside <style>, after the existing #spinner styles.
   ──────────────────────────────────────────────────────────── */

/* ── suggest dropdown ──────────────────────────────── */

#suggest-dropdown {
  display: none;
  position: absolute;
  left: 0;
  right: 0;
  top: 100%;
  z-index: 100;
  background: var(--bg);
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 8px 8px;
  max-height: 320px;
  overflow-y: auto;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

#suggest-dropdown.active {
  display: block;
}

.suggest-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 12px 7px 36px;
  cursor: pointer;
  font-size: 13px;
  color: var(--text);
  border-bottom: 1px solid var(--border-light);
}

.suggest-item:last-child {
  border-bottom: none;
}

.suggest-item.selected {
  background: var(--hover);
}

.suggest-item:hover {
  background: var(--hover);
}

.suggest-kind {
  flex-shrink: 0;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 1px 5px;
  border-radius: 3px;
  color: var(--text-faint);
  background: var(--border-light);
}

.suggest-kind.command {
  color: var(--accent);
  background: color-mix(in srgb, var(--accent) 12%, transparent);
}

.suggest-text {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.suggest-desc {
  flex-shrink: 0;
  font-size: 11px;
  color: var(--text-faint);
}

.suggest-type-badge {
  flex-shrink: 0;
  font-size: 10px;
  color: var(--text-faint);
  opacity: 0.7;
}


/* ── BLOCK 2: HTML ───────────────────────────────────────────
   Insert the dropdown div INSIDE #input-row, right after the
   <div id="spinner"></div> line.
   
   Also add position:relative to #input-row if not already set.
   ──────────────────────────────────────────────────────────── */

<!--
  Add style="position:relative" to #input-row:
  <div id="input-row" style="position:relative">
  
  Then after <div id="spinner"></div>, add:
-->
<div id="suggest-dropdown"></div>


/* ── BLOCK 3: JS ─────────────────────────────────────────────
   Insert this entire block inside <script>, before the
   /* ── init ── */ section.
   ──────────────────────────────────────────────────────────── */

/* ── suggest dropdown ─────────────────────────────── */

let suggestOpen = false;
let suggestIndex = -1;
let suggestItems = [];
let suggestDebounce = null;

async function fetchSuggestions(query) {
  const url = `${DAEMON}/suggest?q=${encodeURIComponent(query)}&limit=10`;
  const resp = await fetch(url);
  if (!resp.ok) return [];
  const data = await resp.json();
  return data.suggestions || [];
}

function renderSuggestions(suggestions) {
  const dropdown = document.getElementById("suggest-dropdown");
  suggestItems = suggestions;
  suggestIndex = -1;

  if (suggestions.length === 0) {
    closeSuggest();
    return;
  }

  dropdown.innerHTML = suggestions.map((s, i) => {
    const kindClass = s.kind === "command" ? "command" : "";
    const desc = s.description
      ? `<span class="suggest-desc">${esc(s.description)}</span>`
      : "";
    const badge = s.item_type
      ? `<span class="suggest-type-badge">${esc(s.item_type)}</span>`
      : "";
    return `<div class="suggest-item" data-index="${i}">
      <span class="suggest-kind ${kindClass}">${esc(s.kind)}</span>
      <span class="suggest-text">${esc(s.text)}</span>
      ${desc}${badge}
    </div>`;
  }).join("");

  dropdown.classList.add("active");
  suggestOpen = true;
}

function closeSuggest() {
  const dropdown = document.getElementById("suggest-dropdown");
  dropdown.classList.remove("active");
  dropdown.innerHTML = "";
  suggestOpen = false;
  suggestIndex = -1;
  suggestItems = [];
}

function selectSuggestion(index) {
  if (index < 0 || index >= suggestItems.length) return;
  const item = suggestItems[index];
  const input = document.getElementById("input");
  input.value = item.text;
  closeSuggest();
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
}

function navigateSuggest(direction) {
  if (!suggestOpen || suggestItems.length === 0) return;
  const items = document.querySelectorAll(".suggest-item");
  if (suggestIndex >= 0 && suggestIndex < items.length) {
    items[suggestIndex].classList.remove("selected");
  }
  suggestIndex += direction;
  if (suggestIndex < 0) suggestIndex = suggestItems.length - 1;
  if (suggestIndex >= suggestItems.length) suggestIndex = 0;
  items[suggestIndex].classList.add("selected");
  items[suggestIndex].scrollIntoView({ block: "nearest" });
}

function liveSuggestFilter() {
  if (!suggestOpen) return;
  clearTimeout(suggestDebounce);
  suggestDebounce = setTimeout(async () => {
    const query = document.getElementById("input").value;
    const suggestions = await fetchSuggestions(query);
    renderSuggestions(suggestions);
  }, 150);
}


/* ── BLOCK 4: REPLACE handleKeydown ──────────────────────────
   Replace the existing handleKeydown function with this one.
   ──────────────────────────────────────────────────────────── */

function handleKeydown(e) {
  if (e.key === "Tab") {
    e.preventDefault();
    if (suggestOpen) {
      closeSuggest();
    } else {
      const query = document.getElementById("input").value;
      fetchSuggestions(query).then(renderSuggestions);
    }
    return;
  }

  if (suggestOpen) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      navigateSuggest(1);
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      navigateSuggest(-1);
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      if (suggestIndex >= 0) {
        selectSuggestion(suggestIndex);
      } else {
        closeSuggest();
        handleSubmit();
      }
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      closeSuggest();
      return;
    }
    liveSuggestFilter();
    return;
  }

  if (e.key === "Enter") {
    e.preventDefault();
    handleSubmit();
  }
  if (e.key === "Escape") {
    e.preventDefault();
    hideWindow();
  }
}


/* ── BLOCK 5: ADD to init ────────────────────────────────────
   Add these two listeners inside the DOMContentLoaded handler,
   after the existing input keydown listener.
   ──────────────────────────────────────────────────────────── */

document.getElementById("input").addEventListener("input", () => {
  if (suggestOpen) liveSuggestFilter();
});

document.getElementById("suggest-dropdown").addEventListener("click", (e) => {
  const item = e.target.closest(".suggest-item");
  if (item) selectSuggestion(parseInt(item.dataset.index, 10));
});

document.addEventListener("click", (e) => {
  if (suggestOpen && !e.target.closest("#input-row")) closeSuggest();
});
