import { api } from "./api.js";

const byId = (id) => document.getElementById(id);

function numberValue(id, fallback) {
  const value = Number(byId(id).value);
  return Number.isFinite(value) ? value : fallback;
}

function csvValue(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function estimateWork(dimensions, rules, initialConditions, size, steps) {
  return rules * initialConditions * size ** Number(dimensions) * steps;
}

export class ExperimentController {
  constructor() {
    this.rows = [];
    this.sortKey = "final";
    this.sortDescending = true;
    this.bindControls();
  }

  activate() {}

  async run() {
    const dimensions = byId("experiment-dimensions").value;
    const rules = Math.trunc(numberValue("experiment-rules", 5));
    const initialConditions = Math.trunc(numberValue("experiment-conditions", 4));
    const size = Math.trunc(numberValue("experiment-size", 24));
    const steps = Math.trunc(numberValue("experiment-steps", 50));
    const density = numberValue("experiment-density", 0.1);
    const seed = Math.trunc(numberValue("experiment-seed", 42));
    const work = estimateWork(dimensions, rules, initialConditions, size, steps);
    const progress = byId("experiment-progress");
    const runButton = byId("experiment-run");
    runButton.disabled = true;
    byId("experiment-export").disabled = true;
    progress.className = "status-message status-running";
    progress.textContent = "Running experiment...";
    try {
      const limits = await api.experimentLimits();
      const browserLimit = Number(limits.browser_max_cell_updates);
      if (!Number.isSafeInteger(browserLimit) || browserLimit < 1) {
        throw new Error("The backend returned an invalid browser workload limit.");
      }
      if (work > browserLimit) {
        progress.className = "status-message status-error";
        progress.textContent = `This run is too large for the interactive server (${work.toLocaleString()} estimated cell-updates; limit ${browserLimit.toLocaleString()}). Use the CLI experiment runner.`;
        return;
      }
      const result = dimensions === "3"
        ? await api.experiment3d({ rules, initial_conditions: initialConditions, size, steps, density, seed })
        : await api.experiment2d({ rules, initial_conditions: initialConditions, size, steps, density, seed });
      this.rows = result.rows || [];
      this.renderTable();
      byId("experiment-export").disabled = !this.rows.length;
      byId("experiment-summary").textContent = `${rules} rules | ${this.rows.length} simulations`;
      progress.className = "status-message status-success";
      progress.textContent = `Completed ${rules} rules and ${this.rows.length} simulations.`;
    } catch (error) {
      progress.className = "status-message status-error";
      progress.textContent = `Experiment failed: ${error.message}`;
    } finally {
      runButton.disabled = false;
    }
  }

  aggregatedRows() {
    const groups = new Map();
    for (const row of this.rows) {
      const key = `${row.dimensions}:${row.rule}`;
      const group = groups.get(key) || { rule: row.rule, dimensions: row.dimensions, rows: [] };
      group.rows.push(row);
      groups.set(key, group);
    }
    return Array.from(groups.values()).map((group) => {
      const count = group.rows.length;
      const mean = (key) => group.rows.reduce((sum, row) => sum + Number(row[key] || 0), 0) / count;
      return {
        ...group,
        runs: count,
        final: mean("final_alive_fraction"),
        changed: mean("mean_changed_fraction"),
        fixed: group.rows.filter((row) => row.fixed_point).length,
      };
    });
  }

  renderTable() {
    const body = byId("experiment-table").querySelector("tbody");
    const filter = byId("experiment-filter").value.trim().toLowerCase();
    const rows = this.aggregatedRows()
      .filter((row) => !filter || row.rule.toLowerCase().includes(filter))
      .sort((left, right) => {
        const a = this.sortKey === "rule" ? left.rule : left[this.sortKey];
        const b = this.sortKey === "rule" ? right.rule : right[this.sortKey];
        const comparison = typeof a === "string" ? a.localeCompare(b) : a - b;
        return this.sortDescending ? -comparison : comparison;
      });
    body.replaceChildren();
    if (!rows.length) {
      const empty = document.createElement("tr");
      empty.innerHTML = '<td colspan="6" class="empty-table">No matching results.</td>';
      body.appendChild(empty);
      return;
    }
    for (const row of rows) {
      const tableRow = document.createElement("tr");
      tableRow.innerHTML = `<td>${row.dimensions}D ${row.rule}</td><td>${row.runs}</td><td>${(row.final * 100).toFixed(2)}%</td><td>${(row.changed * 100).toFixed(2)}%</td><td>${row.fixed} / ${row.runs}</td><td><button class="chip-button" data-open-dimension="${row.dimensions}" data-open-rule="${row.rule}" type="button">Open</button></td>`;
      body.appendChild(tableRow);
    }
    body.querySelectorAll("[data-open-rule]").forEach((button) => {
      button.addEventListener("click", () => {
        window.dispatchEvent(new CustomEvent("open-rule", { detail: { dimensions: button.dataset.openDimension, rule: button.dataset.openRule } }));
      });
    });
  }

  exportCsv() {
    if (!this.rows.length) return;
    const headers = Object.keys(this.rows[0]);
    const lines = [headers.join(","), ...this.rows.map((row) => headers.map((header) => csvValue(row[header])).join(","))];
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "emergent-experiment-results.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  bindControls() {
    byId("experiment-run").addEventListener("click", () => this.run());
    byId("experiment-export").addEventListener("click", () => this.exportCsv());
    byId("experiment-filter").addEventListener("input", () => this.renderTable());
    document.querySelectorAll("#experiment-table th[data-sort]").forEach((header) => {
      header.addEventListener("click", () => {
        const key = header.dataset.sort;
        if (this.sortKey === key) this.sortDescending = !this.sortDescending;
        else {
          this.sortKey = key;
          this.sortDescending = key !== "rule";
        }
        this.renderTable();
      });
    });
  }
}
