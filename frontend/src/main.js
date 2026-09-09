import { api } from "./api.js";
import { renderMetricsChart } from "./chart.js";
import { createRuleEditor } from "./controls.js";
import { GridCanvas } from "./grid.js";
import { CompareController } from "./compare.js";
import { ExperimentController } from "./experiments.js";
import { ThreeLabController } from "./lab3d.js";

const byId = (id) => document.getElementById(id);
const canvas = byId("ca-canvas");
const statusMessage = byId("status-message");
const widthInput = byId("width-input");
const heightInput = byId("height-input");
const densityInput = byId("density-input");
const densityValue = byId("density-value");
const seedInput = byId("seed-input");
const speedInput = byId("speed-input");
const speedValue = byId("speed-value");
const stabilityLimitInput = byId("stability-limit");
const ruleText = byId("rule-text");
const ruleDisplay = byId("rule-display");
const generationValue = byId("generation-value");
const metadataRule = byId("metadata-rule");
const metadataGrid = byId("metadata-grid");
const metadataGeneration = byId("metadata-generation");
const metadataAlive = byId("metadata-alive");
const metadataChanged = byId("metadata-changed");
const metadataBirths = byId("metadata-births");
const metadataDeaths = byId("metadata-deaths");
const metadataDensity = byId("metadata-density");
const metricsChart = byId("metrics-chart");

let currentState = null;
let sessionId = "default";
let playing = false;
let playbackToken = 0;
let editQueue = Promise.resolve();
let randomRuleCounter = 0;
let metricHistory = [];

const MAX_METRIC_POINTS = 180;
const PLAYBACK_INTERVAL_MS = 33;
const MAX_PLAYBACK_BATCH = 2_000;
let playbackLastTime = 0;
let playbackBudget = 0;
let activeMode = "2d";

const threeLab = new ThreeLabController();
const compareLab = new CompareController();
const experimentLab = new ExperimentController();
const controlsToggle = byId("controls-toggle");

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle("status-error", isError);
  statusMessage.classList.toggle("status-success", !isError);
}

controlsToggle?.addEventListener("click", () => {
  const collapsed = document.body.classList.toggle("controls-collapsed");
  controlsToggle.setAttribute("aria-expanded", String(!collapsed));
});

function numberValue(input, fallback) {
  const value = Number(input.value);
  return Number.isFinite(value) ? value : fallback;
}

function setInputsFromState(state) {
  densityInput.value = String(state.density ?? 0.2);
  densityValue.textContent = Number(state.density ?? 0.2).toFixed(2);
  seedInput.value = state.seed ?? 42;
  speedInput.value = String(state.speed ?? 10);
  speedValue.textContent = `${Number(state.speed ?? 10).toFixed(0)} gen/s`;
}

const gridCanvas = new GridCanvas(canvas, {
  onCommit(grid) {
    if (!currentState) return;
    const generation = currentState.generation;
    editQueue = editQueue
      .then(() => api.updateState(sessionId, grid, generation))
      .then((state) => renderState(state))
      .catch((error) => setStatus(`Could not save drawing: ${error.message}`, true));
  },
});

const ruleEditor = createRuleEditor(
  byId("birth-toggles"),
  byId("survival-toggles"),
  (rule) => {
    ruleText.value = rule;
    applyRule(rule);
  },
);

function appendMetricPoint(state, sample) {
  const totalCells = state.width * state.height;
  const alive = Number(sample.alive ?? state.alive ?? 0);
  const changed = Number(sample.changed_cells ?? state.changed_cells ?? 0);
  const births = Number(sample.births ?? state.births ?? 0);
  const deaths = Number(sample.deaths ?? state.deaths ?? 0);
  const point = {
    session_id: state.session_id,
    generation: Number(sample.generation ?? state.generation),
    width: state.width,
    height: state.height,
    alive_fraction: Number(sample.alive_fraction ?? alive / totalCells),
    changed_cells: changed,
    changed_fraction: Number(sample.changed_fraction ?? changed / totalCells),
    births,
    birth_fraction: Number(sample.birth_fraction ?? births / totalCells),
    deaths,
    death_fraction: Number(sample.death_fraction ?? deaths / totalCells),
  };
  const previous = metricHistory.at(-1);
  const sessionChanged = previous && point.session_id !== previous.session_id;
  const dimensionsChanged =
    previous && (previous.width !== point.width || previous.height !== point.height);
  const rewound = previous && point.generation < previous.generation;
  if (sessionChanged || dimensionsChanged || rewound) metricHistory = [];

  const lastPoint = metricHistory.at(-1);
  if (lastPoint?.generation === point.generation) metricHistory[metricHistory.length - 1] = point;
  else metricHistory.push(point);
}

function updateMetricHistory(state) {
  const metrics = Array.isArray(state.metrics) ? state.metrics : [];
  for (const sample of metrics) appendMetricPoint(state, sample);
  appendMetricPoint(state, state);
  if (metricHistory.length > MAX_METRIC_POINTS) metricHistory = metricHistory.slice(-MAX_METRIC_POINTS);
  renderMetricsChart(metricsChart, metricHistory, state.width * state.height);
}

function renderState(state) {
  currentState = state;
  sessionId = state.session_id || sessionId;
  gridCanvas.setState(state.grid, state.width, state.height);
  ruleEditor.setRule(state.rule);
  ruleText.value = state.rule;
  ruleDisplay.textContent = state.rule;
  generationValue.textContent = String(state.generation);
  metadataRule.textContent = state.rule;
  metadataGrid.textContent = `${state.width} x ${state.height}`;
  metadataGeneration.textContent = String(state.generation);
  metadataAlive.textContent = Number(state.alive).toLocaleString();
  metadataChanged.textContent = Number(state.changed_cells || 0).toLocaleString();
  metadataBirths.textContent = Number(state.births || 0).toLocaleString();
  metadataDeaths.textContent = Number(state.deaths || 0).toLocaleString();
  metadataDensity.textContent = Number(state.alive_fraction).toFixed(4);
  updateMetricHistory(state);
  setInputsFromState(state);
}

async function applyRule(rule) {
  if (!currentState) return;
  try {
    const state = await api.setRule(sessionId, rule);
    renderState(state);
    setStatus(`Using ${state.rule}`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function perform(action, successMessage = "Ready") {
  try {
    const state = await action();
    renderState(state);
    setStatus(successMessage);
  } catch (error) {
    setStatus(error.message, true);
  }
}

function pause() {
  const wasPlaying = playing;
  playing = false;
  playbackToken += 1;
  playbackLastTime = 0;
  playbackBudget = 0;
  byId("play-button").disabled = false;
  byId("pause-button").disabled = true;
  if (wasPlaying) setStatus("Paused");
}

async function playbackTick(token) {
  if (!playing || token !== playbackToken) return;
  const now = performance.now();
  const elapsed = playbackLastTime ? Math.min(now - playbackLastTime, 250) : 0;
  playbackLastTime = now;
  playbackBudget += (elapsed / 1000) * Math.max(numberValue(speedInput, 10), 1);
  const steps = Math.min(Math.floor(playbackBudget), MAX_PLAYBACK_BATCH);
  try {
    if (steps > 0) {
      playbackBudget -= steps;
      const collectMetrics = steps === 1;
      const state = await api.step(sessionId, steps, collectMetrics);
      if (playing && token === playbackToken) renderState(state);
    }
  } catch (error) {
    pause();
    setStatus(error.message, true);
    return;
  }
  if (playing && token === playbackToken) {
    window.setTimeout(() => playbackTick(token), PLAYBACK_INTERVAL_MS);
  }
}

function play() {
  if (playing) return;
  playing = true;
  playbackToken += 1;
  playbackLastTime = performance.now();
  playbackBudget = 0;
  const token = playbackToken;
  byId("play-button").disabled = true;
  byId("pause-button").disabled = false;
  setStatus("Playing");
  playbackTick(token);
}

function downloadJson(state) {
  const blob = new Blob([JSON.stringify(state, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `ca-${state.rule.replaceAll("/", "-")}-gen-${state.generation}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function capture2D() {
  if (!currentState) return;
  const source = canvas;
  const output = document.createElement("canvas");
  output.width = source.width;
  output.height = source.height + 42;
  const context = output.getContext("2d");
  context.fillStyle = "#050505";
  context.fillRect(0, 0, output.width, output.height);
  context.drawImage(source, 0, 0);
  context.fillStyle = "#111110";
  context.fillRect(0, source.height, output.width, 42);
  context.fillStyle = "#eeede5";
  context.font = "14px monospace";
  context.fillText(`2D | ${currentState.rule} | Generation ${currentState.generation} | ${currentState.width} x ${currentState.height} | Seed ${currentState.seed}`, 14, source.height + 26);
  const link = document.createElement("a");
  link.download = `life-lab-2d-gen-${currentState.generation}.png`;
  link.href = output.toDataURL("image/png");
  link.click();
}

async function importJson(file) {
  const imported = JSON.parse(await file.text());
  if (!Array.isArray(imported.grid) || !imported.grid.length || !Array.isArray(imported.grid[0])) {
    throw new Error("The file does not contain a valid grid.");
  }
  const height = imported.height || imported.grid.length;
  const width = imported.width || imported.grid[0].length;
  pause();
  const created = await api.createSession({
    session_id: sessionId,
    width,
    height,
    density: imported.density ?? 0.2,
    seed: imported.seed ?? 42,
    rule: imported.rule || "B3/S23",
  });
  const loaded = await api.updateState(sessionId, imported.grid, imported.generation || 0);
  renderState(loaded || created);
  setStatus("Loaded saved state");
}

byId("play-button").addEventListener("click", play);
byId("pause-button").addEventListener("click", pause);
byId("pause-button").disabled = true;
byId("step-button").addEventListener("click", () => {
  pause();
  perform(() => api.step(sessionId), "Advanced one generation");
});
byId("reset-button").addEventListener("click", () => {
  pause();
  perform(() => api.reset(sessionId), "Restored initial state");
});
byId("clear-button").addEventListener("click", () => {
  pause();
  perform(() => api.clear(sessionId), "Cleared current grid");
});
byId("stabilize-button").addEventListener("click", async () => {
  pause();
  try {
    const maxSteps = Math.trunc(numberValue(stabilityLimitInput, 10000));
    const state = await api.runUntilStable(sessionId, maxSteps);
    renderState(state);
    const result = state.settled ? "Fixed point reached" : "Stopped at the safety limit";
    setStatus(`${result} after ${state.steps_run} generations`);
  } catch (error) {
    setStatus(error.message, true);
  }
});
byId("randomize-button").addEventListener("click", () => {
  pause();
  perform(
    () => api.randomize(sessionId, Math.trunc(numberValue(seedInput, 42)), numberValue(densityInput, 0.2)),
    "Generated reproducible initial state",
  );
});
byId("random-rule-button").addEventListener("click", () => {
  const seed = Math.trunc(numberValue(seedInput, 42)) + randomRuleCounter;
  randomRuleCounter += 1;
  perform(() => api.randomRule(sessionId, seed), "Selected reproducible random rule");
});
document.querySelectorAll("[data-rule-preset]").forEach((button) => {
  button.addEventListener("click", () => applyRule(button.dataset.rulePreset));
});
byId("apply-rule-button").addEventListener("click", () => applyRule(ruleText.value));
byId("edit-mode").addEventListener("change", (event) => gridCanvas.setMode(event.target.value));
byId("grid-lines").addEventListener("change", (event) => gridCanvas.setShowGrid(event.target.checked));
byId("capture-button").addEventListener("click", capture2D);
byId("density-input").addEventListener("input", (event) => {
  densityValue.textContent = Number(event.target.value).toFixed(2);
});
byId("speed-input").addEventListener("input", (event) => {
  speedValue.textContent = `${Number(event.target.value).toFixed(0)} gen/s`;
  if (currentState) api.setSpeed(sessionId, Number(event.target.value)).catch(() => {});
});
byId("apply-dimensions-button").addEventListener("click", () => {
  pause();
  const width = Math.trunc(numberValue(widthInput, 128));
  const height = Math.trunc(numberValue(heightInput, 128));
  perform(
    () => api.createSession({
      session_id: sessionId,
      width,
      height,
      density: numberValue(densityInput, 0.2),
      seed: Math.trunc(numberValue(seedInput, 42)),
      rule: currentState?.rule || "B3/S23",
    }),
    `Created ${width} x ${height} world`,
  );
});
byId("save-button").addEventListener("click", async () => {
  try {
    const exported = await api.exportState(sessionId);
    downloadJson(exported);
    setStatus("Saved JSON export");
  } catch (error) {
    setStatus(error.message, true);
  }
});
byId("load-button").addEventListener("click", () => byId("load-input").click());
byId("load-input").addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    await importJson(file);
  } catch (error) {
    setStatus(`Could not load file: ${error.message}`, true);
  } finally {
    event.target.value = "";
  }
});

async function boot() {
  try {
    const state = await api.createSession({
      session_id: "default",
      width: 128,
      height: 128,
      density: 0,
      seed: 42,
      rule: "B3/S23",
    });
    widthInput.value = state.width;
    heightInput.value = state.height;
    // Start with a recognizable glider while keeping the regular randomize
    // controls fully deterministic for research runs.
    const glider = state.grid.map((row) => row.slice());
    [[1, 2], [2, 3], [3, 1], [3, 2], [3, 3]].forEach(([row, column]) => { glider[row + 1][column + 1] = 1; });
    const initial = await api.updateState("default", glider, 0);
    renderState(initial);
    setStatus("Ready - draw cells or press Randomize");
    api.health().then((health) => {
      byId("device-chip").innerHTML = `<span class="status-dot"></span>${health.jax_device}`;
    }).catch(() => {});
  } catch (error) {
    setStatus(`Backend unavailable: ${error.message}`, true);
  }
}

boot();

function setMode(mode) {
  activeMode = mode;
  document.querySelectorAll(".mode-link").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  document.querySelectorAll(".mode-panel").forEach((panel) => {
    panel.classList.toggle("hidden", panel.id !== `mode-${mode}`);
  });
  if (mode === "2d") gridCanvas.render();
  if (mode === "3d") threeLab.activate();
  else threeLab.deactivate();
  if (mode === "compare") compareLab.activate();
  else compareLab.deactivate();
  if (mode === "experiments") experimentLab.activate();
  const query = new URLSearchParams(window.location.search);
  query.set("mode", mode);
  window.history.replaceState({}, "", `?${query.toString()}`);
}

document.querySelectorAll(".mode-link").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

const requestedMode = new URLSearchParams(window.location.search).get("mode");
if (["2d", "3d", "compare", "experiments"].includes(requestedMode)) setMode(requestedMode);

window.addEventListener("open-rule", async (event) => {
  const { dimensions, rule } = event.detail;
  setMode(dimensions === "3" ? "3d" : "2d");
  if (dimensions === "3") {
    await threeLab.activate();
    await threeLab.applyRule(rule);
  } else {
    await applyRule(rule);
  }
});

window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select, textarea")) return;
  if (event.code === "Space") {
    event.preventDefault();
    if (activeMode === "2d") (playing ? pause : play)();
    else if (activeMode === "3d") (threeLab.playing ? threeLab.pause() : threeLab.play());
  } else if (event.code === "ArrowRight") {
    event.preventDefault();
    if (activeMode === "2d") {
      pause();
      perform(() => api.step(sessionId), "Advanced one generation");
    } else if (activeMode === "3d") {
      threeLab.pause();
      threeLab.perform(() => api.step3d(threeLab.sessionId, 1, true), "Advanced one generation");
    }
  } else if (event.key.toLowerCase() === "r") {
    if (activeMode === "2d") {
      pause();
      perform(() => api.reset(sessionId), "Restored initial state");
    } else if (activeMode === "3d") {
      threeLab.pause();
      threeLab.perform(() => api.reset3d(threeLab.sessionId), "Restored initial state");
    }
  } else if (event.key.toLowerCase() === "c") {
    if (activeMode === "2d") {
      pause();
      perform(() => api.clear(sessionId), "Cleared current grid");
    } else if (activeMode === "3d") {
      threeLab.pause();
      threeLab.perform(() => api.clear3d(threeLab.sessionId), "Cleared current grid");
    }
  }
});
