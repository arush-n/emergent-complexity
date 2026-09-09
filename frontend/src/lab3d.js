import { api } from "./api.js";
import { renderMetricsChart } from "./chart.js";
import { createRuleEditor } from "./controls.js";
import { renderSlice } from "./slice.js";
import { ThreeVoxelView } from "./three_view.js";

const byId = (id) => document.getElementById(id);

function numberValue(input, fallback) {
  const value = Number(input.value);
  return Number.isFinite(value) ? value : fallback;
}

export class ThreeLabController {
  constructor() {
    this.sessionId = "3d-default";
    this.currentState = null;
    this.active = false;
    this.initialized = false;
    this.playing = false;
    this.playbackToken = 0;
    this.playbackLastTime = 0;
    this.playbackBudget = 0;
    this.randomRuleCounter = 0;
    this.metricHistory = [];
    this.sliceToken = 0;
    this.MAX_METRIC_POINTS = 180;
    this.view = new ThreeVoxelView(byId("viewport-3d"));

    this.ruleEditor = createRuleEditor(
      byId("3d-birth-toggles"),
      byId("3d-survival-toggles"),
      (rule) => {
        byId("3d-rule-text").value = rule;
        this.applyRule(rule);
      },
      { maxCount: 26, separator: "," },
    );
    this.bindControls();
  }

  setStatus(message, isError = false) {
    const status = byId("3d-status-message");
    status.textContent = message;
    status.classList.toggle("status-error", isError);
    status.classList.toggle("status-success", !isError);
  }

  async activate() {
    this.active = true;
    if (this.initialized) {
      this.view.start();
      this.view.resize();
      return;
    }
    this.setStatus("Preparing 3D world...");
    try {
      await this.view.init();
      this.view.start();
      const config = this.urlConfig();
      const state = await api.createSession3d({
        session_id: this.sessionId,
        depth: config.depth,
        height: config.height,
        width: config.width,
        density: config.density,
        seed: config.seed,
        rule: config.rule,
      });
      this.initialized = true;
      this.renderState(state);
      this.setStatus("Ready - deterministic 3D seed 42");
    } catch (error) {
      this.setStatus(`3D backend unavailable: ${error.message}`, true);
    }
  }

  deactivate() {
    this.active = false;
    this.pause(false);
    this.view.stop();
  }

  urlConfig() {
    const params = new URLSearchParams(window.location.search);
    // A 2D session may leave width/height/density in the URL. Require the
    // 3D-specific depth field before treating those values as a 3D setup.
    const has3dConfig = params.get("mode") === "3d" && params.has("depth");
    const fallbackSize = Math.trunc(Number(has3dConfig ? params.get("size") : 0) || 32);
    const dimension = (name) => Math.max(
      8,
      Math.min(128, Math.trunc(Number(has3dConfig ? params.get(name) : 0) || fallbackSize)),
    );
    const densityValue = Number(has3dConfig ? params.get("density") : NaN);
    const density = Math.max(0, Math.min(1, Number.isFinite(densityValue) ? densityValue : 0.04));
    const seed = Math.trunc(Number(has3dConfig ? params.get("seed") : 0) || 42);
    const rule = (has3dConfig ? params.get("rule") : null) || "B6/S5,6,7";
    return { depth: dimension("depth"), height: dimension("height"), width: dimension("width"), density, seed, rule };
  }

  appendMetricPoint(state, sample) {
    const totalCells = state.width * state.height * state.depth;
    const alive = Number(sample.alive ?? state.alive ?? 0);
    const changed = Number(sample.changed_cells ?? state.changed_cells ?? 0);
    const births = Number(sample.births ?? state.births ?? 0);
    const deaths = Number(sample.deaths ?? state.deaths ?? 0);
    const point = {
      session_id: state.session_id,
      generation: Number(sample.generation ?? state.generation),
      width: state.width,
      height: state.height,
      depth: state.depth,
      alive_fraction: Number(sample.alive_fraction ?? alive / totalCells),
      changed_cells: changed,
      changed_fraction: Number(sample.changed_fraction ?? changed / totalCells),
      births,
      birth_fraction: Number(sample.birth_fraction ?? births / totalCells),
      deaths,
      death_fraction: Number(sample.death_fraction ?? deaths / totalCells),
    };
    const previous = this.metricHistory.at(-1);
    if (
      previous &&
      (previous.session_id !== point.session_id ||
        previous.width !== point.width ||
        previous.height !== point.height ||
        previous.depth !== point.depth ||
        point.generation < previous.generation)
    ) {
      this.metricHistory = [];
    }
    const last = this.metricHistory.at(-1);
    if (last?.generation === point.generation) this.metricHistory[this.metricHistory.length - 1] = point;
    else this.metricHistory.push(point);
  }

  updateMetricHistory(state) {
    const metrics = Array.isArray(state.metrics) ? state.metrics : [];
    metrics.forEach((sample) => this.appendMetricPoint(state, sample));
    this.appendMetricPoint(state, state);
    if (this.metricHistory.length > this.MAX_METRIC_POINTS) {
      this.metricHistory = this.metricHistory.slice(-this.MAX_METRIC_POINTS);
    }
    renderMetricsChart(byId("metrics-chart-3d"), this.metricHistory, state.width * state.height * state.depth);
  }

  renderState(state) {
    this.currentState = state;
    this.sessionId = state.session_id || this.sessionId;
    this.view.setVoxels(state.voxels || [], state.grid_shape, Boolean(state.render_sampled));
    this.view.setBoundsVisible(byId("3d-bounds-input").checked);
    byId("3d-render-warning").classList.toggle("hidden", !state.render_sampled);
    byId("3d-rule-display").textContent = state.rule;
    byId("3d-rule-text").value = state.rule;
    this.ruleEditor.setRule(state.rule);
    byId("3d-generation-value").textContent = String(state.generation);
    byId("3d-metadata-rule").textContent = state.rule;
    byId("3d-metadata-grid").textContent = `${state.depth} x ${state.height} x ${state.width}`;
    byId("3d-metadata-generation").textContent = String(state.generation);
    byId("3d-metadata-alive").textContent = Number(state.alive).toLocaleString();
    byId("3d-metadata-changed").textContent = Number(state.changed_cells || 0).toLocaleString();
    byId("3d-metadata-births").textContent = Number(state.births || 0).toLocaleString();
    byId("3d-metadata-deaths").textContent = Number(state.deaths || 0).toLocaleString();
    byId("3d-metadata-density").textContent = Number(state.alive_fraction).toFixed(4);
    byId("3d-depth-input").value = state.depth;
    byId("3d-height-input").value = state.height;
    byId("3d-width-input").value = state.width;
    byId("3d-density-input").value = String(state.density ?? 0.04);
    byId("3d-density-value").textContent = Number(state.density ?? 0.04).toFixed(2);
    byId("3d-seed-input").value = state.seed ?? 42;
    byId("3d-speed-input").value = String(state.speed ?? 10);
    byId("3d-speed-value").textContent = `${Number(state.speed ?? 10).toFixed(0)} gen/s`;
    const axis = byId("3d-axis-input").value;
    const maxIndex = { z: state.depth, y: state.height, x: state.width }[axis] - 1;
    const sliceInput = byId("3d-slice-index");
    sliceInput.max = String(maxIndex);
    sliceInput.value = String(Math.min(Number(sliceInput.value), maxIndex));
    this.updateMetricHistory(state);
    this.updatePerformance(state);
    if (byId("3d-performance-input").checked) {
      window.requestAnimationFrame(() => {
        if (this.currentState === state) this.updatePerformance(state);
      });
    }
    if (byId("3d-slice-input").checked) this.loadSlice();
    const query = new URLSearchParams({
      mode: "3d",
      rule: state.rule,
      depth: String(state.depth),
      height: String(state.height),
      width: String(state.width),
      density: String(state.density),
      seed: String(state.seed ?? 42),
    });
    window.history.replaceState({}, "", `?${query.toString()}`);
  }

  updatePerformance(state) {
    const stats = this.view.getStats();
    byId("3d-performance-overlay").textContent = [
      `JAX device: ${state.jax_device || "server device"}`,
      `Grid cells: ${Number(state.depth * state.height * state.width).toLocaleString()}`,
      `Simulation: ${Number(state.simulation_ms ?? state.last_step_ms ?? 0).toFixed(2)} ms/request`,
      `Render extract: ${Number(state.render_extract_ms || 0).toFixed(2)} ms`,
      `Payload build: ${Number(state.serialization_ms || 0).toFixed(2)} ms`,
      `Rendered voxels: ${stats.renderedCount.toLocaleString()} / ${Number(state.alive || 0).toLocaleString()}${stats.sampled ? " (sampled)" : ""}`,
      `Browser render: ${stats.lastRenderMs.toFixed(2)} ms/frame`,
      `Browser FPS: ${stats.fps.toFixed(1)}`,
    ].join("\n");
  }

  async applyRule(rule) {
    if (!this.currentState) return;
    try {
      const state = await api.setRule3d(this.sessionId, rule);
      this.renderState(state);
      this.setStatus(`Using ${state.rule}`);
    } catch (error) {
      this.setStatus(error.message, true);
    }
  }

  async perform(action, successMessage = "Ready") {
    try {
      const state = await action();
      this.renderState(state);
      this.setStatus(successMessage);
    } catch (error) {
      this.setStatus(error.message, true);
    }
  }

  pause(showStatus = true) {
    const wasPlaying = this.playing;
    this.playing = false;
    this.playbackToken += 1;
    this.playbackLastTime = 0;
    this.playbackBudget = 0;
    byId("3d-play-button").disabled = false;
    byId("3d-pause-button").disabled = true;
    if (showStatus && wasPlaying) this.setStatus("Paused");
  }

  async playbackTick(token) {
    if (!this.playing || token !== this.playbackToken) return;
    const now = performance.now();
    const elapsed = this.playbackLastTime ? Math.min(now - this.playbackLastTime, 250) : 0;
    this.playbackLastTime = now;
    this.playbackBudget += (elapsed / 1000) * Math.max(numberValue(byId("3d-speed-input"), 10), 1);
    const steps = Math.min(Math.floor(this.playbackBudget), 800);
    try {
      if (steps > 0) {
        this.playbackBudget -= steps;
        const state = await api.step3d(this.sessionId, steps, steps === 1);
        if (this.playing && token === this.playbackToken) this.renderState(state);
      }
    } catch (error) {
      this.pause();
      this.setStatus(error.message, true);
      return;
    }
    if (this.playing && token === this.playbackToken) {
      window.setTimeout(() => this.playbackTick(token), 33);
    }
  }

  play() {
    if (this.playing) return;
    this.playing = true;
    this.playbackToken += 1;
    this.playbackLastTime = performance.now();
    this.playbackBudget = 0;
    const token = this.playbackToken;
    byId("3d-play-button").disabled = true;
    byId("3d-pause-button").disabled = false;
    this.setStatus("Playing");
    this.playbackTick(token);
  }

  async loadSlice() {
    if (!this.currentState) return;
    const axis = byId("3d-axis-input").value;
    const index = Math.trunc(numberValue(byId("3d-slice-index"), 0));
    const token = ++this.sliceToken;
    try {
      const result = await api.slice3d(this.sessionId, axis, index);
      if (token !== this.sliceToken) return;
      byId("3d-slice-panel").classList.remove("hidden");
      byId("3d-slice-label").textContent = `${axis.toUpperCase()} ${index} | alive ${result.alive}`;
      renderSlice(byId("3d-slice-canvas"), result.grid, true);
    } catch (error) {
      this.setStatus(error.message, true);
    }
  }

  bindControls() {
    byId("3d-play-button").addEventListener("click", () => this.play());
    byId("3d-pause-button").addEventListener("click", () => this.pause());
    byId("3d-pause-button").disabled = true;
    byId("3d-step-button").addEventListener("click", () => {
      this.pause();
      this.perform(() => api.step3d(this.sessionId, 1, true), "Advanced one generation");
    });
    byId("3d-reset-button").addEventListener("click", () => {
      this.pause();
      this.perform(() => api.reset3d(this.sessionId), "Restored initial state");
    });
    byId("3d-clear-button").addEventListener("click", () => {
      this.pause();
      this.perform(() => api.clear3d(this.sessionId), "Cleared current grid");
    });
    byId("3d-randomize-button").addEventListener("click", () => {
      this.pause();
      this.perform(
        () => api.randomize3d(this.sessionId, Math.trunc(numberValue(byId("3d-seed-input"), 42)), numberValue(byId("3d-density-input"), 0.04)),
        "Generated reproducible 3D state",
      );
    });
    byId("3d-apply-dimensions").addEventListener("click", () => {
      this.pause();
      const depth = Math.trunc(numberValue(byId("3d-depth-input"), 32));
      const height = Math.trunc(numberValue(byId("3d-height-input"), 32));
      const width = Math.trunc(numberValue(byId("3d-width-input"), 32));
      this.perform(
        () => api.createSession3d({
          session_id: this.sessionId,
          depth,
          height,
          width,
          density: numberValue(byId("3d-density-input"), 0.04),
          seed: Math.trunc(numberValue(byId("3d-seed-input"), 42)),
          rule: this.currentState?.rule || "B6/S5,6,7",
        }),
        `Created ${depth} x ${height} x ${width} world`,
      );
    });
    byId("3d-random-rule-button").addEventListener("click", () => {
      const seed = Math.trunc(numberValue(byId("3d-seed-input"), 42)) + this.randomRuleCounter;
      this.randomRuleCounter += 1;
      this.perform(() => api.randomRule3d(this.sessionId, seed), "Selected reproducible random rule");
    });
    document.querySelectorAll("[data-3d-rule-preset]").forEach((button) => {
      button.addEventListener("click", () => this.applyRule(button.dataset["3dRulePreset"]));
    });
    byId("3d-apply-rule-button").addEventListener("click", () => this.applyRule(byId("3d-rule-text").value));
    byId("3d-stabilize-button").addEventListener("click", () => {
      this.pause();
      this.perform(
        async () => {
          const state = await api.runUntilStable3d(this.sessionId, Math.trunc(numberValue(byId("3d-stability-limit"), 1000)));
          this.setStatus(state.settled ? `Fixed point reached after ${state.steps_run} generations` : `Stopped at ${state.steps_run}-generation limit`);
          return state;
        },
      );
    });
    byId("3d-speed-input").addEventListener("input", (event) => {
      byId("3d-speed-value").textContent = `${Number(event.target.value).toFixed(0)} gen/s`;
      if (this.currentState) api.setSpeed3d(this.sessionId, Number(event.target.value)).catch(() => {});
    });
    byId("3d-density-input").addEventListener("input", (event) => {
      byId("3d-density-value").textContent = Number(event.target.value).toFixed(2);
    });
    byId("3d-bounds-input").addEventListener("change", (event) => this.view.setBoundsVisible(event.target.checked));
    byId("3d-slice-input").addEventListener("change", (event) => {
      byId("3d-slice-panel").classList.toggle("hidden", !event.target.checked);
      if (event.target.checked) this.loadSlice();
    });
    byId("3d-axis-input").addEventListener("change", () => {
      if (this.currentState) {
        const axis = byId("3d-axis-input").value;
        const maximum = { z: this.currentState.depth, y: this.currentState.height, x: this.currentState.width }[axis] - 1;
        byId("3d-slice-index").max = String(maximum);
        byId("3d-slice-index").value = String(Math.min(Math.trunc(maximum / 2), maximum));
      }
      this.loadSlice();
    });
    byId("3d-slice-index").addEventListener("input", () => this.loadSlice());
    byId("3d-reset-camera").addEventListener("click", () => this.view.resetCamera());
    document.querySelectorAll("[data-camera]").forEach((button) => {
      button.addEventListener("click", () => this.view.orient(button.dataset.camera));
    });
    byId("3d-performance-input").addEventListener("change", (event) => {
      byId("3d-performance-overlay").classList.toggle("hidden", !event.target.checked);
      if (event.target.checked && this.currentState) this.updatePerformance(this.currentState);
    });
    byId("3d-capture-button").addEventListener("click", () => {
      if (!this.currentState) return;
      this.view.capture(`3D | ${this.currentState.rule} | Generation ${this.currentState.generation} | ${this.currentState.depth}^3 | Seed ${this.currentState.seed}`);
    });
    byId("3d-export-button").addEventListener("click", async () => {
      try {
        const download = await api.exportState3d(this.sessionId);
        const url = URL.createObjectURL(download.blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = download.filename;
        link.click();
        URL.revokeObjectURL(url);
        this.setStatus("Exported exact 3D state as NPZ");
      } catch (error) {
        this.setStatus(error.message, true);
      }
    });
    byId("3d-share-button").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(window.location.href);
        this.setStatus("Configuration link copied");
      } catch {
        this.setStatus("Copy the current URL to share this configuration");
      }
    });
  }
}
