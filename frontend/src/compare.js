import { api } from "./api.js";
import { GridCanvas } from "./grid.js";
import { ThreeVoxelView } from "./three_view.js";

const byId = (id) => document.getElementById(id);

function numberValue(id, fallback) {
  const value = Number(byId(id).value);
  return Number.isFinite(value) ? value : fallback;
}

export class CompareController {
  constructor() {
    this.session2d = "compare-2d";
    this.session3d = "compare-3d";
    this.state2d = null;
    this.state3d = null;
    this.initialized = false;
    this.active = false;
    this.playing = false;
    this.timer = null;
    this.playbackToken = 0;
    this.playbackLastTime = 0;
    this.playbackBudget = 0;
    this.inFlight = false;
    this.view = new ThreeVoxelView(byId("compare-3d-viewport"), { maxVoxels: 50000 });
    this.grid = new GridCanvas(byId("compare-2d-canvas"), {
      onCommit: (grid) => {
        if (!this.state2d) return;
        api.updateState(this.session2d, grid, this.state2d.generation).then((state) => this.render2d(state)).catch((error) => this.setStatus(error.message, true));
      },
    });
    this.bindControls();
  }

  setStatus(message, isError = false) {
    const element = byId("compare-status");
    element.textContent = message;
    element.classList.toggle("status-error", isError);
    element.classList.toggle("status-success", !isError);
  }

  async activate() {
    this.active = true;
    if (this.initialized) {
      this.view.start();
      this.view.resize();
      return;
    }
    this.setStatus("Preparing comparison...");
    try {
      await this.view.init();
      this.view.start();
      await this.createBoth();
      this.initialized = true;
      this.setStatus("Ready - shared seed and density");
    } catch (error) {
      this.setStatus(`Comparison unavailable: ${error.message}`, true);
    }
  }

  deactivate() {
    this.active = false;
    this.stop();
    this.view.stop();
  }

  async createBoth() {
    const rule2d = byId("compare-rule-2d").value.trim();
    const rule3d = byId("compare-rule-3d").value.trim();
    const density = numberValue("compare-density", 0.1);
    const seed = Math.trunc(numberValue("compare-seed", 42));
    const size2d = Math.trunc(numberValue("compare-size-2d", 64));
    const size3d = Math.trunc(numberValue("compare-size-3d", 32));
    const [state2d, state3d] = await Promise.all([
      api.createSession({ session_id: this.session2d, width: size2d, height: size2d, density, seed, rule: rule2d }),
      api.createSession3d({ session_id: this.session3d, depth: size3d, height: size3d, width: size3d, density, seed, rule: rule3d }),
    ]);
    this.render2d(state2d);
    this.render3d(state3d);
  }

  render2d(state) {
    this.state2d = state;
    this.grid.setState(state.grid, state.width, state.height);
    byId("compare-2d-meta").textContent = state.rule;
    byId("compare-2d-stats").textContent = `Generation ${state.generation} | Alive ${Number(state.alive).toLocaleString()} (${(state.alive_fraction * 100).toFixed(2)}%) | Changed ${(state.changed_fraction * 100).toFixed(2)}%`;
  }

  render3d(state) {
    this.state3d = state;
    this.view.setVoxels(state.voxels || [], state.grid_shape, Boolean(state.render_sampled));
    byId("compare-3d-meta").textContent = state.rule;
    byId("compare-3d-stats").textContent = `Generation ${state.generation} | Alive ${Number(state.alive).toLocaleString()} (${(state.alive_fraction * 100).toFixed(2)}%) | Changed ${(state.changed_fraction * 100).toFixed(2)}%`;
  }

  async stepBoth(steps = 1, announce = true) {
    if (this.inFlight) return false;
    this.inFlight = true;
    try {
      const [state2d, state3d] = await Promise.all([
        api.step(this.session2d, steps, steps === 1),
        api.step3d(this.session3d, steps, steps === 1),
      ]);
      this.render2d(state2d);
      this.render3d(state3d);
      if (announce) this.setStatus(steps === 1 ? "Advanced both systems one generation" : `Advanced both systems ${steps} generations`);
      return true;
    } catch (error) {
      this.setStatus(error.message, true);
      return false;
    } finally {
      this.inFlight = false;
    }
  }

  start() {
    if (this.playing) return;
    this.playing = true;
    this.playbackToken += 1;
    this.playbackLastTime = performance.now();
    this.playbackBudget = 0;
    const token = this.playbackToken;
    byId("compare-play").disabled = true;
    byId("compare-pause").disabled = false;
    this.setStatus("Playing both systems");
    this.playbackTick(token);
  }

  async playbackTick(token) {
    if (!this.playing || !this.active || token !== this.playbackToken) return;
    const now = performance.now();
    const elapsed = this.playbackLastTime ? Math.min(now - this.playbackLastTime, 250) : 0;
    this.playbackLastTime = now;
    this.playbackBudget += (elapsed / 1000) * Math.max(numberValue("compare-speed", 10), 1);
    const steps = Math.min(Math.floor(this.playbackBudget), 800);
    if (steps > 0) {
      this.playbackBudget -= steps;
      const completed = await this.stepBoth(steps, false);
      if (!completed) {
        this.stop();
        return;
      }
    }
    if (this.playing && this.active && token === this.playbackToken) {
      this.timer = window.setTimeout(() => this.playbackTick(token), 33);
    }
  }

  stop() {
    this.playing = false;
    this.playbackToken += 1;
    this.playbackLastTime = 0;
    this.playbackBudget = 0;
    if (this.timer) window.clearTimeout(this.timer);
    this.timer = null;
    byId("compare-play").disabled = false;
    byId("compare-pause").disabled = true;
  }

  bindControls() {
    byId("compare-step").addEventListener("click", () => this.stepBoth());
    byId("compare-play").addEventListener("click", () => this.start());
    byId("compare-pause").addEventListener("click", () => this.stop());
    byId("compare-randomize").addEventListener("click", async () => {
      this.stop();
      try {
        const size2d = Math.trunc(numberValue("compare-size-2d", 64));
        const size3d = Math.trunc(numberValue("compare-size-3d", 32));
        if (!this.state2d || this.state2d.width !== size2d || !this.state3d || this.state3d.width !== size3d) await this.createBoth();
        else {
          const seed = Math.trunc(numberValue("compare-seed", 42));
          const density = numberValue("compare-density", 0.1);
          const [state2d, state3d] = await Promise.all([
            api.randomize(this.session2d, seed, density),
            api.randomize3d(this.session3d, seed, density),
          ]);
          this.render2d(state2d);
          this.render3d(state3d);
        }
        this.setStatus("Generated both states from the same seed");
      } catch (error) {
        this.setStatus(error.message, true);
      }
    });
    byId("compare-reset").addEventListener("click", async () => {
      this.stop();
      try {
        const [state2d, state3d] = await Promise.all([api.reset(this.session2d), api.reset3d(this.session3d)]);
        this.render2d(state2d);
        this.render3d(state3d);
        this.setStatus("Restored both initial states");
      } catch (error) {
        this.setStatus(error.message, true);
      }
    });
    byId("compare-rule-2d").addEventListener("change", async (event) => {
      try { this.render2d(await api.setRule(this.session2d, event.target.value)); } catch (error) { this.setStatus(error.message, true); }
    });
    byId("compare-rule-3d").addEventListener("change", async (event) => {
      try { this.render3d(await api.setRule3d(this.session3d, event.target.value)); } catch (error) { this.setStatus(error.message, true); }
    });
    byId("compare-speed").addEventListener("input", (event) => { byId("compare-speed-value").textContent = `${Number(event.target.value).toFixed(0)} gen/s`; });
    byId("compare-pause").disabled = true;
  }
}
