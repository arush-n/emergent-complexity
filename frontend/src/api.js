const API_ROOT = "/api";

async function request(path, options = {}) {
  const response = await fetch(`${API_ROOT}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof body === "object" && body !== null ? body.detail : body;
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return body;
}

async function download(path, fallbackFilename) {
  const response = await fetch(`${API_ROOT}${path}`);
  if (!response.ok) {
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await response.json() : await response.text();
    const detail = typeof body === "object" && body !== null ? body.detail : body;
    throw new Error(detail || `Download failed with status ${response.status}`);
  }
  const disposition = response.headers.get("content-disposition") || "";
  const filenameMatch = disposition.match(/filename="([^"]+)"/i);
  return {
    blob: await response.blob(),
    filename: filenameMatch?.[1] || fallbackFilename,
  };
}

function sessionQuery(sessionId) {
  return `?session_id=${encodeURIComponent(sessionId)}`;
}

export const api = {
  health() {
    return request("/health");
  },
  createSession(config) {
    return request("/session", { method: "POST", body: JSON.stringify(config) });
  },
  state(sessionId) {
    return request(`/state${sessionQuery(sessionId)}`);
  },
  reset(sessionId) {
    return request("/reset", { method: "POST", body: JSON.stringify({ session_id: sessionId }) });
  },
  clear(sessionId) {
    return request("/clear", { method: "POST", body: JSON.stringify({ session_id: sessionId }) });
  },
  step(sessionId, steps = 1, collectMetrics = false) {
    return request("/step", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, steps, collect_metrics: collectMetrics }),
    });
  },
  randomize(sessionId, seed, density) {
    return request("/randomize", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, seed, density }),
    });
  },
  setRule(sessionId, rule) {
    return request("/rule", { method: "POST", body: JSON.stringify({ session_id: sessionId, rule }) });
  },
  runUntilStable(sessionId, maxSteps = 100) {
    return request("/run-until-stable", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, max_steps: maxSteps }),
    });
  },
  randomRule(sessionId, seed = null) {
    return request("/random-rule", { method: "POST", body: JSON.stringify({ session_id: sessionId, seed }) });
  },
  updateState(sessionId, grid, generation = 0) {
    return request("/state", {
      method: "PUT",
      body: JSON.stringify({ session_id: sessionId, grid, generation }),
    });
  },
  setSpeed(sessionId, speed) {
    return request("/speed", { method: "POST", body: JSON.stringify({ session_id: sessionId, speed }) });
  },
  exportState(sessionId) {
    return request(`/export${sessionQuery(sessionId)}`);
  },
  createSession3d(config) {
    return request("/3d/session", { method: "POST", body: JSON.stringify(config) });
  },
  state3d(sessionId, maxVoxels = 75000) {
    return request(`/3d/state${sessionQuery(sessionId)}&max_voxels=${maxVoxels}`);
  },
  reset3d(sessionId) {
    return request("/3d/reset", { method: "POST", body: JSON.stringify({ session_id: sessionId }) });
  },
  clear3d(sessionId) {
    return request("/3d/clear", { method: "POST", body: JSON.stringify({ session_id: sessionId }) });
  },
  step3d(sessionId, steps = 1, collectMetrics = false) {
    return request("/3d/step", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, steps, collect_metrics: collectMetrics }),
    });
  },
  randomize3d(sessionId, seed, density) {
    return request("/3d/randomize", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, seed, density }),
    });
  },
  setRule3d(sessionId, rule) {
    return request("/3d/rule", { method: "POST", body: JSON.stringify({ session_id: sessionId, rule }) });
  },
  randomRule3d(sessionId, seed = null) {
    return request("/3d/random-rule", { method: "POST", body: JSON.stringify({ session_id: sessionId, seed }) });
  },
  runUntilStable3d(sessionId, maxSteps = 100) {
    return request("/3d/run-until-stable", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, max_steps: maxSteps }),
    });
  },
  setSpeed3d(sessionId, speed) {
    return request("/3d/speed", { method: "POST", body: JSON.stringify({ session_id: sessionId, speed }) });
  },
  slice3d(sessionId, axis, index) {
    return request(`/3d/slice?session_id=${encodeURIComponent(sessionId)}&axis=${axis}&index=${index}`);
  },
  exportState3d(sessionId) {
    return download(`/3d/export${sessionQuery(sessionId)}`, `life-lab-3d-state.npz`);
  },
  exportView3d(sessionId, maxVoxels = 75000) {
    return request(`/3d/export-view${sessionQuery(sessionId)}&max_voxels=${maxVoxels}`);
  },
  experiment3d(config) {
    return request("/experiments/3d", { method: "POST", body: JSON.stringify(config) });
  },
  experiment2d(config) {
    return request("/experiments/2d", { method: "POST", body: JSON.stringify(config) });
  },
  experimentLimits() {
    return request("/experiments/limits");
  },
};
