const configuredBase = String(globalThis.EMERGENT_CONFIG?.apiBase || "").replace(/\/+$/, "");
const API_ROOT = `${configuredBase}/api`;

async function responseError(response, fallback) {
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : await response.text();
  const detail = typeof body === "object" && body !== null ? body.detail : body;
  return new Error(detail || `${fallback} with status ${response.status}`);
}

async function request(path, options = {}) {
  const response = await fetch(`${API_ROOT}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) throw await responseError(response, "Request failed");
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response.text();
}

async function binary(path) {
  const response = await fetch(`${API_ROOT}${path}`);
  if (!response.ok) throw await responseError(response, "Download failed");
  return response;
}

async function download(path, fallbackFilename) {
  const response = await binary(path);
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

function unpack2d(bytes, width, height) {
  const cellCount = width * height;
  if (bytes.length < Math.ceil(cellCount / 8)) throw new Error("The 2D render payload is truncated.");
  const grid = new Uint8Array(cellCount);
  for (let index = 0; index < cellCount; index += 1) {
    grid[index] = (bytes[index >> 3] >> (index & 7)) & 1;
  }
  return grid;
}

async function hydrate2d(state) {
  const response = await binary(`/render${sessionQuery(state.session_id)}`);
  const width = Number(state.width);
  const height = Number(state.height);
  const bytes = new Uint8Array(await response.arrayBuffer());
  return {
    ...state,
    grid: unpack2d(bytes, width, height),
    render_bytes: bytes.length,
  };
}

async function hydrate3d(state, maxVoxels = 75000) {
  const response = await binary(`/3d/render${sessionQuery(state.session_id)}&max_voxels=${maxVoxels}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.length % 3 !== 0) throw new Error("The 3D render payload is malformed.");
  return {
    ...state,
    grid_shape: [Number(state.depth), Number(state.height), Number(state.width)],
    voxels: bytes,
    rendered_voxels: Number(response.headers.get("x-rendered-voxels") || bytes.length / 3),
    render_sampled: response.headers.get("x-render-sampled") === "true",
    render_limit: Number(response.headers.get("x-render-limit") || maxVoxels),
    render_extract_ms: Number(response.headers.get("x-render-extract-ms") || 0),
    serialization_ms: Number(response.headers.get("x-serialization-ms") || 0),
    render_bytes: bytes.length,
  };
}

export const api = {
  health() {
    return request("/health");
  },
  createSession(config) {
    return request("/session", { method: "POST", body: JSON.stringify(config) }).then(hydrate2d);
  },
  state(sessionId) {
    return request(`/state${sessionQuery(sessionId)}`).then(hydrate2d);
  },
  reset(sessionId) {
    return request("/reset", { method: "POST", body: JSON.stringify({ session_id: sessionId }) }).then(hydrate2d);
  },
  clear(sessionId) {
    return request("/clear", { method: "POST", body: JSON.stringify({ session_id: sessionId }) }).then(hydrate2d);
  },
  step(sessionId, steps = 1, collectMetrics = false) {
    return request("/step", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, steps, collect_metrics: collectMetrics }),
    }).then(hydrate2d);
  },
  randomize(sessionId, seed, density) {
    return request("/randomize", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, seed, density }),
    }).then(hydrate2d);
  },
  setRule(sessionId, rule) {
    return request("/rule", { method: "POST", body: JSON.stringify({ session_id: sessionId, rule }) }).then(hydrate2d);
  },
  runUntilStable(sessionId, maxSteps = 100) {
    return request("/run-until-stable", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, max_steps: maxSteps }),
    }).then(hydrate2d);
  },
  randomRule(sessionId, seed = null) {
    return request("/random-rule", { method: "POST", body: JSON.stringify({ session_id: sessionId, seed }) }).then(hydrate2d);
  },
  updateState(sessionId, grid, generation = 0) {
    return request("/state", {
      method: "PUT",
      body: JSON.stringify({ session_id: sessionId, grid, generation }),
    }).then(hydrate2d);
  },
  exportState(sessionId) {
    return request(`/export${sessionQuery(sessionId)}`);
  },
  createSession3d(config) {
    return request("/3d/session", { method: "POST", body: JSON.stringify(config) }).then(hydrate3d);
  },
  state3d(sessionId, maxVoxels = 75000) {
    return request(`/3d/state${sessionQuery(sessionId)}&max_voxels=${maxVoxels}`).then((state) => hydrate3d(state, maxVoxels));
  },
  reset3d(sessionId) {
    return request("/3d/reset", { method: "POST", body: JSON.stringify({ session_id: sessionId }) }).then(hydrate3d);
  },
  clear3d(sessionId) {
    return request("/3d/clear", { method: "POST", body: JSON.stringify({ session_id: sessionId }) }).then(hydrate3d);
  },
  step3d(sessionId, steps = 1, collectMetrics = false) {
    return request("/3d/step", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, steps, collect_metrics: collectMetrics }),
    }).then(hydrate3d);
  },
  randomize3d(sessionId, seed, density) {
    return request("/3d/randomize", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, seed, density }),
    }).then(hydrate3d);
  },
  setRule3d(sessionId, rule) {
    return request("/3d/rule", { method: "POST", body: JSON.stringify({ session_id: sessionId, rule }) }).then(hydrate3d);
  },
  randomRule3d(sessionId, seed = null) {
    return request("/3d/random-rule", { method: "POST", body: JSON.stringify({ session_id: sessionId, seed }) }).then(hydrate3d);
  },
  runUntilStable3d(sessionId, maxSteps = 100) {
    return request("/3d/run-until-stable", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, max_steps: maxSteps }),
    }).then(hydrate3d);
  },
  slice3d(sessionId, axis, index) {
    return request(`/3d/slice?session_id=${encodeURIComponent(sessionId)}&axis=${axis}&index=${index}`);
  },
  exportState3d(sessionId) {
    return download(`/3d/export${sessionQuery(sessionId)}`, "life-lab-3d-state.npz");
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
