const SVG_NS = "http://www.w3.org/2000/svg";
const WIDTH = 640;
const HEIGHT = 310;
const PADDING = { right: 20, left: 48 };
const POPULATION_PLOT = { top: 30, height: 96 };
const EVENTS_PLOT = { top: 178, height: 92 };

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function addText(svg, text, x, y, attributes = {}) {
  const label = svgElement("text", { x, y, ...attributes });
  label.textContent = text;
  svg.appendChild(label);
  return label;
}

function clamp(value) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function pointValue(point, key, totalCells) {
  const fractionKeys = {
    alive: "alive_fraction",
    activity: "changed_fraction",
    births: "birth_fraction",
    deaths: "death_fraction",
  };
  const countKeys = {
    alive: "alive",
    activity: "changed_cells",
    births: "births",
    deaths: "deaths",
  };
  const fraction = point[fractionKeys[key]];
  if (fraction !== undefined) return clamp(Number(fraction));
  return clamp(Number(point[countKeys[key]] || 0) / Math.max(totalCells, 1));
}

function pathFor(points, key, totalCells, plot) {
  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const x = (index) => PADDING.left + (index / Math.max(points.length - 1, 1)) * plotWidth;
  const y = (value) => plot.top + (1 - value) * plot.height;
  return points
    .map((point, index) => {
      const command = index === 0 ? "M" : "L";
      return `${command} ${x(index).toFixed(2)} ${y(pointValue(point, key, totalCells)).toFixed(2)}`;
    })
    .join(" ");
}

function drawPlotFrame(svg, plot, label) {
  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  addText(svg, label, PADDING.left, plot.top - 10, {
    fill: "#b7c9df",
    "font-size": 11,
    "font-weight": 700,
    "letter-spacing": "0.08em",
  });
  for (const fraction of [0, 0.5, 1]) {
    const y = plot.top + (1 - fraction) * plot.height;
    svg.appendChild(svgElement("line", {
      x1: PADDING.left,
      y1: y,
      x2: PADDING.left + plotWidth,
      y2: y,
      stroke: "#25364f",
      "stroke-width": 1,
    }));
    addText(svg, `${Math.round(fraction * 100)}%`, PADDING.left - 10, y + 4, {
      fill: "#8492a9",
      "font-size": 11,
      "text-anchor": "end",
    });
  }
}

function drawSeries(svg, points, key, color, totalCells, plot, options = {}) {
  svg.appendChild(svgElement("path", {
    d: pathFor(points, key, totalCells, plot),
    fill: "none",
    stroke: color,
    "stroke-width": options.width || 2,
    "stroke-dasharray": options.dash || "",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
  }));
}

/** Render useful transition diagnostics without adding a chart dependency. */
export function renderMetricsChart(svg, points, totalCells) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    "Population, activity, births, and deaths over generations",
  );

  const title = svgElement("title");
  title.textContent = "Population, activity, births, and deaths over generations";
  svg.appendChild(title);
  svg.appendChild(svgElement("rect", {
    x: 0,
    y: 0,
    width: WIDTH,
    height: HEIGHT,
    rx: 12,
    fill: "#0a1220",
  }));

  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  drawPlotFrame(svg, POPULATION_PLOT, "STATE");
  drawPlotFrame(svg, EVENTS_PLOT, "TRANSITIONS");

  if (!points.length) {
    addText(svg, "Start the simulation to build a history", WIDTH / 2, 150, {
      fill: "#8492a9",
      "font-size": 13,
      "text-anchor": "middle",
    });
    return;
  }

  drawSeries(svg, points, "alive", "#70e2bd", totalCells, POPULATION_PLOT, { width: 3 });
  drawSeries(svg, points, "activity", "#f0bd76", totalCells, POPULATION_PLOT, { dash: "5 5" });
  drawSeries(svg, points, "births", "#6fb3ff", totalCells, EVENTS_PLOT, { width: 2 });
  drawSeries(svg, points, "deaths", "#f39c92", totalCells, EVENTS_PLOT, { width: 2 });

  const lastIndex = points.length - 1;
  const lastX = PADDING.left + (lastIndex / Math.max(points.length - 1, 1)) * plotWidth;
  for (const [key, color, plot] of [
    ["alive", "#70e2bd", POPULATION_PLOT],
    ["activity", "#f0bd76", POPULATION_PLOT],
    ["births", "#6fb3ff", EVENTS_PLOT],
    ["deaths", "#f39c92", EVENTS_PLOT],
  ]) {
    const lastY = plot.top + (1 - pointValue(points[lastIndex], key, totalCells)) * plot.height;
    svg.appendChild(svgElement("circle", {
      cx: lastX,
      cy: lastY,
      r: 3.5,
      fill: "#0a1220",
      stroke: color,
      "stroke-width": 2,
    }));
  }

  const firstGeneration = points[0].generation;
  const lastGeneration = points[lastIndex].generation;
  addText(svg, `gen ${firstGeneration}`, PADDING.left, HEIGHT - 14, {
    fill: "#8492a9",
    "font-size": 11,
  });
  addText(svg, `gen ${lastGeneration}`, PADDING.left + plotWidth, HEIGHT - 14, {
    fill: "#8492a9",
    "font-size": 11,
    "text-anchor": "end",
  });
}
