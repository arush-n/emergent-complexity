export class GridCanvas {
  constructor(canvas, { onCommit } = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.onCommit = onCommit || (() => {});
    this.grid = [];
    this.width = 0;
    this.height = 0;
    this.mode = "toggle";
    this.showGrid = true;
    this.dragging = false;
    this.pointerButton = 0;
    this.visited = new Set();
    this.dirty = false;

    canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
    canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
    canvas.addEventListener("pointerup", (event) => this.pointerUp(event));
    canvas.addEventListener("pointercancel", (event) => this.pointerUp(event));
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    window.addEventListener("resize", () => this.render());
  }

  setState(grid, width, height) {
    this.width = width;
    this.height = height;
    this.grid = grid.map((row) => row.slice());
    this.render();
  }

  setMode(mode) {
    this.mode = mode;
  }

  setShowGrid(showGrid) {
    this.showGrid = showGrid;
    this.render();
  }

  layout() {
    const bounds = this.canvas.getBoundingClientRect();
    const cssWidth = Math.max(bounds.width, 1);
    const cssHeight = Math.max(bounds.height, 1);
    const pixelRatio = window.devicePixelRatio || 1;
    const pixelWidth = Math.max(Math.floor(cssWidth * pixelRatio), 1);
    const pixelHeight = Math.max(Math.floor(cssHeight * pixelRatio), 1);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }
    return {
      cssWidth,
      cssHeight,
      pixelRatio,
      cellSize: Math.min(cssWidth / Math.max(this.width, 1), cssHeight / Math.max(this.height, 1)),
    };
  }

  render() {
    if (!this.context || !this.width || !this.height) return;
    const { cssWidth, cssHeight, pixelRatio, cellSize } = this.layout();
    const offsetX = (cssWidth - cellSize * this.width) / 2;
    const offsetY = (cssHeight - cellSize * this.height) / 2;
    const context = this.context;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.fillStyle = "#050910";
    context.fillRect(0, 0, cssWidth, cssHeight);

    context.fillStyle = "#d8fff0";
    for (let row = 0; row < this.height; row += 1) {
      for (let col = 0; col < this.width; col += 1) {
        if (this.grid[row]?.[col]) {
          context.fillRect(offsetX + col * cellSize, offsetY + row * cellSize, cellSize, cellSize);
        }
      }
    }

    if (this.showGrid && cellSize >= 3) {
      context.beginPath();
      context.strokeStyle = "rgba(106, 153, 189, .16)";
      context.lineWidth = 1;
      for (let col = 0; col <= this.width; col += 1) {
        const x = Math.round(offsetX + col * cellSize) + 0.5;
        context.moveTo(x, offsetY);
        context.lineTo(x, offsetY + cellSize * this.height);
      }
      for (let row = 0; row <= this.height; row += 1) {
        const y = Math.round(offsetY + row * cellSize) + 0.5;
        context.moveTo(offsetX, y);
        context.lineTo(offsetX + cellSize * this.width, y);
      }
      context.stroke();
    }
  }

  cellAt(event) {
    const bounds = this.canvas.getBoundingClientRect();
    const { cssWidth, cssHeight, cellSize } = this.layout();
    const offsetX = (cssWidth - cellSize * this.width) / 2;
    const offsetY = (cssHeight - cellSize * this.height) / 2;
    const x = event.clientX - bounds.left - offsetX;
    const y = event.clientY - bounds.top - offsetY;
    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);
    if (row < 0 || row >= this.height || col < 0 || col >= this.width) return null;
    return { row, col };
  }

  applyCell(event) {
    const cell = this.cellAt(event);
    if (!cell) return;
    const key = `${cell.row}:${cell.col}`;
    if (this.visited.has(key)) return;
    this.visited.add(key);
    const erase = this.pointerButton === 2 || this.mode === "erase";
    const paint = this.mode === "paint" && this.pointerButton !== 2;
    if (erase) this.grid[cell.row][cell.col] = 0;
    else if (paint) this.grid[cell.row][cell.col] = 1;
    else this.grid[cell.row][cell.col] = this.grid[cell.row][cell.col] ? 0 : 1;
    this.dirty = true;
    this.render();
  }

  pointerDown(event) {
    if (event.button !== 0 && event.button !== 2) return;
    event.preventDefault();
    this.dragging = true;
    this.pointerButton = event.button;
    this.visited = new Set();
    this.dirty = false;
    this.canvas.setPointerCapture?.(event.pointerId);
    this.applyCell(event);
  }

  pointerMove(event) {
    if (!this.dragging) return;
    event.preventDefault();
    this.applyCell(event);
  }

  pointerUp(event) {
    if (!this.dragging) return;
    this.dragging = false;
    this.canvas.releasePointerCapture?.(event.pointerId);
    if (this.dirty) this.onCommit(this.grid.map((row) => row.slice()));
    this.visited.clear();
  }
}

