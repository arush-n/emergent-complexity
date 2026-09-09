export class GridCanvas {
  constructor(canvas, { onCommit } = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d", { alpha: false, desynchronized: true });
    this.bitmapCanvas = document.createElement("canvas");
    this.bitmapContext = this.bitmapCanvas.getContext("2d", { alpha: false });
    this.bitmapData = null;
    const endianProbe = new Uint8Array(new Uint32Array([0x0a0b0c0d]).buffer);
    this.backgroundPixel = endianProbe[0] === 0x0d ? 0xff050505 : 0x050505ff;
    this.alivePixel = endianProbe[0] === 0x0d ? 0xffe5eeee : 0xeeeee5ff;
    this.onCommit = onCommit || (() => {});
    this.grid = new Uint8Array();
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
    if (this.bitmapCanvas.width !== width || this.bitmapCanvas.height !== height) {
      this.bitmapCanvas.width = width;
      this.bitmapCanvas.height = height;
      this.bitmapData = null;
    }
    if (grid instanceof Uint8Array) {
      this.grid = new Uint8Array(grid);
    } else {
      this.grid = Uint8Array.from(grid.flatMap((row) => row));
    }
    this.render();
  }

  toRows() {
    return Array.from({ length: this.height }, (_, row) =>
      Array.from(this.grid.slice(row * this.width, (row + 1) * this.width)),
    );
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
    context.imageSmoothingEnabled = false;
    const renderedBitmap = this.renderBitmap(context, offsetX, offsetY, cellSize);
    if (!renderedBitmap) {
      context.fillStyle = "#050505";
      context.fillRect(0, 0, cssWidth, cssHeight);
      context.fillStyle = "#eeede5";
      for (let row = 0; row < this.height; row += 1) {
        for (let col = 0; col < this.width; col += 1) {
          if (this.grid[row * this.width + col]) {
            context.fillRect(offsetX + col * cellSize, offsetY + row * cellSize, cellSize, cellSize);
          }
        }
      }
    }

    if (this.showGrid && cellSize >= 4) {
      context.beginPath();
      context.strokeStyle = "#252523";
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

  renderBitmap(context, offsetX, offsetY, cellSize) {
    if (!this.bitmapContext) return false;
    if (!this.bitmapData) {
      this.bitmapData = this.bitmapContext.createImageData(this.width, this.height);
    }
    const pixels = new Uint32Array(this.bitmapData.data.buffer);
    pixels.fill(this.backgroundPixel);
    for (let index = 0; index < this.grid.length; index += 1) {
      if (this.grid[index]) pixels[index] = this.alivePixel;
    }
    this.bitmapContext.putImageData(this.bitmapData, 0, 0);
    context.drawImage(
      this.bitmapCanvas,
      0,
      0,
      this.width,
      this.height,
      offsetX,
      offsetY,
      cellSize * this.width,
      cellSize * this.height,
    );
    return true;
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
    const index = cell.row * this.width + cell.col;
    if (erase) this.grid[index] = 0;
    else if (paint) this.grid[index] = 1;
    else this.grid[index] = this.grid[index] ? 0 : 1;
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
    if (this.dirty) this.onCommit(this.toRows());
    this.visited.clear();
  }
}
