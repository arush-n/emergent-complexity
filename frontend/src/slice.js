export function renderSlice(canvas, grid, showGrid = true) {
  const rows = grid.length;
  const columns = rows ? grid[0].length : 0;
  if (!rows || !columns) return;
  const size = Math.max(1, Math.floor(Math.min(520 / columns, 300 / rows)));
  canvas.width = columns * size;
  canvas.height = rows * size;
  const context = canvas.getContext("2d");
  context.fillStyle = "#050505";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      if (!grid[row][column]) continue;
      context.fillStyle = "#eeede5";
      context.fillRect(column * size, row * size, size, size);
    }
  }
  if (showGrid && size > 3) {
    context.strokeStyle = "#252523";
    context.lineWidth = 1;
    context.beginPath();
    for (let column = 0; column <= columns; column += 1) {
      context.moveTo(column * size + 0.5, 0);
      context.lineTo(column * size + 0.5, canvas.height);
    }
    for (let row = 0; row <= rows; row += 1) {
      context.moveTo(0, row * size + 0.5);
      context.lineTo(canvas.width, row * size + 0.5);
    }
    context.stroke();
  }
}
