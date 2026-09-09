let threePromise;

async function loadThree() {
  if (!threePromise) {
    threePromise = Promise.all([
      import("three"),
      import("three/addons/controls/OrbitControls.js"),
    ]);
  }
  return threePromise;
}

export class ThreeVoxelView {
  constructor(container, { maxVoxels = 75000 } = {}) {
    this.container = container;
    this.maxVoxels = maxVoxels;
    this.THREE = null;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.mesh = null;
    this.bounds = null;
    this.dimensions = [1, 1, 1];
    this.renderedCount = 0;
    this.sampled = false;
    this.ready = false;
    this.fps = 0;
    this.frameCount = 0;
    this.fpsStarted = performance.now();
    this.resizeObserver = null;
    this.animationFrame = null;
    this.pendingVoxels = null;
  }

  async init() {
    if (this.ready) return true;
    try {
      const [THREE, controlsModule] = await loadThree();
      this.THREE = THREE;
      const { OrbitControls } = controlsModule;
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color("#05080d");
      this.camera = new THREE.PerspectiveCamera(48, 1, 0.1, 5000);
      this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.container.querySelectorAll("canvas").forEach((canvas) => canvas.remove());
      this.container.appendChild(this.renderer.domElement);

      this.scene.add(new THREE.AmbientLight("#a9c4bd", 1.6));
      const keyLight = new THREE.DirectionalLight("#eafff7", 2.8);
      keyLight.position.set(20, 35, 25);
      this.scene.add(keyLight);
      const fillLight = new THREE.PointLight("#70e2bd", 1.4, 220);
      fillLight.position.set(-25, -15, 25);
      this.scene.add(fillLight);

      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.screenSpacePanning = true;
      this.controls.minDistance = 2;
      this.controls.maxDistance = 4000;

      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.container);
      this.resize();
      this.ready = true;
      const loading = this.container.querySelector(".viewport-message");
      if (loading) loading.remove();
      if (this.pendingVoxels) this.setVoxels(...this.pendingVoxels);
      this.startAnimation();
      return true;
    } catch (error) {
      const loading = this.container.querySelector(".viewport-message");
      if (loading) loading.textContent = `3D renderer unavailable: ${error.message}`;
      return false;
    }
  }

  resize() {
    if (!this.renderer || !this.camera) return;
    const width = Math.max(this.container.clientWidth, 320);
    const height = Math.max(this.container.clientHeight, 320);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  startAnimation() {
    if (this.animationFrame) return;
    const render = (now) => {
      this.animationFrame = requestAnimationFrame(render);
      this.controls?.update();
      if (this.renderer && this.scene && this.camera) this.renderer.render(this.scene, this.camera);
      this.frameCount += 1;
      const elapsed = now - this.fpsStarted;
      if (elapsed >= 1000) {
        this.fps = (this.frameCount * 1000) / elapsed;
        this.frameCount = 0;
        this.fpsStarted = now;
      }
    };
    this.animationFrame = requestAnimationFrame(render);
  }

  setVoxels(voxels, dimensions, sampled = false) {
    const dimensionsChanged = this.dimensions.some((value, index) => value !== dimensions[index]);
    this.pendingVoxels = [voxels, dimensions, sampled];
    this.dimensions = [dimensions[0], dimensions[1], dimensions[2]];
    this.sampled = sampled;
    if (!this.ready) return;
    const [depth, height, width] = this.dimensions;
    const THREE = this.THREE;
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.mesh = null;
    }
    const count = Math.min(voxels.length, this.maxVoxels);
    this.renderedCount = count;
    if (count > 0) {
      const maxDimension = Math.max(depth, height, width);
      const voxelSize = Math.min(1, 26 / maxDimension);
      const geometry = new THREE.BoxGeometry(voxelSize * 0.86, voxelSize * 0.86, voxelSize * 0.86);
      const material = new THREE.MeshStandardMaterial({
        color: "#d8fff0",
        emissive: "#214d41",
        emissiveIntensity: 0.38,
        roughness: 0.64,
        metalness: 0.08,
      });
      this.mesh = new THREE.InstancedMesh(geometry, material, count);
      this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      const matrix = new THREE.Matrix4();
      const color = new THREE.Color();
      for (let index = 0; index < count; index += 1) {
        const [z, y, x] = voxels[index];
        matrix.setPosition(
          x - width / 2 + 0.5,
          height / 2 - y - 0.5,
          z - depth / 2 + 0.5,
        );
        this.mesh.setMatrixAt(index, matrix);
        color.setHSL(0.43, 0.42, 0.72 + (z / Math.max(depth, 1)) * 0.15);
        this.mesh.setColorAt(index, color);
      }
      this.mesh.instanceMatrix.needsUpdate = true;
      if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
      this.scene.add(this.mesh);
    }
    this.updateBounds();
    if (dimensionsChanged) this.resetCamera();
  }

  updateBounds() {
    if (!this.ready) return;
    const [depth, height, width] = this.dimensions;
    const THREE = this.THREE;
    if (this.bounds) {
      this.scene.remove(this.bounds);
      this.bounds.geometry.dispose();
      this.bounds.material.dispose();
    }
    const maxDimension = Math.max(depth, height, width);
    const box = new THREE.BoxGeometry(width, height, depth);
    const edges = new THREE.EdgesGeometry(box);
    box.dispose();
    const material = new THREE.LineBasicMaterial({ color: "#2b4854", transparent: true, opacity: 0.78 });
    this.bounds = new THREE.LineSegments(edges, material);
    this.scene.add(this.bounds);
    if (!this.camera.userData.hasFramed) {
      this.camera.userData.hasFramed = true;
      this.resetCamera();
    }
    this.camera.far = Math.max(5000, maxDimension * 20);
    this.camera.updateProjectionMatrix();
  }

  setBoundsVisible(visible) {
    if (this.bounds) this.bounds.visible = visible;
  }

  resetCamera() {
    const [depth, height, width] = this.dimensions;
    const distance = Math.max(depth, height, width) * 1.9;
    this.camera.position.set(distance, distance * 0.8, distance);
    this.camera.up.set(0, 1, 0);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  orient(view) {
    const [depth, height, width] = this.dimensions;
    const distance = Math.max(depth, height, width) * 2.3;
    this.camera.up.set(0, 1, 0);
    if (view === "front") this.camera.position.set(0, 0, distance);
    else if (view === "side") this.camera.position.set(distance, 0, 0);
    else if (view === "top") {
      this.camera.position.set(0, distance, 0);
      this.camera.up.set(0, 0, -1);
    } else {
      this.resetCamera();
      return;
    }
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  capture(metadata) {
    if (!this.renderer) return;
    const source = this.renderer.domElement;
    const canvas = document.createElement("canvas");
    canvas.width = source.width;
    canvas.height = source.height + 42;
    const context = canvas.getContext("2d");
    context.fillStyle = "#05080d";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(source, 0, 0);
    context.fillStyle = "#0b1321";
    context.fillRect(0, source.height, canvas.width, 42);
    context.fillStyle = "#d8fff0";
    context.font = "14px monospace";
    context.fillText(metadata, 14, source.height + 26);
    const link = document.createElement("a");
    link.download = `life-lab-3d-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }

  getStats() {
    return { renderedCount: this.renderedCount, sampled: this.sampled, fps: this.fps };
  }
}
