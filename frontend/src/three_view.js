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

function sameDimensions(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
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
    this.voxelGeometry = null;
    this.voxelMaterial = null;
    this.meshCapacity = 0;
    this.bounds = null;
    this.dimensions = [1, 1, 1];
    this.renderedCount = 0;
    this.sampled = false;
    this.ready = false;
    this.active = false;
    this.fps = 0;
    this.lastRenderMs = 0;
    this.frameCount = 0;
    this.fpsStarted = performance.now();
    this.resizeObserver = null;
    this.animationFrame = null;
    this.pendingVoxels = null;
    this.renderRequested = false;
    this.onControlsChange = () => this.requestRender();
    this.onVisibilityChange = () => {
      if (document.visibilityState === "visible") this.requestRender();
      else this.cancelRender();
    };
  }

  async init() {
    if (this.ready) return true;
    try {
      const [THREE, controlsModule] = await loadThree();
      this.THREE = THREE;
      const { OrbitControls } = controlsModule;
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color("#050505");
      this.camera = new THREE.PerspectiveCamera(48, 1, 0.1, 5000);
      this.renderer = new THREE.WebGLRenderer({
        antialias: false,
        alpha: false,
        powerPreference: "low-power",
      });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.container.querySelectorAll("canvas").forEach((canvas) => canvas.remove());
      this.container.appendChild(this.renderer.domElement);

      this.scene.add(new THREE.AmbientLight("#ffffff", 1.1));
      const keyLight = new THREE.DirectionalLight("#ffffff", 1.8);
      keyLight.position.set(20, 35, 25);
      this.scene.add(keyLight);
      const fillLight = new THREE.DirectionalLight("#d9d9d2", 0.55);
      fillLight.position.set(-25, -15, 25);
      this.scene.add(fillLight);

      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.screenSpacePanning = true;
      this.controls.minDistance = 2;
      this.controls.maxDistance = 4000;
      this.controls.addEventListener("change", this.onControlsChange);

      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.container);
      document.addEventListener("visibilitychange", this.onVisibilityChange);
      this.resize();
      this.ready = true;
      const loading = this.container.querySelector(".viewport-message");
      if (loading) loading.remove();
      const pending = this.pendingVoxels;
      this.pendingVoxels = null;
      if (pending) this.setVoxels(...pending);
      this.requestRender();
      return true;
    } catch (error) {
      const loading = this.container.querySelector(".viewport-message");
      if (loading) loading.textContent = `3D renderer unavailable: ${error.message}`;
      return false;
    }
  }

  start() {
    this.active = true;
    this.requestRender();
  }

  stop() {
    this.active = false;
    this.cancelRender();
  }

  dispose() {
    this.stop();
    this.resizeObserver?.disconnect();
    document.removeEventListener("visibilitychange", this.onVisibilityChange);
    this.controls?.removeEventListener("change", this.onControlsChange);
    this.disposeVoxelMesh();
    if (this.bounds) {
      this.scene?.remove(this.bounds);
      this.bounds.geometry.dispose();
      this.bounds.material.dispose();
      this.bounds = null;
    }
    this.controls?.dispose();
    this.renderer?.dispose();
    this.renderer?.domElement.remove();
    this.resizeObserver = null;
    this.controls = null;
    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this.ready = false;
  }

  cancelRender() {
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = null;
    this.renderRequested = false;
  }

  requestRender() {
    if (!this.ready || !this.active || document.visibilityState === "hidden" || this.renderRequested) return;
    this.renderRequested = true;
    this.animationFrame = requestAnimationFrame((now) => {
      this.animationFrame = null;
      this.renderRequested = false;
      if (!this.ready || !this.active || document.visibilityState === "hidden") return;
      const changed = this.controls?.update() || false;
      this.renderNow(now);
      if (changed) this.requestRender();
    });
  }

  renderNow(now = performance.now()) {
    if (!this.renderer || !this.scene || !this.camera) return;
    const renderStart = performance.now();
    this.renderer.render(this.scene, this.camera);
    this.lastRenderMs = performance.now() - renderStart;
    this.frameCount += 1;
    const elapsed = now - this.fpsStarted;
    if (elapsed >= 1000) {
      this.fps = (this.frameCount * 1000) / elapsed;
      this.frameCount = 0;
      this.fpsStarted = now;
    }
  }

  resize() {
    if (!this.renderer || !this.camera) return;
    const width = Math.max(this.container.clientWidth, 320);
    const height = Math.max(this.container.clientHeight, 320);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.requestRender();
  }

  setVoxels(voxels, dimensions, sampled = false) {
    const nextDimensions = dimensions.map(Number);
    const dimensionsChanged = !sameDimensions(this.dimensions, nextDimensions);
    this.pendingVoxels = [voxels, nextDimensions, sampled];
    this.dimensions = nextDimensions;
    this.sampled = sampled;
    if (!this.ready) return;
    this.pendingVoxels = null;

    const [depth, height, width] = this.dimensions;
    const coordinateCount = voxels instanceof Uint8Array ? voxels.length / 3 : voxels.length;
    const count = Math.min(coordinateCount, this.maxVoxels);
    this.ensureMesh(count, dimensionsChanged);
    this.renderedCount = count;
    if (this.mesh) {
      const matrix = new this.THREE.Matrix4();
      this.mesh.count = count;
      for (let index = 0; index < count; index += 1) {
        let z;
        let y;
        let x;
        if (voxels instanceof Uint8Array) {
          const offset = index * 3;
          z = voxels[offset];
          y = voxels[offset + 1];
          x = voxels[offset + 2];
        } else {
          [z, y, x] = voxels[index];
        }
        matrix.setPosition(
          x - width / 2 + 0.5,
          height / 2 - y - 0.5,
          z - depth / 2 + 0.5,
        );
        this.mesh.setMatrixAt(index, matrix);
      }
      this.mesh.instanceMatrix.needsUpdate = true;
    }
    if (dimensionsChanged || !this.bounds) {
      this.updateBounds();
      if (dimensionsChanged) this.resetCamera();
    }
    this.requestRender();
  }

  ensureMesh(requiredCount, dimensionsChanged) {
    if (!this.ready || (!dimensionsChanged && this.mesh && requiredCount <= this.meshCapacity)) return;
    this.disposeVoxelMesh();
    if (requiredCount === 0) return;

    const THREE = this.THREE;
    const [depth, height, width] = this.dimensions;
    const maxDimension = Math.max(depth, height, width);
    const voxelSize = Math.min(1, 26 / maxDimension);
    const capacity = Math.min(
      this.maxVoxels,
      Math.max(1, 2 ** Math.ceil(Math.log2(Math.max(requiredCount, 1)))),
    );
    this.voxelGeometry = new THREE.BoxGeometry(
      voxelSize * 0.86,
      voxelSize * 0.86,
      voxelSize * 0.86,
    );
    this.voxelMaterial = new THREE.MeshLambertMaterial({
      color: "#eeede5",
    });
    this.mesh = new THREE.InstancedMesh(this.voxelGeometry, this.voxelMaterial, capacity);
    this.mesh.count = 0;
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    this.meshCapacity = capacity;
    this.scene.add(this.mesh);
  }

  disposeVoxelMesh() {
    if (this.mesh) this.scene?.remove(this.mesh);
    this.mesh?.geometry.dispose();
    this.mesh?.material.dispose();
    this.mesh = null;
    this.voxelGeometry = null;
    this.voxelMaterial = null;
    this.meshCapacity = 0;
    this.renderedCount = 0;
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
    const box = new THREE.BoxGeometry(width, height, depth);
    const edges = new THREE.EdgesGeometry(box);
    box.dispose();
    const material = new THREE.LineBasicMaterial({ color: "#41413d", transparent: true, opacity: 0.9 });
    this.bounds = new THREE.LineSegments(edges, material);
    this.scene.add(this.bounds);
    if (!this.camera.userData.hasFramed) {
      this.camera.userData.hasFramed = true;
      this.resetCamera();
    }
    this.camera.far = Math.max(5000, Math.max(depth, height, width) * 20);
    this.camera.updateProjectionMatrix();
  }

  setBoundsVisible(visible) {
    if (this.bounds) this.bounds.visible = visible;
    this.requestRender();
  }

  resetCamera() {
    if (!this.camera || !this.controls) return;
    const [depth, height, width] = this.dimensions;
    const distance = Math.max(depth, height, width) * 1.9;
    this.camera.position.set(distance, distance * 0.8, distance);
    this.camera.up.set(0, 1, 0);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.requestRender();
  }

  orient(view) {
    if (!this.camera || !this.controls) return;
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
    this.requestRender();
  }

  capture(metadata) {
    if (!this.renderer) return;
    this.renderNow();
    const source = this.renderer.domElement;
    const canvas = document.createElement("canvas");
    canvas.width = source.width;
    canvas.height = source.height + 42;
    const context = canvas.getContext("2d");
    context.fillStyle = "#050505";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(source, 0, 0);
    context.fillStyle = "#111110";
    context.fillRect(0, source.height, canvas.width, 42);
    context.fillStyle = "#e8e7df";
    context.font = "14px monospace";
    context.fillText(metadata, 14, source.height + 26);
    const link = document.createElement("a");
    link.download = `life-lab-3d-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }

  getStats() {
    return {
      renderedCount: this.renderedCount,
      sampled: this.sampled,
      fps: this.fps,
      lastRenderMs: this.lastRenderMs,
    };
  }
}
