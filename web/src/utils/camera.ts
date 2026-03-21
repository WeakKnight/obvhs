/**
 * Orbit Camera with mouse drag rotation and scroll zoom.
 * Outputs eye, lookAt, up, fov suitable for GPU uniform upload.
 */
export class OrbitCamera {
  public target: [number, number, number] = [0, 0, 0];
  public distance: number = 5;
  public theta: number = Math.PI * 0.25; // azimuth
  public phi: number = Math.PI * 0.35; // elevation (0 = top, PI = bottom)
  public fov: number = 60;

  private isDragging = false;
  private lastX = 0;
  private lastY = 0;
  private dirty = true;

  constructor(private canvas: HTMLCanvasElement) {
    this.attach();
  }

  private attach(): void {
    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.lastX;
      const dy = e.clientY - this.lastY;
      this.lastX = e.clientX;
      this.lastY = e.clientY;

      this.theta -= dx * 0.005;
      this.phi = Math.max(0.05, Math.min(Math.PI - 0.05, this.phi + dy * 0.005));
      this.dirty = true;
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.distance *= 1 + e.deltaY * 0.001;
      this.distance = Math.max(0.5, Math.min(100, this.distance));
      this.dirty = true;
    }, { passive: false });
  }

  /** Returns the eye position in world space. */
  get eye(): [number, number, number] {
    const sinPhi = Math.sin(this.phi);
    return [
      this.target[0] + this.distance * sinPhi * Math.cos(this.theta),
      this.target[1] + this.distance * Math.cos(this.phi),
      this.target[2] + this.distance * sinPhi * Math.sin(this.theta),
    ];
  }

  /** Write camera data to a Float32Array for uniform upload (64 bytes).
   * @param tlasStart — node index where TLAS begins in the concatenated bvh_nodes buffer
   */
  toUniformData(width: number, height: number, frame: number, tlasStart: number = 0): ArrayBuffer {
    const buf = new ArrayBuffer(64);
    const f32 = new Float32Array(buf);
    const u32 = new Uint32Array(buf);

    const e = this.eye;
    f32[0] = e[0]; f32[1] = e[1]; f32[2] = e[2]; // eye
    f32[3] = 0; // pad
    f32[4] = this.target[0]; f32[5] = this.target[1]; f32[6] = this.target[2]; // look_at
    f32[7] = 0; // pad
    f32[8] = 0; f32[9] = 1; f32[10] = 0; // up
    f32[11] = this.fov; // fov
    u32[12] = width; u32[13] = height; u32[14] = frame;
    u32[15] = tlasStart; // tlas_start — aligns with tray_racing ViewUniform.tlas_start

    return buf;
  }

  /** Check if the camera has moved since last query. Resets dirty flag. */
  consumeDirty(): boolean {
    const d = this.dirty;
    this.dirty = false;
    return d;
  }

  markDirty(): void {
    this.dirty = true;
  }
}
