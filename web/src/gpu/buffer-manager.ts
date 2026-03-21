/**
 * GPU Buffer Manager — manages storage buffers for BVH data on the GPU.
 * Handles zero-copy uploads from WASM linear memory.
 */
export class BufferManager {
  private device: GPUDevice;
  private buffers: Map<string, GPUBuffer> = new Map();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Create or update a storage buffer from a Uint8Array (zero-copy from WASM).
   * Only recreates the buffer if size changes.
   */
  uploadBuffer(name: string, data: Uint8Array, usage?: number): GPUBuffer {
    const gpuUsage =
      usage ?? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    let buf = this.buffers.get(name);

    if (!buf || buf.size < data.byteLength) {
      buf?.destroy();
      buf = this.device.createBuffer({
        label: name,
        size: Math.max(data.byteLength, 16), // minimum 16 bytes
        usage: gpuUsage,
      });
      this.buffers.set(name, buf);
    }

    this.device.queue.writeBuffer(buf, 0, data as unknown as ArrayBuffer);
    return buf;
  }

  /**
   * Upload BVH data from WASM memory pointers.
   * Uses zero-copy view into WASM linear memory.
   */
  uploadFromWasm(
    name: string,
    wasmMemory: WebAssembly.Memory,
    ptr: number,
    byteLen: number,
    usage?: number
  ): GPUBuffer {
    const view = new Uint8Array(wasmMemory.buffer, ptr, byteLen);
    return this.uploadBuffer(name, view, usage);
  }

  /**
   * Get an existing buffer by name.
   */
  getBuffer(name: string): GPUBuffer | undefined {
    return this.buffers.get(name);
  }

  /**
   * Create a uniform buffer from typed data.
   */
  createUniformBuffer(name: string, data: ArrayBuffer): GPUBuffer {
    return this.uploadBuffer(
      name,
      new Uint8Array(data),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
  }

  /**
   * Create a storage buffer for read-write (e.g., output buffers).
   */
  createStorageBuffer(
    name: string,
    size: number,
    readWrite: boolean = false
  ): GPUBuffer {
    let buf = this.buffers.get(name);
    if (!buf || buf.size < size) {
      buf?.destroy();
      buf = this.device.createBuffer({
        label: name,
        size: Math.max(size, 16),
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          (readWrite ? GPUBufferUsage.COPY_SRC : 0),
      });
      this.buffers.set(name, buf);
    }
    return buf;
  }

  /**
   * Destroy all managed buffers.
   */
  destroy(): void {
    for (const buf of this.buffers.values()) {
      buf.destroy();
    }
    this.buffers.clear();
  }
}
