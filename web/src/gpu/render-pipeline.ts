/**
 * Render Pipeline wrapper — fullscreen quad blit from texture to canvas.
 */
export class RenderPipeline {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline | null = null;
  private sampler: GPUSampler | null = null;
  private bindGroup: GPUBindGroup | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Initialize the fullscreen blit render pipeline.
   */
  async init(
    shaderSource: string,
    canvasFormat: GPUTextureFormat
  ): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: 'fullscreen-shader',
      code: shaderSource,
    });

    this.sampler = this.device.createSampler({
      label: 'blit-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    this.pipeline = await this.device.createRenderPipelineAsync({
      label: 'fullscreen-blit-pipeline',
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: canvasFormat }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
  }

  /**
   * Update the source texture for blitting.
   */
  updateTexture(texture: GPUTexture): void {
    if (!this.pipeline || !this.sampler) {
      throw new Error('Pipeline not initialized');
    }
    this.bindGroup = this.device.createBindGroup({
      label: 'fullscreen-bind-group',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
  }

  /**
   * Render the fullscreen quad to the given texture view (canvas).
   */
  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void {
    if (!this.pipeline || !this.bindGroup) {
      throw new Error('Pipeline not initialized or texture not set');
    }
    const pass = encoder.beginRenderPass({
      label: 'fullscreen-blit-pass',
      colorAttachments: [
        {
          view: targetView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3, 1, 0, 0);
    pass.end();
  }
}
