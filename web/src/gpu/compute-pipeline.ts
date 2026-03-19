/**
 * Compute Pipeline wrapper — loads WGSL source and creates compute pipelines.
 */
export class ComputePipeline {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private _pipelineLayout: GPUPipelineLayout | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Initialize the compute pipeline from WGSL source.
   * @param shaderSource Combined WGSL source code.
   * @param entryPoint Compute entry point name (default: "main").
   * @param bindGroupLayouts Explicit bind group layouts (optional).
   */
  async init(
    shaderSource: string,
    entryPoint: string = 'main',
    bindGroupLayouts?: GPUBindGroupLayout[]
  ): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: `compute-shader`,
      code: shaderSource,
    });

    if (bindGroupLayouts) {
      this._pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts,
      });
    }

    this.pipeline = await this.device.createComputePipelineAsync({
      label: `compute-pipeline`,
      layout: this._pipelineLayout ?? 'auto',
      compute: {
        module: shaderModule,
        entryPoint,
      },
    });

    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }

  /**
   * Create a bind group for group 0 from buffer entries.
   */
  createBindGroup(
    entries: { binding: number; resource: GPUBindingResource }[],
    groupIndex: number = 0
  ): GPUBindGroup {
    if (!this.pipeline) throw new Error('Pipeline not initialized');
    return this.device.createBindGroup({
      label: `compute-bind-group-${groupIndex}`,
      layout: this.pipeline.getBindGroupLayout(groupIndex),
      entries,
    });
  }

  /**
   * Dispatch the compute shader.
   */
  dispatch(
    encoder: GPUCommandEncoder,
    bindGroups: GPUBindGroup[],
    workgroupCountX: number,
    workgroupCountY: number = 1,
    workgroupCountZ: number = 1
  ): void {
    if (!this.pipeline) throw new Error('Pipeline not initialized');
    const pass = encoder.beginComputePass({ label: 'compute-pass' });
    pass.setPipeline(this.pipeline);
    for (let i = 0; i < bindGroups.length; i++) {
      pass.setBindGroup(i, bindGroups[i]);
    }
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
    pass.end();
  }

  get layout(): GPUBindGroupLayout | null {
    return this.bindGroupLayout;
  }
}
