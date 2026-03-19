/**
 * obvhs-web — Application Entry Point
 * Initializes WebGPU + WASM, manages tab switching between demos.
 */
import { initGpu } from './gpu/device';
import { BvhManager } from './bvh/bvh-manager';
import { RaytraceDemo } from './demos/raytrace-demo';
import { PhysicsDemo } from './demos/physics-demo';

const loading = document.getElementById('loading')!;
const loadingText = loading.querySelector('.loading-text')!;

// Stat elements
const statBuild = document.getElementById('statBuild')!;
const statGpu = document.getElementById('statGpu')!;
const statFps = document.getElementById('statFps')!;
const statTris = document.getElementById('statTris')!;
const statNodes = document.getElementById('statNodes')!;

function updateStats(stats: {
  buildTimeMs: number;
  gpuTimeMs: number;
  fps: number;
  triangles: number;
  nodes: number;
}) {
  statBuild.textContent = stats.buildTimeMs.toFixed(1);
  statGpu.textContent = stats.gpuTimeMs.toFixed(1);
  statFps.textContent = String(stats.fps);
  statTris.textContent = formatNumber(stats.triangles);
  statNodes.textContent = formatNumber(stats.nodes);
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return String(n);
}

async function main() {
  // Check WebGPU support
  if (!navigator.gpu) {
    loadingText.textContent =
      'WebGPU is not supported. Please use Chrome 113+ or Edge 113+.';
    return;
  }

  loadingText.textContent = 'Initializing WebGPU…';

  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;

  // Set initial canvas size
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;

  // Initialize GPU
  const { device, context, format } = await initGpu(canvas);

  loadingText.textContent = 'Loading WASM module…';

  // Initialize BVH Manager with WASM
  const bvhManager = new BvhManager();

  // Note: WASM module needs to be built first with: npm run build:wasm
  try {
    await bvhManager.init();
  } catch (e) {
    // If WASM not built yet, show helpful message
    console.warn('WASM module not found, attempting to proceed with demo mode:', e);
    loadingText.textContent = 
      'WASM module not found. Run: cd crates/obvhs-wasm && wasm-pack build --target web --out-dir ../../web/src/wasm/pkg';
    return;
  }

  loadingText.textContent = 'Building BVH and initializing shaders…';

  // Create demos
  let activeDemo: RaytraceDemo | PhysicsDemo | null = null;
  let activeTab = 'raytrace';
  let physicsDemo: PhysicsDemo | null = null;

  const raytraceDemo = new RaytraceDemo(device, context, format, canvas, bvhManager);
  await raytraceDemo.init();
  raytraceDemo.onStats = updateStats;

  // Start with ray tracing demo
  activeDemo = raytraceDemo;
  raytraceDemo.start();

  // Hide loading
  loading.classList.add('hidden');

  // ── Tab Switching ──
  const tabs = document.querySelectorAll<HTMLButtonElement>('.tab-btn');
  const raytraceParams = document.getElementById('raytraceParams')!;
  const physicsParams = document.getElementById('physicsParams')!;

  tabs.forEach((btn) => {
    btn.addEventListener('click', async () => {
      const tab = btn.dataset.tab!;
      if (tab === activeTab) return;

      tabs.forEach((t) => t.classList.remove('active'));
      btn.classList.add('active');
      activeTab = tab;

      // Stop current demo
      activeDemo?.stop();

      if (tab === 'raytrace') {
        raytraceParams.style.display = '';
        physicsParams.style.display = 'none';
        activeDemo = raytraceDemo;
        raytraceDemo.start();
      } else {
        raytraceParams.style.display = 'none';
        physicsParams.style.display = '';

        if (!physicsDemo) {
          loadingText.textContent = 'Initializing physics simulation…';
          loading.classList.remove('hidden');

          const sphereCount = parseInt(
            (document.getElementById('sphereCount') as HTMLInputElement).value
          );
          physicsDemo = new PhysicsDemo(device, context, format, canvas, bvhManager);
          await physicsDemo.init(sphereCount);
          physicsDemo.onStats = updateStats;

          loading.classList.add('hidden');
        }
        activeDemo = physicsDemo;
        physicsDemo.start();
      }
    });
  });

  // ── Side Panel Toggle ──
  const sidePanel = document.getElementById('sidePanel')!;
  const panelToggle = document.getElementById('panelToggle')!;
  panelToggle.addEventListener('click', () => {
    sidePanel.classList.toggle('collapsed');
    panelToggle.textContent = sidePanel.classList.contains('collapsed') ? '▶' : '◀';
  });

  // ── Slider bindings ──
  const bindSlider = (id: string, valId: string, onChange?: (val: number) => void) => {
    const slider = document.getElementById(id) as HTMLInputElement;
    const valSpan = document.getElementById(valId)!;
    slider.addEventListener('input', () => {
      valSpan.textContent = slider.value;
      onChange?.(parseFloat(slider.value));
    });
  };

  bindSlider('searchDist', 'searchDistVal');
  bindSlider('resScale', 'resScaleVal', (val) => {
    raytraceDemo.setResolutionScale(val);
  });
  bindSlider('sphereCount', 'sphereCountVal');

  // ── Physics controls ──
  const bvhStrategy = document.getElementById('bvhStrategy') as HTMLSelectElement;
  bvhStrategy.addEventListener('change', () => {
    physicsDemo?.setUpdateStrategy(
      bvhStrategy.value as 'rebuild' | 'reinsert' | 'partial_rebuild'
    );
  });

  const addSphereBtn = document.getElementById('addSphereBtn')!;
  addSphereBtn.addEventListener('click', () => {
    physicsDemo?.addNewSphere();
  });
}

main().catch((err) => {
  console.error('Fatal:', err);
  loadingText.textContent = `Error: ${err.message}`;
});
