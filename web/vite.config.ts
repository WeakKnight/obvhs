import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import { resolve } from 'path';
import { readFileSync } from 'fs';

// Plugin to inline .wgsl files as string constants in library build
function wgslInlinePlugin() {
  return {
    name: 'wgsl-inline',
    transform(code: string, id: string) {
      if (id.endsWith('.wgsl?raw') || (id.endsWith('.wgsl') && !id.includes('node_modules'))) {
        const filePath = id.replace('?raw', '');
        const content = readFileSync(filePath, 'utf-8');
        return {
          code: `export default ${JSON.stringify(content)};`,
          map: null,
        };
      }
    },
  };
}

const isLib = process.env.BUILD_MODE === 'lib';

export default defineConfig({
  plugins: [wasm(), ...(isLib ? [wgslInlinePlugin()] : [])],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    headers: {
      // Required for SharedArrayBuffer (future rayon/wasm-bindgen-rayon support)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: isLib
    ? {
        target: 'es2022',
        lib: {
          entry: resolve(__dirname, 'src/index.ts'),
          formats: ['es'],
          fileName: 'obvhs',
        },
        rollupOptions: {
          // WASM module is external — users provide it themselves or use the bundled one
          external: [/\.\/wasm\/pkg/, /obvhs-wasm/],
          output: {
            // Preserve module structure for tree-shaking
            preserveModules: false,
          },
        },
        sourcemap: true,
        minify: false,
      }
    : {
        target: 'es2022',
      },
});
