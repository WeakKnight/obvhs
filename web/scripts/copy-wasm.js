/**
 * Copies WASM artifacts from src/wasm/pkg/ to dist/wasm/ for npm distribution.
 *
 * This is necessary because npm pack in a git sub-directory respects the
 * parent .gitignore, which excludes src/wasm/pkg/. By copying into dist/
 * (which is freshly built and not gitignored at pack time), we ensure the
 * WASM files are included in the published package.
 */
import { mkdirSync, copyFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const src = join(root, 'src', 'wasm', 'pkg');
const dest = join(root, 'dist', 'wasm');

const files = [
  'obvhs_wasm.js',
  'obvhs_wasm.d.ts',
  'obvhs_wasm_bg.wasm',
  'obvhs_wasm_bg.wasm.d.ts',
];

if (!existsSync(src)) {
  console.error(`ERROR: ${src} not found. Run "npm run build:wasm" first.`);
  process.exit(1);
}

mkdirSync(dest, { recursive: true });

for (const file of files) {
  const from = join(src, file);
  const to = join(dest, file);
  if (existsSync(from)) {
    copyFileSync(from, to);
    console.log(`  ✓ ${file}`);
  } else {
    console.warn(`  ⚠ ${file} not found, skipping`);
  }
}

console.log(`\nWASM files copied to dist/wasm/`);
