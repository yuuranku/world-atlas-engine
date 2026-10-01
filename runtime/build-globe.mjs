import {build} from 'esbuild';
import {readFile, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {dirname, resolve} from 'node:path';
const root = dirname(fileURLToPath(import.meta.url));
const destination = resolve(root, '../src/world_atlas/core/web');
await build({
  entryPoints:[resolve(destination, 'globe-entry.js')],
  outfile:resolve(destination, 'globe.js'),
  bundle:true, minify:true, format:'iife', target:'es2020',
  nodePaths:[resolve(root, 'node_modules')], legalComments:'eof',
});
await writeFile(resolve(destination, 'three-license.txt'), await readFile(resolve(root, 'node_modules/three/LICENSE')));
