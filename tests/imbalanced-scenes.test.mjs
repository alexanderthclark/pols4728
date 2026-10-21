import { test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

test('logit animations agree with count-derived odds and plotted coefficients', () => {
  execFileSync(process.execPath, [
    fileURLToPath(new URL('./imbalanced-scene-check.cjs', import.meta.url)),
    fileURLToPath(new URL('../docs/imbalanced-data/', import.meta.url)),
  ], { encoding: 'utf8' });
});
